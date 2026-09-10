from __future__ import annotations

"""Bounded, stateful GatedDeltaNet donor organs.

The implementation is intentionally one value head rather than a full Qwen
linear-attention layer.  It accepts already converted projection tensors, so
the source checkpoint can be sliced and width-converted offline without
materializing the rest of the donor block.
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def l2_normalize(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    value = value.float()
    return value * torch.rsqrt(value.square().sum(dim=-1, keepdim=True) + eps)


def causal_depthwise_silu(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Causal depthwise convolution used by the Qwen GDN path.

    ``value`` is ``[batch, time, channels]`` and ``weight`` is
    ``[channels, kernel]``.  The explicit left pad makes the boundary state
    unambiguous and avoids relying on a runtime-specific fused kernel.
    """

    if value.ndim != 3 or weight.ndim != 2 or value.shape[-1] != weight.shape[0]:
        raise ValueError("causal depthwise input/weight geometry mismatch")
    channels, kernel = weight.shape
    transposed = value.float().transpose(1, 2)
    padded = F.pad(transposed, (kernel - 1, 0))
    filtered = F.conv1d(
        padded,
        weight.float().reshape(channels, 1, kernel),
        groups=channels,
    )
    return F.silu(filtered.transpose(1, 2))


def _causal_depthwise_silu_with_state(
    value: torch.Tensor,
    weight: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the same convolution while exposing its finite causal state."""

    channels, kernel = weight.shape
    transposed = value.float().transpose(1, 2)
    history_width = max(kernel - 1, 0)
    if history_width:
        if initial_state is None:
            history = transposed.new_zeros(transposed.shape[0], channels, history_width)
        else:
            expected = (transposed.shape[0], channels, history_width)
            if tuple(initial_state.shape) != expected:
                raise ValueError(f"expected convolution state {expected}, got {tuple(initial_state.shape)}")
            history = initial_state.float().to(transposed.device)
        padded = torch.cat((history, transposed), dim=-1)
        filtered = F.conv1d(
            padded,
            weight.float().reshape(channels, 1, kernel),
            groups=channels,
        )
        next_state = padded[:, :, -history_width:].contiguous()
    else:
        filtered = F.conv1d(
            transposed,
            weight.float().reshape(channels, 1, kernel),
            groups=channels,
        )
        next_state = transposed[:, :, :0].contiguous()
    return F.silu(filtered.transpose(1, 2)), next_state


def _run_gdn_core(
    bus: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
    conv_qkv: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
    """Run only the trained Q/K/V -> delta-state subgraph.

    Keeping this primitive separate lets a core-only graft discard Qwen's
    output gate, normalization, and output projection instead of retaining
    tensors that cannot contribute to the exported recurrent state.
    """

    q = F.linear(bus.float(), q_weight)
    k = F.linear(bus.float(), k_weight)
    v = F.linear(bus.float(), v_weight)
    a = F.linear(bus.float(), a_weight).squeeze(-1)
    b = F.linear(bus.float(), b_weight).squeeze(-1)
    mixed = torch.cat((q, k, v), dim=-1)
    recurrent_state: torch.Tensor | None
    convolution_state: torch.Tensor | None
    if state is None:
        recurrent_state = None
        convolution_state = None
    elif isinstance(state, tuple):
        if len(state) != 2:
            raise ValueError("GDN state tuple must contain recurrent and convolution states")
        recurrent_state, convolution_state = state
    else:
        recurrent_state = state
        convolution_state = None
    mixed, next_convolution_state = _causal_depthwise_silu_with_state(
        mixed,
        conv_qkv,
        convolution_state,
    )
    q, k, v = mixed.split(128, dim=-1)
    q = l2_normalize(q)
    k = l2_normalize(k)
    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    batch = int(bus.shape[0])
    if recurrent_state is None:
        recurrent = torch.zeros(batch, 128, 128, dtype=torch.float32, device=bus.device)
    else:
        expected = (batch, 128, 128)
        if tuple(recurrent_state.shape) != expected:
            raise ValueError(f"expected recurrent state {expected}, got {tuple(recurrent_state.shape)}")
        recurrent = recurrent_state.float().to(bus.device)
    core_values = []
    for index in range(int(bus.shape[1])):
        q_t = q[:, index]
        k_t = k[:, index]
        v_t = v[:, index]
        recurrent = recurrent * g[:, index].exp().view(batch, 1, 1)
        kv_mem = (recurrent * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta[:, index].view(batch, 1)
        recurrent = recurrent + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_values.append((recurrent * q_t.unsqueeze(-1)).sum(dim=-2))
    core = torch.stack(core_values, dim=1)
    details = {
        "query": q,
        "key": k,
        "value": v,
        "beta": beta,
        "log_decay": g,
        "core": core,
    }
    return core, (recurrent, next_convolution_state), details


class QwenGatedDeltaCoreOrgan(nn.Module):
    """The minimal frozen Qwen GDN recurrent subgraph.

    This organ intentionally stops before ``z`` gating, RMS normalization,
    and the Qwen output projection.  Those tensors belong to the full head,
    but are not needed by the state computation or by the core socket.  The
    resulting resident payload is therefore the smallest v0 closed subgraph
    selected so far: Q/K/V, beta/decay projections, causal convolution, and
    the recurrent delta update.
    """

    interface_version = "qwen-gdn-core-organ-v1"

    def __init__(
        self,
        *,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        a_weight: torch.Tensor,
        b_weight: torch.Tensor,
        conv_qkv: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        donor_variant: str = "actual",
    ):
        super().__init__()
        matrices = {
            "q_weight": q_weight,
            "k_weight": k_weight,
            "v_weight": v_weight,
            "a_weight": a_weight,
            "b_weight": b_weight,
            "conv_qkv": conv_qkv,
            "A_log": A_log,
            "dt_bias": dt_bias,
        }
        for name, tensor in matrices.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            self.register_buffer(name, tensor.detach().cpu().float().contiguous())
        if tuple(self.q_weight.shape) != tuple(self.k_weight.shape) or tuple(self.q_weight.shape) != tuple(self.v_weight.shape):
            raise ValueError("q/k/v weights must have the same one-head geometry")
        if self.q_weight.ndim != 2 or self.q_weight.shape[0] != 128:
            raise ValueError("the v0 core organ is fixed to a 128-wide Qwen head")
        if tuple(self.a_weight.shape) != (1, self.q_weight.shape[1]) or tuple(self.b_weight.shape) != tuple(self.a_weight.shape):
            raise ValueError("a/b weights must be one scalar projection per selected head")
        if tuple(self.conv_qkv.shape[:1]) != (384,) or self.conv_qkv.ndim != 2:
            raise ValueError("conv_qkv must have shape [384, kernel]")
        if self.A_log.numel() != 1 or self.dt_bias.numel() != 1:
            raise ValueError("one selected head needs one A_log and dt_bias scalar")
        self.donor_variant = str(donor_variant)
        self.bus_dim = int(self.q_weight.shape[1])
        self.head_dim = 128
        self.kernel_size = int(self.conv_qkv.shape[1])
        self.state_shape = [self.head_dim, self.head_dim]

    @property
    def donor_parameter_count(self) -> int:
        return int(sum(buffer.numel() for buffer in self.buffers()))

    @property
    def trainable_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad))

    def interface_signature(self) -> dict[str, Any]:
        return {
            "version": self.interface_version,
            "input": ["batch", "time", self.bus_dim],
            "output": ["batch", "time", self.head_dim],
            "state": {
                "recurrent": ["batch", self.head_dim, self.head_dim],
                "convolution": ["batch", 384, max(self.kernel_size - 1, 0)],
            },
            "state_owner": "foreign-gdn-core-organ",
            "donor_core_frozen": True,
            "donor_variant": self.donor_variant,
            "kernel_size": self.kernel_size,
        }

    def forward(
        self,
        bus: torch.Tensor,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        return_state: bool = False,
        return_intermediates: bool = False,
    ):
        if bus.ndim != 3 or bus.shape[-1] != self.bus_dim:
            raise ValueError(f"expected [batch, time, {self.bus_dim}] input")
        core, next_state, details = _run_gdn_core(
            bus,
            self.q_weight,
            self.k_weight,
            self.v_weight,
            self.a_weight,
            self.b_weight,
            self.conv_qkv,
            self.A_log,
            self.dt_bias,
            state,
        )
        if return_intermediates:
            if return_state:
                return core, next_state, details
            return core, details
        if return_state:
            return core, next_state
        return core

    def forward_core(
        self,
        bus: torch.Tensor,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        return_state: bool = False,
        return_intermediates: bool = False,
    ):
        return self.forward(
            bus,
            state,
            return_state=return_state,
            return_intermediates=return_intermediates,
        )


class TrainableGatedDeltaCoreOrgan(QwenGatedDeltaCoreOrgan):
    """Fresh same-geometry GDN core used as a relearning control.

    The computation and tensor geometry are identical to the donor core, but
    every core tensor is freshly initialized and trainable.  Keeping this
    control next to the frozen organ prevents a fresh same-mechanism result
    from being mistaken for a native Remora implementation or for a donor
    transplant.
    """

    interface_version = "fresh-gdn-core-organ-v1"
    core_tensor_names = (
        "q_weight",
        "k_weight",
        "v_weight",
        "a_weight",
        "b_weight",
        "conv_qkv",
        "A_log",
        "dt_bias",
    )

    def __init__(
        self,
        *,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        a_weight: torch.Tensor,
        b_weight: torch.Tensor,
        conv_qkv: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        donor_variant: str = "fresh",
    ):
        super().__init__(
            q_weight=q_weight,
            k_weight=k_weight,
            v_weight=v_weight,
            a_weight=a_weight,
            b_weight=b_weight,
            conv_qkv=conv_qkv,
            A_log=A_log,
            dt_bias=dt_bias,
            donor_variant=donor_variant,
        )
        for name in self.core_tensor_names:
            value = getattr(self, name).detach().clone()
            del self._buffers[name]
            self.register_parameter(name, nn.Parameter(value))

    @property
    def donor_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))


class QwenGatedDeltaHead(nn.Module):
    """A frozen one-head Qwen GatedDeltaNet organ.

    Tensor conventions match ``torch.nn.functional.linear``:

    - q/k/v/z/a/b weights are ``[output, input_bus]``;
    - ``conv_qkv`` is ``[384, kernel]`` for one 128-dimensional value head;
    - ``out_weight`` is ``[output_bus, 128]``;
    - ``A_log``, ``dt_bias`` and ``norm_weight`` retain donor values.

    The recurrent state is owned by the organ and has shape
    ``[batch, 128, 128]``.  The causal-convolution history is also state and
    has shape ``[batch, 384, kernel-1]``.  ``return_state=True`` returns the
    pair ``(recurrent_state, convolution_state)`` so chunked execution does
    not silently lose the donor's short causal context.
    """

    interface_version = "qwen-gdn-value-head-organ-v1"

    def __init__(
        self,
        *,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        z_weight: torch.Tensor,
        a_weight: torch.Tensor,
        b_weight: torch.Tensor,
        conv_qkv: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        out_weight: torch.Tensor,
        donor_variant: str = "actual",
    ):
        super().__init__()
        matrices = {
            "q_weight": q_weight,
            "k_weight": k_weight,
            "v_weight": v_weight,
            "z_weight": z_weight,
            "a_weight": a_weight,
            "b_weight": b_weight,
            "conv_qkv": conv_qkv,
            "A_log": A_log,
            "dt_bias": dt_bias,
            "norm_weight": norm_weight,
            "out_weight": out_weight,
        }
        for name, tensor in matrices.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            self.register_buffer(name, tensor.detach().cpu().float().contiguous())
        if tuple(self.q_weight.shape) != tuple(self.k_weight.shape) or tuple(self.q_weight.shape) != tuple(self.v_weight.shape):
            raise ValueError("q/k/v weights must have the same one-head geometry")
        if tuple(self.q_weight.shape) != (128, self.q_weight.shape[1]):
            raise ValueError("the v0 organ is fixed to a 128-wide Qwen head")
        if tuple(self.z_weight.shape) != tuple(self.q_weight.shape):
            raise ValueError("z weight must match the selected value-head width")
        if tuple(self.a_weight.shape) != (1, self.q_weight.shape[1]) or tuple(self.b_weight.shape) != tuple(self.a_weight.shape):
            raise ValueError("a/b weights must be one scalar projection per selected head")
        if tuple(self.conv_qkv.shape[:1]) != (384,) or self.conv_qkv.ndim != 2:
            raise ValueError("conv_qkv must have shape [384, kernel]")
        if self.A_log.numel() != 1 or self.dt_bias.numel() != 1:
            raise ValueError("one selected head needs one A_log and dt_bias scalar")
        if tuple(self.norm_weight.shape) != (128,):
            raise ValueError("norm weight must have shape [128]")
        if self.out_weight.ndim != 2 or self.out_weight.shape[1] != 128:
            raise ValueError("out weight must have shape [output_bus, 128]")
        self.donor_variant = str(donor_variant)
        self.bus_dim = int(self.q_weight.shape[1])
        self.output_dim = int(self.out_weight.shape[0])
        self.head_dim = 128
        self.kernel_size = int(self.conv_qkv.shape[1])
        self.state_shape = [self.head_dim, self.head_dim]

    @property
    def donor_parameter_count(self) -> int:
        return int(sum(buffer.numel() for buffer in self.buffers()))

    @property
    def trainable_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad))

    def interface_signature(self) -> dict[str, Any]:
        return {
            "version": self.interface_version,
            "input": ["batch", "time", self.bus_dim],
            "output": ["batch", "time", self.output_dim],
            "state": {
                "recurrent": ["batch", self.head_dim, self.head_dim],
                "convolution": ["batch", 384, max(self.kernel_size - 1, 0)],
            },
            "state_owner": "foreign-gdn-head-organ",
            "donor_core_frozen": True,
            "donor_variant": self.donor_variant,
            "kernel_size": self.kernel_size,
        }

    def _project(self, bus: torch.Tensor) -> tuple[torch.Tensor, ...]:
        q = F.linear(bus.float(), self.q_weight)
        k = F.linear(bus.float(), self.k_weight)
        v = F.linear(bus.float(), self.v_weight)
        z = F.linear(bus.float(), self.z_weight)
        a = F.linear(bus.float(), self.a_weight).squeeze(-1)
        b = F.linear(bus.float(), self.b_weight).squeeze(-1)
        return q, k, v, z, a, b

    def forward_core(
        self,
        bus: torch.Tensor,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        return_state: bool = False,
        return_intermediates: bool = False,
    ):
        """Run the same recurrent subgraph without the full-head output path."""

        if bus.ndim != 3 or bus.shape[-1] != self.bus_dim:
            raise ValueError(f"expected [batch, time, {self.bus_dim}] input")
        core, next_state, details = _run_gdn_core(
            bus,
            self.q_weight,
            self.k_weight,
            self.v_weight,
            self.a_weight,
            self.b_weight,
            self.conv_qkv,
            self.A_log,
            self.dt_bias,
            state,
        )
        if return_intermediates:
            if return_state:
                return core, next_state, details
            return core, details
        if return_state:
            return core, next_state
        return core

    def forward(
        self,
        bus: torch.Tensor,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        return_state: bool = False,
        return_intermediates: bool = False,
    ):
        if bus.ndim != 3 or bus.shape[-1] != self.bus_dim:
            raise ValueError(f"expected [batch, time, {self.bus_dim}] input")
        recurrent_state: torch.Tensor | None
        convolution_state: torch.Tensor | None
        if state is None:
            recurrent_state = None
            convolution_state = None
        elif isinstance(state, tuple):
            if len(state) != 2:
                raise ValueError("GDN state tuple must contain recurrent and convolution states")
            recurrent_state, convolution_state = state
        else:
            # Backward-compatible recurrent-only state; the missing causal
            # context is treated as zeros and is explicit in the result.
            recurrent_state = state
            convolution_state = None
        q, k, v, z, a, b = self._project(bus)
        mixed = torch.cat((q, k, v), dim=-1)
        mixed, next_convolution_state = _causal_depthwise_silu_with_state(
            mixed,
            self.conv_qkv,
            convolution_state,
        )
        q, k, v = mixed.split(self.head_dim, dim=-1)
        q = l2_normalize(q)
        k = l2_normalize(k)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        batch = int(bus.shape[0])
        if recurrent_state is None:
            recurrent = torch.zeros(batch, self.head_dim, self.head_dim, dtype=torch.float32, device=bus.device)
        else:
            expected = (batch, self.head_dim, self.head_dim)
            if tuple(recurrent_state.shape) != expected:
                raise ValueError(f"expected recurrent state {expected}, got {tuple(recurrent_state.shape)}")
            recurrent = recurrent_state.float().to(bus.device)
        core_values = []
        for index in range(int(bus.shape[1])):
            q_t = q[:, index]
            k_t = k[:, index]
            v_t = v[:, index]
            recurrent = recurrent * g[:, index].exp().view(batch, 1, 1)
            kv_mem = (recurrent * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta[:, index].view(batch, 1)
            recurrent = recurrent + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            core_values.append((recurrent * q_t.unsqueeze(-1)).sum(dim=-2))
        core = torch.stack(core_values, dim=1)
        variance = core.square().mean(dim=-1, keepdim=True)
        normalized = self.norm_weight.view(1, 1, -1) * core * torch.rsqrt(variance + 1e-6)
        normalized = normalized * F.silu(z.float())
        output = F.linear(normalized, self.out_weight)
        if return_intermediates:
            details = {
                "query": q,
                "key": k,
                "value": v,
                "beta": beta,
                "log_decay": g,
                "core": core,
                "normalized": normalized,
            }
            if return_state:
                return output, (recurrent, next_convolution_state), details
            return output, details
        if return_state:
            return output, (recurrent, next_convolution_state)
        return output


class QwenGatedDeltaCoreSocket(nn.Module):
    """Expose a donor GDN state core through a real Remora branch socket.

    The full Qwen head output projection is not assumed to be a compatible
    Remora representation.  This socket deliberately ends at the trained
    recurrent ``core`` state, then exports it through a fixed geometry port
    into a Remora block's ``plastic`` branch.  The donor core and the base
    port are frozen by default; optional low-rank repairs are the only
    trainable parameters.

    ``interface_permutation`` is an explicit shifted-interface control.  It
    changes only the fixed input port and therefore cannot be mistaken for a
    different donor core.
    """

    interface_version = "qwen-gdn-core-socket-v1"

    def __init__(
        self,
        organ: QwenGatedDeltaHead | QwenGatedDeltaCoreOrgan,
        d_model: int,
        *,
        repair_rank: int = 0,
        interface_permutation: torch.Tensor | list[int] | tuple[int, ...] | None = None,
        interface_scale: float = 1.0,
        input_projection: torch.Tensor | None = None,
    ):
        super().__init__()
        if not isinstance(organ, (QwenGatedDeltaHead, QwenGatedDeltaCoreOrgan)):
            raise TypeError("core socket requires a Qwen GDN head or core organ")
        if int(repair_rank) < 0:
            raise ValueError("repair_rank must be non-negative")
        self.organ = organ
        self.d_model = int(d_model)
        self.bus_dim = int(organ.bus_dim)
        self.core_dim = int(organ.head_dim)
        self.repair_rank = int(repair_rank)
        if input_projection is not None:
            if interface_permutation is not None:
                raise ValueError("input_projection and interface_permutation are mutually exclusive")
            projection = input_projection.detach().cpu().float().contiguous()
            expected = (self.bus_dim, self.d_model)
            if tuple(projection.shape) != expected:
                raise ValueError(f"input_projection must have shape {expected}, got {tuple(projection.shape)}")
            permutation = None
            self.input_projection_kind = "fixed_analytic"
        else:
            if self.bus_dim > self.d_model:
                raise ValueError("the identity core socket requires bus_dim <= d_model")
            if interface_permutation is None:
                permutation = tuple(range(self.bus_dim))
            else:
                permutation = tuple(int(index) for index in interface_permutation)
                if len(permutation) != self.bus_dim or set(permutation) != set(range(self.bus_dim)):
                    raise ValueError("interface_permutation must be a permutation of bus coordinates")
            self.input_projection_kind = "identity_prefix"
        self.interface_permutation = permutation
        self.interface_scale = float(interface_scale)
        self.input_port = nn.Linear(self.d_model, self.bus_dim, bias=False)
        self.output_port = nn.Linear(self.core_dim, self.d_model, bias=False)
        with torch.no_grad():
            if input_projection is None:
                self.input_port.weight.zero_()
                self.input_port.weight[:, : self.bus_dim] = torch.eye(self.bus_dim)[list(permutation)] * self.interface_scale
            else:
                self.input_port.weight.copy_(projection * self.interface_scale)
            self.output_port.weight.zero_()
            copied = min(self.core_dim, self.d_model)
            self.output_port.weight[:copied, :copied] = torch.eye(copied)
        for parameter in self.input_port.parameters():
            parameter.requires_grad = False
        for parameter in self.output_port.parameters():
            parameter.requires_grad = False
        if self.repair_rank:
            self.input_repair_down = nn.Linear(self.d_model, self.repair_rank, bias=False)
            self.input_repair_up = nn.Linear(self.repair_rank, self.bus_dim, bias=False)
            self.output_repair_down = nn.Linear(self.core_dim, self.repair_rank, bias=False)
            self.output_repair_up = nn.Linear(self.repair_rank, self.d_model, bias=False)
            nn.init.normal_(self.input_repair_down.weight, std=self.d_model ** -0.5)
            nn.init.normal_(self.output_repair_down.weight, std=self.core_dim ** -0.5)
            nn.init.zeros_(self.input_repair_up.weight)
            nn.init.zeros_(self.output_repair_up.weight)
        self.last_core: torch.Tensor | None = None
        self.last_value: torch.Tensor | None = None
        self.last_organ_output: torch.Tensor | None = None
        self.last_output: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError(f"expected [batch, time, {self.d_model}] hidden input")
        bus = self.input_port(hidden.float())
        if self.repair_rank:
            bus = bus + self.input_repair_up(self.input_repair_down(hidden.float()))
        organ_output, _state, details = self.organ.forward_core(
            bus,
            return_state=True,
            return_intermediates=True,
        )
        core = details["core"]
        output = self.output_port(core)
        if self.repair_rank:
            output = output + self.output_repair_up(self.output_repair_down(core))
        self.last_core = core
        self.last_value = details["value"]
        self.last_organ_output = organ_output
        self.last_output = output
        return output

    def port_parameters(self) -> list[nn.Parameter]:
        if not self.repair_rank:
            return []
        return [
            *self.input_repair_down.parameters(),
            *self.input_repair_up.parameters(),
            *self.output_repair_down.parameters(),
            *self.output_repair_up.parameters(),
        ]

    @property
    def donor_parameter_count(self) -> int:
        return self.organ.donor_parameter_count

    @property
    def trainable_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.port_parameters()))

    def interface_signature(self) -> dict[str, Any]:
        return {
            "version": self.interface_version,
            "input": ["batch", "time", self.d_model],
            "output": ["batch", "time", self.d_model],
            "exported_representation": ["batch", "time", self.core_dim],
            "state": self.organ.interface_signature()["state"],
            "state_owner": "foreign-gdn-core-organ",
            "donor_core_frozen": not any(parameter.requires_grad for parameter in self.organ.parameters()),
            "donor_variant": self.organ.donor_variant,
            "interface_permutation": list(self.interface_permutation) if self.interface_permutation is not None else None,
            "interface_scale": self.interface_scale,
            "input_projection_kind": self.input_projection_kind,
            "input_projection_shape": list(self.input_port.weight.shape),
            "repair_rank": self.repair_rank,
            "trainable_parameters": self.trainable_parameter_count,
        }


def make_gdn_variant(tensors: dict[str, torch.Tensor], variant: str, *, seed: int = 0) -> dict[str, torch.Tensor]:
    """Create matched actual/random/shuffled/zero controls without mutation."""

    if variant not in {"actual", "random", "shuffled", "zero"}:
        raise ValueError(variant)
    result = {name: value.detach().cpu().float().contiguous() for name, value in tensors.items()}
    if variant == "actual":
        return {name: value.clone() for name, value in result.items()}
    if variant == "zero":
        return {name: torch.zeros_like(value) for name, value in result.items()}
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if variant == "random":
        return {
            name: torch.randn(value.shape, generator=generator) * value.std(unbiased=False).clamp_min(1e-6)
            for name, value in result.items()
        }
    output: dict[str, torch.Tensor] = {}
    for name, value in result.items():
        flat = value.reshape(-1)
        output[name] = flat[torch.randperm(flat.numel(), generator=generator)].reshape(value.shape).contiguous()
    return output
