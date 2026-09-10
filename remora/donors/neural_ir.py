from __future__ import annotations

"""A deliberately small neural intermediate representation.

This is not a compiler.  It is a provenance-bearing description of the
primitive operations needed to reproduce the first real donor organ.  The IR
is kept small until a second donor family proves that more machinery is
needed.
"""

from dataclasses import asdict, dataclass, field
from typing import Any


PRIMITIVES = {
    "affine",
    "normalization",
    "gating",
    "recurrence",
    "attention_mixing",
    "elementwise_nonlinearity",
    "elementwise_mul",
    "routing",
    "residual_add",
    "convolution",
}


@dataclass(frozen=True)
class NeuralIROp:
    op_id: str
    primitive: str
    inputs: list[str]
    output: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.primitive not in PRIMITIVES:
            raise ValueError(f"unknown neural IR primitive: {self.primitive}")
        if not self.op_id or not self.output:
            raise ValueError("neural IR operations need non-empty ids")


@dataclass
class NeuralComponentIR:
    component_id: str
    source_model: str
    source_revision: str
    import_mode: str
    input_contract: dict[str, Any]
    output_contract: dict[str, Any]
    state_contract: dict[str, Any]
    parameter_tensors: list[dict[str, Any]]
    operations: list[NeuralIROp]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.component_id or not self.source_model or not self.source_revision:
            raise ValueError("neural IR needs donor identity and revision")
        values = set(self.input_contract.get("names", []))
        for operation in self.operations:
            operation.validate()
            missing = [name for name in operation.inputs if name not in values]
            if missing:
                raise ValueError(f"IR operation {operation.op_id} references unknown values {missing}")
            if operation.output in values:
                raise ValueError(f"IR output value is redefined: {operation.output}")
            values.add(operation.output)
        outputs = set(self.output_contract.get("names", []))
        if not outputs.issubset(values):
            raise ValueError(f"IR output contract references unknown values: {sorted(outputs - values)}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def qwen_shared_expert_ir(
    *,
    component_id: str = "qwen3.8.language.layer0.shared_expert",
    source_revision: str,
    tensor_names: list[str],
) -> NeuralComponentIR:
    """Describe Qwen's shared expert and its optional scalar gate."""

    has_scalar_gate = any(name.endswith("shared_expert_gate.weight") for name in tensor_names)
    operations = [
        NeuralIROp("gate_affine", "affine", ["hidden"], "gate_pre", {"tensor": next(name for name in tensor_names if name.endswith("shared_expert.gate_proj.weight")), "bias": False}),
        NeuralIROp("up_affine", "affine", ["hidden"], "up_pre", {"tensor": next(name for name in tensor_names if name.endswith("shared_expert.up_proj.weight")), "bias": False}),
        NeuralIROp("silu_gate", "elementwise_nonlinearity", ["gate_pre"], "gate", {"function": "silu"}),
        NeuralIROp("gated_product", "elementwise_mul", ["gate", "up_pre"], "hidden_product"),
        NeuralIROp("down_affine", "affine", ["hidden_product"], "core_output", {"tensor": next(name for name in tensor_names if name.endswith("shared_expert.down_proj.weight")), "bias": False}),
    ]
    output_names = ["core_output"]
    if has_scalar_gate:
        gate_tensor = next(name for name in tensor_names if name.endswith("shared_expert_gate.weight"))
        operations.extend(
            [
                NeuralIROp("shared_gate_affine", "affine", ["hidden"], "shared_gate_pre", {"tensor": gate_tensor, "bias": False, "output_width": 1}),
                NeuralIROp("shared_gate_sigmoid", "gating", ["shared_gate_pre"], "shared_gate", {"function": "sigmoid"}),
                NeuralIROp("shared_output_gate", "elementwise_mul", ["core_output", "shared_gate"], "output"),
            ]
        )
        output_names = ["output"]
    else:
        operations.append(NeuralIROp("ungated_output", "residual_add", ["core_output"], "output", {"identity_residual": False}))
    ir = NeuralComponentIR(
        component_id=component_id,
        source_model="Qwen/Qwen3.8-Flash-Next",
        source_revision=source_revision,
        import_mode="submodule_graft",
        input_contract={"names": ["hidden"], "shape": ["...", 2560], "dtype": "BF16-compatible"},
        output_contract={"names": output_names, "shape": ["...", 2560], "dtype": "BF16-compatible"},
        state_contract={"kind": "stateless", "owned_state": None},
        parameter_tensors=[{"name": name} for name in tensor_names],
        operations=operations,
        metadata={
            "architecture_family": "shared_swiglu_expert",
            "activation": "silu",
            "reference_implementation": "Qwen4ExpTextMLP plus shared_expert_gate from official Transformers implementation",
            "conversion_status": "mechanism_represented; numerical equivalence must be measured separately",
        },
    )
    ir.validate()
    return ir


def qwen_gated_delta_head_ir(
    *,
    component_id: str,
    source_revision: str,
    tensor_names: list[str],
    layer: int,
    value_head: int,
    input_width: int = 2560,
    output_width: int = 96,
) -> NeuralComponentIR:
    """Describe the minimal primitive graph for one converted GDN head."""

    if len(tensor_names) != 11:
        raise ValueError("GDN IR tensor_names must be q,k,v,z,a,b,conv,A_log,dt_bias,norm,out")
    q_name, k_name, v_name, z_name, a_name, b_name, conv_name, A_name, dt_name, norm_name, out_name = tensor_names

    operations = [
        NeuralIROp("q_projection", "affine", ["hidden"], "q_pre", {"tensor": q_name, "bias": False}),
        NeuralIROp("k_projection", "affine", ["hidden"], "k_pre", {"tensor": k_name, "bias": False}),
        NeuralIROp("v_projection", "affine", ["hidden"], "v_pre", {"tensor": v_name, "bias": False}),
        NeuralIROp("z_projection", "affine", ["hidden"], "z", {"tensor": z_name, "bias": False}),
        NeuralIROp("beta_projection", "affine", ["hidden"], "beta_pre", {"tensor": b_name, "bias": False}),
        NeuralIROp("decay_projection", "affine", ["hidden"], "decay_pre", {"tensor": a_name, "bias": False}),
        NeuralIROp("causal_qkv_convolution", "convolution", ["q_pre", "k_pre", "v_pre"], "qkv", {"tensor": conv_name, "causal": True, "depthwise": True}),
        NeuralIROp("q_l2_normalization", "normalization", ["qkv"], "q", {"kind": "l2", "eps": 1e-6, "split": "q"}),
        NeuralIROp("k_l2_normalization", "normalization", ["qkv"], "k", {"kind": "l2", "eps": 1e-6, "split": "k"}),
        NeuralIROp("value_split", "residual_add", ["qkv"], "v", {"identity_residual": False, "split": "v"}),
        NeuralIROp("beta_gate", "gating", ["beta_pre"], "beta", {"function": "sigmoid"}),
        NeuralIROp("log_decay", "gating", ["decay_pre"], "g", {"function": "-exp(A_log)*softplus(a+dt_bias)", "A_log": A_name, "dt_bias": dt_name}),
        NeuralIROp("gated_delta_state", "recurrence", ["q", "k", "v", "beta", "g"], "core", {"state_shape": [128, 128], "update": "delta_rule"}),
        NeuralIROp("gated_rms_norm", "normalization", ["core", "z"], "normalized", {"kind": "rms_then_silu_gate", "eps": 1e-6, "weight": norm_name}),
        NeuralIROp("output_projection", "affine", ["normalized"], "output", {"tensor": out_name, "bias": False}),
    ]
    ir = NeuralComponentIR(
        component_id=component_id,
        source_model="Qwen/Qwen3.8-Flash-Next",
        source_revision=source_revision,
        import_mode="submodule_subspace_graft",
        input_contract={"names": ["hidden"], "shape": ["...", input_width], "dtype": "BF16-compatible"},
        output_contract={"names": ["output"], "shape": ["...", output_width], "dtype": "FP32-compatible"},
        state_contract={"kind": "recurrent", "owned_state": {"recurrent": ["batch", 128, 128], "convolution": ["batch", 384, 3]}, "reset": "zero_or_supplied"},
        parameter_tensors=[{"name": name} for name in tensor_names],
        operations=operations,
        metadata={
            "architecture_family": "gated_deltanet_linear_attention",
            "layer": int(layer),
            "value_head": int(value_head),
            "conversion_status": "primitive graph represented; exact numerical and width-conversion checks are separate experiment outputs",
        },
    )
    ir.validate()
    return ir


def qwen_gated_delta_core_ir(
    *,
    component_id: str,
    source_revision: str,
    tensor_names: list[str],
    layer: int,
    value_head: int,
    input_width: int = 96,
    output_width: int = 128,
    head_dim: int = 128,
    conv_channels: int = 384,
    conv_kernel: int = 4,
) -> NeuralComponentIR:
    """Describe the smallest closed Qwen GDN recurrent core used by the graft.

    Unlike :func:`qwen_gated_delta_head_ir`, this graph intentionally stops at
    the recurrent state readout.  Qwen's ``z`` gate, RMS normalization, and
    output projection are separate co-adapted machinery and are not silently
    counted as part of the transplanted organ.
    """

    if len(tensor_names) != 8:
        raise ValueError("GDN core IR tensor_names must be q,k,v,a,b,conv,A_log,dt_bias")
    q_name, k_name, v_name, a_name, b_name, conv_name, A_name, dt_name = tensor_names
    if input_width <= 0 or output_width <= 0 or head_dim <= 0:
        raise ValueError("GDN core IR widths must be positive")
    if output_width != head_dim:
        raise ValueError("v0 GDN core output width must equal its value-head dimension")
    if conv_channels != 3 * head_dim:
        raise ValueError("GDN causal-convolution channels must contain q, k, and v")
    if conv_kernel <= 0:
        raise ValueError("GDN convolution kernel must be positive")

    operations = [
        NeuralIROp("q_projection", "affine", ["hidden"], "q_pre", {"tensor": q_name, "bias": False}),
        NeuralIROp("k_projection", "affine", ["hidden"], "k_pre", {"tensor": k_name, "bias": False}),
        NeuralIROp("v_projection", "affine", ["hidden"], "v_pre", {"tensor": v_name, "bias": False}),
        NeuralIROp("beta_projection", "affine", ["hidden"], "beta_pre", {"tensor": b_name, "bias": False}),
        NeuralIROp("decay_projection", "affine", ["hidden"], "decay_pre", {"tensor": a_name, "bias": False}),
        NeuralIROp(
            "causal_qkv_convolution",
            "convolution",
            ["q_pre", "k_pre", "v_pre"],
            "qkv",
            {"tensor": conv_name, "causal": True, "depthwise": True, "channels": conv_channels, "kernel": conv_kernel},
        ),
        NeuralIROp("q_l2_normalization", "normalization", ["qkv"], "q", {"kind": "l2", "eps": 1e-6, "split": "q", "width": head_dim}),
        NeuralIROp("k_l2_normalization", "normalization", ["qkv"], "k", {"kind": "l2", "eps": 1e-6, "split": "k", "width": head_dim}),
        NeuralIROp("value_split", "residual_add", ["qkv"], "v", {"identity_residual": False, "split": "v", "width": head_dim}),
        NeuralIROp("beta_gate", "gating", ["beta_pre"], "beta", {"function": "sigmoid"}),
        NeuralIROp(
            "log_decay",
            "gating",
            ["decay_pre"],
            "g",
            {"function": "-exp(A_log)*softplus(a+dt_bias)", "A_log": A_name, "dt_bias": dt_name},
        ),
        NeuralIROp(
            "gated_delta_state",
            "recurrence",
            ["q", "k", "v", "beta", "g"],
            "core",
            {"state_shape": [head_dim, head_dim], "update": "delta_rule", "state_owner": "component"},
        ),
    ]
    ir = NeuralComponentIR(
        component_id=component_id,
        source_model="Qwen/Qwen3.8-Flash-Next",
        source_revision=source_revision,
        import_mode="submodule_subspace_graft_core_socket",
        input_contract={"names": ["hidden"], "shape": ["...", input_width], "dtype": "BF16-compatible"},
        output_contract={"names": ["core"], "shape": ["...", output_width], "dtype": "FP32-compatible"},
        state_contract={
            "kind": "recurrent",
            "owned_state": {
                "recurrent": ["batch", head_dim, head_dim],
                "convolution": ["batch", conv_channels, max(conv_kernel - 1, 0)],
            },
            "reset": "zero_or_supplied",
        },
        parameter_tensors=[{"name": name} for name in tensor_names],
        operations=operations,
        metadata={
            "architecture_family": "gated_deltanet_linear_attention",
            "layer": int(layer),
            "value_head": int(value_head),
            "closed_subgraph": "q/k/v + beta/decay projections + causal qkv convolution + delta recurrence",
            "excluded_coadapted_tensors": ["z_weight", "norm_weight", "out_weight"],
            "conversion_status": "primitive graph represented; exact core equivalence and socket utility are separate experiment outputs",
        },
    )
    ir.validate()
    return ir
