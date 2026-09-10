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
