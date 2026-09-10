from __future__ import annotations

"""Explicit local-runtime boundary for resident open-weight donors.

Nothing in this module is imported by the metadata inspector. Model loading
requires an explicit opt-in flag at the API boundary and local-only files.
The returned donor is an external, frozen query service; it cannot promote a
Remora candidate or write model weights.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

from .response import DonorResponseRecord
from .activation import DonorActivationRecord


@dataclass(frozen=True)
class DonorRuntimeSpec:
    model_path: str
    runtime_id: str
    device: str = "cpu"
    max_prompt_chars: int = 4096
    max_new_tokens: int = 32
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    allow_model_load: bool = False

    def validate(self) -> Path:
        path = Path(self.model_path).expanduser().resolve()
        if not self.allow_model_load:
            raise PermissionError("donor model loading requires explicit allow_model_load=True")
        if not path.is_dir():
            raise NotADirectoryError(path)
        if not self.runtime_id:
            raise ValueError("runtime_id is required")
        if self.max_prompt_chars <= 0 or self.max_new_tokens <= 0:
            raise ValueError("prompt and generation limits must be positive")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"unsupported runtime dtype {self.dtype}")
        if not (path / "config.json").is_file():
            raise FileNotFoundError(path / "config.json")
        return path


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


class LocalTransformersDonor:
    """A frozen, bounded response interface around one explicitly loaded model."""

    def __init__(self, spec: DonorRuntimeSpec):
        path = spec.validate()
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers is required for the explicit donor runtime") from exc

        self.spec = spec
        self.path = path
        self.config_repairs: list[str] = []
        self.last_activation_capture_count = 0
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(path),
            local_files_only=True,
            trust_remote_code=spec.trust_remote_code,
        )
        config = AutoConfig.from_pretrained(
            str(path),
            local_files_only=True,
            trust_remote_code=spec.trust_remote_code,
        )
        # Nanbeige 4.2's checked-in config has rope_scaling=null while its
        # cached custom modeling code expects a scaling dictionary.  A
        # factor-1 linear scaling is the equivalent compatibility repair; it
        # is recorded by the runtime provenance rather than editing donor
        # files in place.
        if getattr(config, "model_type", None) == "nanbeige":
            rope_scaling = getattr(config, "rope_scaling", None)
            if rope_scaling is None or "type" not in rope_scaling:
                # Transformers 5 normalizes the legacy null/default field to
                # {rope_type: default}; the local Nanbeige remote code still
                # reads the Transformers 4 {type, factor} spelling.
                config.rope_scaling = {"type": "linear", "factor": 1.0}
                self.config_repairs.append("nanbeige_rope_scaling_default_to_legacy_linear_factor_1")
        load_kwargs = {
            "local_files_only": True,
            "trust_remote_code": spec.trust_remote_code,
            "low_cpu_mem_usage": True,
            "torch_dtype": _torch_dtype(spec.dtype),
            "config": config,
        }
        requested_device = torch.device(spec.device)
        if requested_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            load_kwargs["device_map"] = {"": requested_device.index or 0}
        self.model = AutoModelForCausalLM.from_pretrained(str(path), **load_kwargs)
        if requested_device.type != "cuda":
            self.model.to(requested_device)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise TypeError("donor prompt must be text")
        if not prompt or len(prompt) > self.spec.max_prompt_chars:
            raise ValueError("donor prompt is empty or exceeds the configured limit")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        input_length = int(inputs["input_ids"].shape[-1])
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            for _ in range(self.spec.max_new_tokens):
                # Manual greedy decoding is deliberately used instead of the
                # donor's generate() helper. Several resident custom models
                # carry a Transformers-4 cache implementation that fails
                # under newer generation utilities. No KV cache keeps this
                # compatibility path bounded and explicit.
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                logits = output.logits if hasattr(output, "logits") else output[0]
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat((input_ids, next_token), dim=1)
                attention_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=1)
        return self.tokenizer.decode(input_ids[0, input_length:].tolist(), skip_special_tokens=True)

    def capture_activation(self, prompt: str, layer_name: str) -> torch.Tensor:
        """Capture only the final-token output of one explicitly named layer."""

        if not isinstance(prompt, str):
            raise TypeError("donor prompt must be text")
        if not prompt or len(prompt) > self.spec.max_prompt_chars:
            raise ValueError("donor prompt is empty or exceeds the configured limit")
        modules = dict(self.model.named_modules())
        if layer_name not in modules:
            raise KeyError(f"donor layer is not present: {layer_name}")
        captured: list[torch.Tensor] = []

        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(value, torch.Tensor) or value.ndim < 2:
                raise TypeError(f"donor layer {layer_name} did not emit a tensor sequence")
            captured.append(value.detach()[0, -1].cpu().contiguous())

        handle = modules[layer_name].register_forward_hook(hook)
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with torch.inference_mode():
                self.model(
                    **inputs,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError(f"donor layer {layer_name} emitted no captures")
        # Recurrent/looped donor architectures may execute a named layer more
        # than once. The final invocation is the post-loop representation; the
        # record remains explicit about the selected layer and final-token
        # pooling rather than silently concatenating loop states.
        self.last_activation_capture_count = len(captured)
        return captured[-1]

    def activation_records(
        self,
        prompts: Iterable[str],
        *,
        donor_id: str,
        layer_name: str,
        lineage_key: str | None = None,
    ) -> tuple[list[DonorActivationRecord], dict[str, torch.Tensor]]:
        records = []
        activations: dict[str, torch.Tensor] = {}
        lineage = lineage_key or f"{self.spec.runtime_id}:activation-session"
        for index, prompt in enumerate(prompts):
            record_id = f"{self.spec.runtime_id}-activation-{index:04d}"
            activation = self.capture_activation(prompt, layer_name)
            record = DonorActivationRecord.create(
                record_id,
                donor_id,
                layer_name,
                prompt,
                activation,
                lineage_key=lineage,
                runtime={
                    "runtime_id": self.spec.runtime_id,
                    "model_path": str(self.path),
                    "device": str(self.device),
                    "pooling": "final_token",
                    "loop_capture_count": self.last_activation_capture_count,
                    "config_repairs": list(self.config_repairs),
                },
                accepted=True,
            )
            records.append(record)
            activations[record_id] = activation
        return records, activations

    def response_records(
        self,
        prompts: Iterable[str],
        *,
        donor_id: str,
        verifier_id: str,
        verifier: Callable[[str, str], bool] | None = None,
        lineage_key: str | None = None,
    ) -> list[DonorResponseRecord]:
        records = []
        lineage = lineage_key or f"{self.spec.runtime_id}:query-session"
        for index, prompt in enumerate(prompts):
            response = self.generate(prompt)
            passed = bool(verifier(prompt, response)) if verifier is not None else False
            records.append(
                DonorResponseRecord.create(
                    f"{self.spec.runtime_id}-response-{index:04d}",
                    donor_id,
                    prompt,
                    response,
                    lineage_key=lineage,
                    runtime={
                        "runtime_id": self.spec.runtime_id,
                        "model_path": str(self.path),
                        "device": str(self.device),
                        "sampling": "greedy",
                        "max_new_tokens": self.spec.max_new_tokens,
                        "config_repairs": list(self.config_repairs),
                    },
                    verifier={"verifier_id": verifier_id, "passed": passed},
                    accepted=passed,
                )
            )
        return records
