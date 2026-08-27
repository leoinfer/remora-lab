//! HAR source compiler: lex -> parse -> typed semantics -> logical IR ->
//! optimized physical IR -> immutable Rust execution plan.

use har_decode_control::language::{
    DecodeObjective, DecodePolicyPlan, Epoch as ControlEpoch, FallbackPolicy, HorizonMode,
    ImmutableExecutionPlan, MemoryTier as ControlTier, ModelIdentity as ControlModelIdentity,
    ResourceCost, SamplingLocation, TelemetryRequirements,
};
use har_lang_ast::Program;
use har_lang_diagnostics::Diagnostic;
use har_lang_lexer::lex;
use har_lang_parser::parse;
use har_lang_semantics::v0::{
    analyze, KernelRequirement, MemoryTier, QuantFormat, TypedProgram, ValueAuthority,
};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalNode {
    pub kind: String,
    pub name: String,
    pub dependencies: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalIr {
    pub schema: String,
    pub nodes: Vec<LogicalNode>,
}

#[derive(Clone, Debug)]
pub struct CompiledProgram {
    pub source_name: String,
    pub typed: TypedProgram,
    pub logical_ir: LogicalIr,
    pub physical_plan: ImmutableExecutionPlan,
    pub source_digest: String,
}

pub fn compile_source(
    source_name: impl Into<String>,
    source: &str,
) -> Result<CompiledProgram, Vec<Diagnostic>> {
    let source_name = source_name.into();
    let tokens = lex(source)?;
    let program = parse(&tokens, source_name.clone())?;
    compile_ast(program, source)
}

pub fn compile_file(path: impl AsRef<Path>) -> Result<CompiledProgram, Vec<Diagnostic>> {
    let path = path.as_ref();
    let source = match fs::read_to_string(path) {
        Ok(source) => source,
        Err(error) => {
            return Err(vec![Diagnostic::error(
                "C0001",
                format!("cannot read {}: {error}", path.display()),
                None,
            )])
        }
    };
    compile_source(path.display().to_string(), &source)
}

pub fn compile_ast(program: Program, source: &str) -> Result<CompiledProgram, Vec<Diagnostic>> {
    let typed = analyze(&program)?;
    let logical_ir = lower_logical(&typed);
    let physical_plan = lower_physical(&typed)
        .map_err(|message| vec![Diagnostic::error("C2001", message, None)])?;
    Ok(CompiledProgram {
        source_name: program.source_name,
        typed,
        logical_ir,
        physical_plan,
        source_digest: digest(source),
    })
}

fn lower_logical(program: &TypedProgram) -> LogicalIr {
    let mut nodes = Vec::new();
    nodes.push(LogicalNode {
        kind: "target".into(),
        name: program.target.name.clone(),
        dependencies: Vec::new(),
    });
    nodes.push(LogicalNode {
        kind: "model".into(),
        name: program.model.name.clone(),
        dependencies: Vec::new(),
    });
    for tier in &program.tiers {
        nodes.push(LogicalNode {
            kind: "memory_tier".into(),
            name: tier.name.clone(),
            dependencies: Vec::new(),
        });
    }
    for tensor in &program.tensors {
        nodes.push(LogicalNode {
            kind: "tensor".into(),
            name: tensor.name.clone(),
            dependencies: vec![format!("tier:{}", tensor.tier_name)],
        });
    }
    for phase in &program.phases {
        nodes.push(LogicalNode {
            kind: "phase".into(),
            name: phase.name.clone(),
            dependencies: phase.dependencies.clone(),
        });
    }
    nodes.push(LogicalNode {
        kind: "quality".into(),
        name: program.quality.name.clone(),
        dependencies: Vec::new(),
    });
    nodes.push(LogicalNode {
        kind: "decode_policy".into(),
        name: program.decode.name.clone(),
        dependencies: vec!["quality".into(), "model".into()],
    });
    LogicalIr {
        schema: "har-logical-ir/v1".into(),
        nodes,
    }
}

fn lower_physical(program: &TypedProgram) -> Result<ImmutableExecutionPlan, String> {
    let identity = ControlModelIdentity::new(
        program.model.model_sha256.clone(),
        program.model.build_hash.clone(),
        program.model.config_hash.clone(),
        ControlEpoch(program.model.epoch.0),
    );
    let sampling = match program.decode.sampling_location.as_str() {
        "cpu" => SamplingLocation::Cpu,
        "vulkan" => SamplingLocation::Vulkan,
        "backend" => SamplingLocation::Backend,
        other => {
            return Err(format!(
                "unsupported sampling location `{other}` after semantic validation"
            ))
        }
    };
    let fallback = match program.decode.fallback.as_str() {
        "depth_zero" => FallbackPolicy::DepthZero,
        "exact_target" => FallbackPolicy::ExactTarget,
        "required" => FallbackPolicy::Required,
        other => {
            return Err(format!(
                "unsupported fallback `{other}` after semantic validation"
            ))
        }
    };
    let decode = DecodePolicyPlan::new(
        program.decode.name.clone(),
        program.decode.min_horizon,
        program.decode.max_horizon,
        DecodeObjective::AcceptedTokensPerCompleteCost,
        program.decode.require_exact_acceptance,
        sampling,
        fallback,
        program
            .decode
            .required_epoch
            .as_ref()
            .map(|epoch| ControlEpoch(epoch.0)),
        ResourceCost {
            required_nvme_bytes: program.decode.cost.required_nvme_bytes,
            ram_to_vram_bytes: program.decode.cost.ram_to_vram_bytes,
            vram_bytes: program.decode.cost.vram_bytes,
            verification_compute: program.decode.cost.verification_compute,
            queue_slots: program.decode.cost.queue_slots,
        },
    )
    .ok_or_else(|| "decode policy cannot be represented as an immutable V0 plan".to_string())?;
    let mut tiers = vec![map_tier(program.target.storage)];
    for tier in &program.tiers {
        for mapped in [map_tier(tier.hot), map_tier(tier.warm), map_tier(tier.cold)] {
            if !tiers.contains(&mapped) {
                tiers.push(mapped);
            }
        }
    }
    for tensor in &program.tensors {
        let mapped = map_tier(tensor.tier);
        if !tiers.contains(&mapped) {
            tiers.push(mapped);
        }
    }
    if !tiers.contains(&ControlTier::VramSlot) {
        tiers.push(ControlTier::VramSlot);
    }
    let telemetry_names = program
        .telemetry
        .iter()
        .flat_map(|telemetry| telemetry.requirements.clone())
        .collect::<Vec<_>>();
    let telemetry = TelemetryRequirements::new(telemetry_names)
        .ok_or_else(|| "telemetry requirements are empty".to_string())?;
    ImmutableExecutionPlan::new(
        identity,
        tiers,
        decode,
        telemetry,
        program.quality.authority.clone(),
        program.target.vram_budget_bytes,
        program.target.host_ram_budget_bytes,
    )
    .ok_or_else(|| "physical plan has invalid target budgets or memory tiers".to_string())
}

fn map_tier(tier: MemoryTier) -> ControlTier {
    match tier {
        MemoryTier::DirectNvme => ControlTier::DirectNvme,
        MemoryTier::RamMapped => ControlTier::RamMapped,
        MemoryTier::PinnedRam => ControlTier::PinnedRam,
        MemoryTier::Vram => ControlTier::Vram,
        MemoryTier::VramSlot => ControlTier::VramSlot,
        MemoryTier::Scratch => ControlTier::Scratch,
    }
}

fn format_tier(tier: ControlTier) -> &'static str {
    match tier {
        ControlTier::DirectNvme => "direct_nvme",
        ControlTier::RamMapped => "ram_mapped",
        ControlTier::PinnedRam => "pinned_ram",
        ControlTier::Vram => "vram",
        ControlTier::VramSlot => "vram_slot",
        ControlTier::Scratch => "scratch",
    }
}
fn format_format(format: &QuantFormat) -> &'static str {
    match format {
        QuantFormat::Q4KS => "Q4_K_S",
        QuantFormat::Q4KM => "Q4_K_M",
        QuantFormat::Q8_0 => "Q8_0",
        QuantFormat::F16 => "F16",
        QuantFormat::F32 => "F32",
    }
}
fn format_kernel(kernel: &KernelRequirement) -> &'static str {
    match kernel {
        KernelRequirement::Cpu => "cpu",
        KernelRequirement::Vulkan => "vulkan",
        KernelRequirement::MtpVerify => "mtp_verify",
        KernelRequirement::Sampling => "sampling",
        KernelRequirement::Q4KMatVec => "q4_k_matvec",
        KernelRequirement::QuantizedMulMat => "q8_k_matvec",
        KernelRequirement::Attention => "attention",
        KernelRequirement::EmbeddingLookup => "embedding_lookup",
    }
}
fn format_authority(authority: &ValueAuthority) -> &'static str {
    match authority {
        ValueAuthority::Exact => "exact",
        ValueAuthority::Approximate => "approximate",
    }
}

impl CompiledProgram {
    pub fn to_json(&self) -> String {
        let mut output = String::new();
        output.push_str("{\n");
        field_string(&mut output, 1, "schema", "har-compiled-program/v1", true);
        field_string(&mut output, 1, "source_name", &self.source_name, true);
        field_string(&mut output, 1, "source_digest", &self.source_digest, true);
        output.push_str("  \"model_identity\": {\n");
        field_string(&mut output, 2, "name", &self.typed.model.name, true);
        field_string(&mut output, 2, "identity", &self.typed.model.identity, true);
        field_string(
            &mut output,
            2,
            "model_sha256",
            &self.typed.model.model_sha256,
            true,
        );
        field_string(
            &mut output,
            2,
            "build_hash",
            &self.typed.model.build_hash,
            true,
        );
        field_string(
            &mut output,
            2,
            "config_hash",
            &self.typed.model.config_hash,
            true,
        );
        field_number(&mut output, 2, "epoch", self.typed.model.epoch.0, false);
        output.push_str("  },\n");
        output.push_str("  \"validated_logical_ir\": {\n");
        field_string(&mut output, 2, "schema", &self.logical_ir.schema, true);
        output.push_str("    \"nodes\": [\n");
        for (index, node) in self.logical_ir.nodes.iter().enumerate() {
            output.push_str("      {\n");
            field_string(&mut output, 3, "kind", &node.kind, true);
            field_string(&mut output, 3, "name", &node.name, true);
            field_strings(&mut output, 3, "dependencies", &node.dependencies, false);
            output.push_str(if index + 1 == self.logical_ir.nodes.len() {
                "      }\n"
            } else {
                "      },\n"
            });
        }
        output.push_str("    ]\n  },\n");
        output.push_str("  \"immutable_physical_plan\": {\n");
        field_string(&mut output, 2, "schema", "har-physical-plan/v1", true);
        output.push_str("    \"identity\": {\n");
        field_string(
            &mut output,
            3,
            "model_sha256",
            self.physical_plan.identity().model_sha256(),
            true,
        );
        field_string(
            &mut output,
            3,
            "build_hash",
            self.physical_plan.identity().build_hash(),
            true,
        );
        field_string(
            &mut output,
            3,
            "config_hash",
            self.physical_plan.identity().config_hash(),
            true,
        );
        field_number(
            &mut output,
            3,
            "epoch",
            self.physical_plan.identity().epoch().0,
            false,
        );
        output.push_str("    },\n");
        field_number(
            &mut output,
            2,
            "vram_budget_bytes",
            self.physical_plan.target_vram_budget_bytes(),
            true,
        );
        field_number(
            &mut output,
            2,
            "host_ram_budget_bytes",
            self.physical_plan.target_host_ram_budget_bytes(),
            true,
        );
        field_strings(
            &mut output,
            2,
            "required_memory_tiers",
            &self
                .physical_plan
                .required_memory_tiers()
                .iter()
                .map(|tier| format_tier(*tier).to_string())
                .collect::<Vec<_>>(),
            true,
        );
        output.push_str("    \"decode_policy\": {\n");
        let decode = self.physical_plan.decode();
        field_string(&mut output, 3, "name", decode.name(), true);
        field_number(
            &mut output,
            3,
            "min_horizon",
            u64::from(decode.min_horizon()),
            true,
        );
        field_number(
            &mut output,
            3,
            "max_horizon",
            u64::from(decode.max_horizon()),
            true,
        );
        field_string(
            &mut output,
            3,
            "mode",
            if decode.mode() == HorizonMode::Elastic {
                "elastic"
            } else {
                "fixed"
            },
            true,
        );
        field_string(
            &mut output,
            3,
            "objective",
            "accepted_tokens_per_complete_cost",
            true,
        );
        field_bool(
            &mut output,
            3,
            "require_exact_acceptance",
            decode.require_exact_acceptance(),
            true,
        );
        field_string(
            &mut output,
            3,
            "sampling_location",
            match decode.sampling_location() {
                SamplingLocation::Cpu => "cpu",
                SamplingLocation::Vulkan => "vulkan",
                SamplingLocation::Backend => "backend",
            },
            true,
        );
        field_string(
            &mut output,
            3,
            "sampling_policy",
            &self.typed.decode.sampling_policy,
            true,
        );
        field_string(
            &mut output,
            3,
            "placement",
            &self.typed.decode.placement,
            true,
        );
        field_number(
            &mut output,
            3,
            "gpu_layers",
            u64::from(self.typed.decode.gpu_layers),
            true,
        );
        field_number(
            &mut output,
            3,
            "gpu_layers_total",
            u64::from(self.typed.decode.gpu_layers_total),
            true,
        );
        field_bool(
            &mut output,
            3,
            "topology_matched",
            self.typed.decode.topology_matched,
            true,
        );
        field_string(
            &mut output,
            3,
            "fallback",
            match decode.fallback() {
                FallbackPolicy::DepthZero => "depth_zero",
                FallbackPolicy::ExactTarget => "exact_target",
                FallbackPolicy::Required => "required",
            },
            false,
        );
        output.push_str("    },\n");
        field_string(
            &mut output,
            2,
            "exact_authority",
            self.physical_plan.exact_authority(),
            true,
        );
        field_strings(
            &mut output,
            2,
            "telemetry_requirements",
            self.physical_plan.telemetry().requirements(),
            false,
        );
        output.push_str("  },\n");
        output.push_str("  \"required_memory_tiers\": [\n");
        for (index, tensor) in self.typed.tensors.iter().enumerate() {
            output.push_str("    {\n");
            field_string(&mut output, 3, "tensor", &tensor.name, true);
            field_string(
                &mut output,
                3,
                "format",
                format_format(&tensor.format),
                true,
            );
            field_string(&mut output, 3, "logical_tier", &tensor.tier_name, true);
            field_string(
                &mut output,
                3,
                "tier",
                format_tier(map_tier(tensor.tier)),
                true,
            );
            field_string(
                &mut output,
                3,
                "authority",
                format_authority(&tensor.authority),
                true,
            );
            field_string(
                &mut output,
                3,
                "kernel",
                format_kernel(&tensor.kernel),
                false,
            );
            output.push_str(if index + 1 == self.typed.tensors.len() {
                "    }\n"
            } else {
                "    },\n"
            });
        }
        output.push_str("  ],\n");
        field_string(
            &mut output,
            1,
            "compiler",
            "har-lang-compiler/stable-rust-v0",
            false,
        );
        output.push_str("}\n");
        output
    }
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::new();
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            other => escaped.push(other),
        }
    }
    escaped
}
fn indent(output: &mut String, level: usize) {
    for _ in 0..level {
        output.push_str("  ");
    }
}
fn field_string(output: &mut String, level: usize, key: &str, value: &str, comma: bool) {
    indent(output, level);
    let _ = writeln!(
        output,
        "\"{}\": \"{}\"{}",
        json_escape(key),
        json_escape(value),
        if comma { "," } else { "" }
    );
}
fn field_number(output: &mut String, level: usize, key: &str, value: u64, comma: bool) {
    indent(output, level);
    let _ = writeln!(
        output,
        "\"{}\": {}{}",
        key,
        value,
        if comma { "," } else { "" }
    );
}
fn field_bool(output: &mut String, level: usize, key: &str, value: bool, comma: bool) {
    indent(output, level);
    let _ = writeln!(
        output,
        "\"{}\": {}{}",
        key,
        value,
        if comma { "," } else { "" }
    );
}
fn field_strings(output: &mut String, level: usize, key: &str, values: &[String], comma: bool) {
    indent(output, level);
    output.push_str(&format!("\"{}\": [", key));
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        output.push_str(&format!("\"{}\"", json_escape(value)));
    }
    output.push_str(if comma { "],\n" } else { "]\n" });
}

fn digest(source: &str) -> String {
    // FNV-1a is used only as a deterministic source fingerprint.  Model,
    // build, and configuration identities remain caller-supplied SHA-256
    // values and are never replaced by this fingerprint.
    let mut value = 0xcbf29ce484222325u64;
    for byte in source.as_bytes() {
        value ^= u64::from(*byte);
        value = value.wrapping_mul(0x100000001b3);
    }
    format!(
        "{value:016x}{:016x}{:016x}{:016x}",
        value.rotate_left(13),
        value.rotate_left(29),
        value.rotate_left(47)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_rejects_missing_fallback() {
        let source = r#"target t { gpu "g"; wave 32; vram_budget 1 GiB; host_ram_budget 1 GiB; storage direct_nvme; }
model m { identity "m"; model_sha256 "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"; build_hash "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"; config_hash "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"; }
quality exact { authority full_model; fallback forbidden; }
decode d { horizon 0..3; optimize accepted_tokens_per_complete_cost; require exact_acceptance; sampling cpu; fallback depth_zero; }
telemetry t { require accepted_tokens; }"#;
        let errors = compile_source("test.har", source).expect_err("must reject");
        assert!(errors.iter().any(|error| error.code == "S1033"));
    }
}
