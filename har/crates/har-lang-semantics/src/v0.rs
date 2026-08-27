//! Typed semantic AST and fail-closed validation for HAR V0.
//!
//! This crate is an offline/startup compiler stage.  It does not expose a
//! parser or syntax object to the decode hot path.

use har_lang_ast::{Block, Field, Program, Value};
use har_lang_diagnostics::{Diagnostic, Diagnostics};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape(pub Vec<u64>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantFormat {
    Q4KS,
    Q4KM,
    Q8_0,
    F16,
    F32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryTier {
    DirectNvme,
    RamMapped,
    PinnedRam,
    Vram,
    VramSlot,
    Scratch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidencyState {
    Cold,
    Warm,
    Hot,
    Resident,
    TransferRequired,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelRequirement {
    Cpu,
    Vulkan,
    MtpVerify,
    Sampling,
    Q4KMatVec,
    QuantizedMulMat,
    Attention,
    EmbeddingLookup,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueAuthority {
    Exact,
    Approximate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generation(pub u64);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Epoch(pub u64);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Certificate {
    pub kind: String,
    pub authority: String,
    pub epoch: Epoch,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactValue<T> {
    pub value: T,
    pub authority: String,
    pub certificate: Option<Certificate>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ApproximateValue<T> {
    pub value: T,
    pub error_bound: Option<f64>,
    pub certificate: Option<Certificate>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelIdentity {
    pub name: String,
    pub identity: String,
    pub model_sha256: String,
    pub build_hash: String,
    pub config_hash: String,
    pub epoch: Epoch,
    pub size_gib: f64,
    pub context_length: u32,
    pub batch: u32,
    pub kv_cache_type: String,
    pub nextn_predict_layers: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TargetSpec {
    pub name: String,
    pub gpu: String,
    pub gpu_arch: String,
    pub wave: u32,
    pub vram_budget_bytes: u64,
    pub host_ram_budget_bytes: u64,
    pub storage: MemoryTier,
    pub capabilities: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorSpec {
    pub name: String,
    pub format: QuantFormat,
    pub shape: Shape,
    pub tier: MemoryTier,
    pub tier_name: String,
    pub authority: ValueAuthority,
    pub kernel: KernelRequirement,
    pub required_vram_bytes: u64,
    pub generation: Generation,
    pub model_root: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TierSpec {
    pub name: String,
    pub hot: MemoryTier,
    pub warm: MemoryTier,
    pub cold: MemoryTier,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QualitySpec {
    pub name: String,
    pub authority: String,
    pub exact: bool,
    pub fallback_required: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResourceCostSpec {
    pub required_nvme_bytes: u64,
    pub ram_to_vram_bytes: u64,
    pub vram_bytes: u64,
    pub verification_compute: f64,
    pub queue_slots: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecodeSpec {
    pub name: String,
    pub min_horizon: u8,
    pub max_horizon: u8,
    pub objective: String,
    pub require_exact_acceptance: bool,
    pub sampling_location: String,
    pub sampling_policy: String,
    pub placement: String,
    pub gpu_layers: u32,
    pub gpu_layers_total: u32,
    pub topology_matched: bool,
    pub requires_model: String,
    pub fallback: String,
    pub required_epoch: Option<Epoch>,
    pub cost: ResourceCostSpec,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FactSpec {
    pub name: String,
    pub metric: String,
    pub value: f64,
    pub evidence: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionPhaseSpec {
    pub name: String,
    pub dependencies: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dependency {
    pub from: String,
    pub to: String,
    pub generation: Generation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TelemetrySpec {
    pub name: String,
    pub requirements: Vec<String>,
}

/// One executable physical operation declaration.  Every field is an explicit
/// identity or contract; nothing is inferred at runtime.
#[derive(Clone, Debug, PartialEq)]
pub struct OperationSpec {
    pub name: String,
    pub model_root: String,
    pub package_root: String,
    pub payload_entry: String,
    pub source_tensor: String,
    pub kernel: KernelRequirement,
    pub quant_format: QuantFormat,
    pub rows: u32,
    pub columns: u32,
    pub alignment: u64,
    pub source_tier: MemoryTier,
    pub destination_tier: MemoryTier,
    pub require_exact_checksum: bool,
    pub reference_tolerance: f64,
    pub fallback: String,
    pub generation: Generation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypedProgram {
    pub target: TargetSpec,
    pub model: ModelIdentity,
    pub tensors: Vec<TensorSpec>,
    pub tiers: Vec<TierSpec>,
    pub quality: QualitySpec,
    pub decode: DecodeSpec,
    pub phases: Vec<ExecutionPhaseSpec>,
    pub dependencies: Vec<Dependency>,
    pub telemetry: Vec<TelemetrySpec>,
    pub facts: Vec<FactSpec>,
    pub operations: Vec<OperationSpec>,
}

pub fn analyze(program: &Program) -> Result<TypedProgram, Vec<Diagnostic>> {
    let mut diagnostics = Diagnostics::default();
    let target_blocks = blocks(program, "target");
    let model_blocks = blocks(program, "model");
    let quality_blocks = blocks(program, "quality");
    let decode_blocks = blocks(program, "decode");
    if target_blocks.len() != 1 {
        diagnostics.push(Diagnostic::error(
            "S1001",
            "HAR V0 requires exactly one target declaration",
            target_blocks.first().map(|block| block.span),
        ));
    }
    if model_blocks.len() != 1 {
        diagnostics.push(Diagnostic::error(
            "S1002",
            "HAR V0 requires exactly one model declaration",
            model_blocks.first().map(|block| block.span),
        ));
    }
    if quality_blocks.len() != 1 {
        diagnostics.push(Diagnostic::error(
            "S1003",
            "HAR V0 requires exactly one quality declaration",
            quality_blocks.first().map(|block| block.span),
        ));
    }
    if decode_blocks.len() != 1 {
        diagnostics.push(Diagnostic::error(
            "S1004",
            "HAR V0 requires exactly one decode declaration",
            decode_blocks.first().map(|block| block.span),
        ));
    }

    let target = target_blocks
        .first()
        .map(|block| parse_target(block, &mut diagnostics))
        .unwrap_or_else(|| TargetSpec {
            name: "missing".into(),
            gpu: String::new(),
            gpu_arch: String::new(),
            wave: 0,
            vram_budget_bytes: 0,
            host_ram_budget_bytes: 0,
            storage: MemoryTier::DirectNvme,
            capabilities: Vec::new(),
        });
    let model = model_blocks
        .first()
        .map(|block| parse_model(block, &mut diagnostics))
        .unwrap_or_else(|| ModelIdentity {
            name: "missing".into(),
            identity: String::new(),
            model_sha256: String::new(),
            build_hash: String::new(),
            config_hash: String::new(),
            epoch: Epoch(0),
            size_gib: 0.0,
            context_length: 0,
            batch: 0,
            kv_cache_type: String::new(),
            nextn_predict_layers: 0,
        });
    let quality = quality_blocks
        .first()
        .map(|block| parse_quality(block, &mut diagnostics))
        .unwrap_or_else(|| QualitySpec {
            name: "missing".into(),
            authority: String::new(),
            exact: false,
            fallback_required: false,
        });
    let decode = decode_blocks
        .first()
        .map(|block| parse_decode(block, &mut diagnostics))
        .unwrap_or_else(|| DecodeSpec {
            name: "missing".into(),
            min_horizon: 0,
            max_horizon: 0,
            objective: String::new(),
            require_exact_acceptance: false,
            sampling_location: String::new(),
            sampling_policy: String::new(),
            placement: String::new(),
            gpu_layers: 0,
            gpu_layers_total: 0,
            topology_matched: false,
            requires_model: String::new(),
            fallback: String::new(),
            required_epoch: None,
            cost: ResourceCostSpec {
                required_nvme_bytes: 0,
                ram_to_vram_bytes: 0,
                vram_bytes: 0,
                verification_compute: 0.0,
                queue_slots: 0,
            },
        });

    let mut tensors = Vec::new();
    let mut names = HashSet::new();
    for block in blocks(program, "tensor") {
        if !names.insert(block.name.clone()) {
            diagnostics.push(Diagnostic::error(
                "S1005",
                format!("duplicate tensor `{}`", block.name),
                Some(block.span),
            ));
        }
        if let Some(tensor) = parse_tensor(block, &mut diagnostics) {
            tensors.push(tensor);
        }
    }
    let mut tiers = Vec::new();
    names.clear();
    for block in blocks(program, "tier") {
        if !names.insert(block.name.clone()) {
            diagnostics.push(Diagnostic::error(
                "S1006",
                format!("duplicate tier `{}`", block.name),
                Some(block.span),
            ));
        }
        if let Some(tier) = parse_tier(block, &mut diagnostics) {
            tiers.push(tier);
        }
    }
    // A tensor may name a logical tier declaration (for example
    // `expert_weights`) or a concrete physical tier. Resolve logical names to
    // their hot placement before lowering; an unresolved name is an error.
    for tensor in &mut tensors {
        if let Some(tier) = tiers.iter().find(|tier| tier.name == tensor.tier_name) {
            tensor.tier = tier.hot;
        } else if !matches!(
            tensor.tier_name.as_str(),
            "direct_nvme" | "nvme" | "ram_mapped" | "pinned_ram" | "vram" | "vram_slot" | "scratch"
        ) {
            diagnostics.push(Diagnostic::error(
                "S1054",
                format!(
                    "tensor `{}` references missing logical tier `{}`",
                    tensor.name, tensor.tier_name
                ),
                None,
            ));
        }
    }

    let mut phases = Vec::new();
    names.clear();
    for block in blocks(program, "phase") {
        if !names.insert(block.name.clone()) {
            diagnostics.push(Diagnostic::error(
                "S1007",
                format!("duplicate phase `{}`", block.name),
                Some(block.span),
            ));
        }
        phases.push(parse_phase(block, &mut diagnostics));
    }
    let mut dependencies = phases
        .iter()
        .flat_map(|phase| {
            phase.dependencies.iter().map(|dependency| Dependency {
                from: phase.name.clone(),
                to: dependency.clone(),
                generation: Generation(model.epoch.0),
            })
        })
        .collect::<Vec<_>>();
    for block in blocks(program, "dependency") {
        check_known_fields(block, &["from", "to", "generation"], &mut diagnostics);
        let from = required_atom(block, "from", &mut diagnostics);
        let to = required_atom(block, "to", &mut diagnostics);
        let generation = field(block, "generation")
            .map(|field| {
                Generation(parse_u64(
                    &field.value,
                    "generation",
                    block,
                    &mut diagnostics,
                ))
            })
            .unwrap_or(Generation(model.epoch.0));
        dependencies.push(Dependency {
            from,
            to,
            generation,
        });
    }
    check_phase_graph(&phases, &mut diagnostics);

    let mut telemetry = Vec::new();
    for block in blocks(program, "telemetry") {
        telemetry.push(parse_telemetry(block, &mut diagnostics));
    }
    if telemetry.is_empty() {
        diagnostics.push(Diagnostic::error(
            "S1017",
            "decode programs must declare telemetry requirements",
            decode_blocks.first().map(|block| block.span),
        ));
    }
    for block in blocks(program, "budget") {
        check_known_fields(
            block,
            &["vram", "host_ram", "nvme", "epoch"],
            &mut diagnostics,
        );
    }
    let mut facts = Vec::new();
    names.clear();
    for block in blocks(program, "fact") {
        if !names.insert(block.name.clone()) {
            diagnostics.push(Diagnostic::error(
                "S1062",
                format!("duplicate fact `{}`", block.name),
                Some(block.span),
            ));
        }
        facts.push(parse_fact(block, &mut diagnostics));
    }

    let mut operations = Vec::new();
    names.clear();
    for block in blocks(program, "operation") {
        if !names.insert(block.name.clone()) {
            diagnostics.push(Diagnostic::error(
                "S1074",
                format!("duplicate operation `{}`", block.name),
                Some(block.span),
            ));
        }
        operations.push(parse_operation(block, &mut diagnostics));
    }

    for block in &program.declarations {
        if !matches!(
            block.kind.as_str(),
            "target"
                | "model"
                | "quality"
                | "decode"
                | "tensor"
                | "tier"
                | "phase"
                | "telemetry"
                | "budget"
                | "dependency"
                | "fact"
                | "operation"
        ) {
            diagnostics.push(Diagnostic::error(
                "S1018",
                format!("unsupported declaration kind `{}`", block.kind),
                Some(block.span),
            ));
        }
    }
    check_cross_constraints(
        &target,
        &model,
        &quality,
        &decode,
        &tensors,
        &mut diagnostics,
    );

    if diagnostics.has_errors() {
        Err(diagnostics.items)
    } else {
        Ok(TypedProgram {
            target,
            model,
            tensors,
            tiers,
            quality,
            decode,
            phases,
            dependencies,
            telemetry,
            facts,
            operations,
        })
    }
}

fn blocks<'a>(program: &'a Program, kind: &str) -> Vec<&'a Block> {
    program
        .declarations
        .iter()
        .filter(|block| block.kind == kind)
        .collect()
}

fn field<'a>(block: &'a Block, key: &str) -> Option<&'a Field> {
    block.fields.iter().find(|field| field.key == key)
}
fn fields<'a>(block: &'a Block, key: &str) -> Vec<&'a Field> {
    block
        .fields
        .iter()
        .filter(|field| field.key == key)
        .collect()
}

fn atom(value: &Value) -> Option<&str> {
    value.as_atom()
}
fn string_or_atom(value: &Value) -> Option<&str> {
    value.as_string().or_else(|| value.as_ident())
}

fn required_atom(block: &Block, key: &str, diagnostics: &mut Diagnostics) -> String {
    match field(block, key).and_then(|field| atom(&field.value)) {
        Some(value) => value.to_string(),
        None => {
            diagnostics.push(Diagnostic::error(
                "S1000",
                format!("`{} {}` is required", block.kind, key),
                Some(block.span),
            ));
            String::new()
        }
    }
}

fn parse_number(value: &Value, key: &str, block: &Block, diagnostics: &mut Diagnostics) -> f64 {
    let Some(raw) = atom(value) else {
        diagnostics.push(Diagnostic::error(
            "S1020",
            format!("field `{key}` requires a number"),
            Some(block.span),
        ));
        return 0.0;
    };
    match raw.parse::<f64>() {
        Ok(number) if number.is_finite() && number >= 0.0 => number,
        _ => {
            diagnostics.push(Diagnostic::error(
                "S1021",
                format!("invalid non-negative number `{raw}`"),
                Some(block.span),
            ));
            0.0
        }
    }
}

fn parse_u64(value: &Value, key: &str, block: &Block, diagnostics: &mut Diagnostics) -> u64 {
    let number = parse_number(value, key, block, diagnostics);
    if number.fract() != 0.0 || number > u64::MAX as f64 {
        diagnostics.push(Diagnostic::error(
            "S1022",
            format!("`{key}` must be an integer"),
            Some(block.span),
        ));
        return 0;
    }
    number as u64
}

fn parse_quantity(value: &Value, key: &str, block: &Block, diagnostics: &mut Diagnostics) -> u64 {
    let (number, unit) = match value {
        Value::Quantity { number, unit } => (number, unit),
        _ => {
            diagnostics.push(Diagnostic::error(
                "S1023",
                format!("`{key}` requires a quantity such as `15.9 GiB`"),
                Some(block.span),
            ));
            return 0;
        }
    };
    let number = match number.parse::<f64>() {
        Ok(number) if number.is_finite() && number >= 0.0 => number,
        _ => {
            diagnostics.push(Diagnostic::error(
                "S1024",
                format!("invalid quantity number `{number}`"),
                Some(block.span),
            ));
            return 0;
        }
    };
    let multiplier = match unit.as_str() {
        "B" => 1.0,
        "KiB" => 1024.0,
        "MiB" => 1024.0_f64.powi(2),
        "GiB" => 1024.0_f64.powi(3),
        "TiB" => 1024.0_f64.powi(4),
        _ => {
            diagnostics.push(Diagnostic::error(
                "S1025",
                format!("unsupported capacity unit `{unit}`"),
                Some(block.span),
            ));
            return 0;
        }
    };
    let bytes = number * multiplier;
    if bytes > u64::MAX as f64 {
        diagnostics.push(Diagnostic::error(
            "S1026",
            "quantity exceeds u64 capacity",
            Some(block.span),
        ));
        0
    } else {
        bytes.round() as u64
    }
}

fn parse_target(block: &Block, diagnostics: &mut Diagnostics) -> TargetSpec {
    check_known_fields(
        block,
        &[
            "gpu",
            "gpu_arch",
            "wave",
            "vram_budget",
            "host_ram_budget",
            "storage",
            "capability",
        ],
        diagnostics,
    );
    let gpu = field(block, "gpu")
        .and_then(|field| string_or_atom(&field.value))
        .unwrap_or("")
        .to_string();
    let gpu_arch = field(block, "gpu_arch")
        .and_then(|field| string_or_atom(&field.value))
        .unwrap_or("")
        .to_string();
    let wave = field(block, "wave")
        .map(|field| parse_u64(&field.value, "wave", block, diagnostics) as u32)
        .unwrap_or_else(|| {
            diagnostics.push(Diagnostic::error(
                "S1030",
                "target requires `wave`",
                Some(block.span),
            ));
            0
        });
    let vram = field(block, "vram_budget")
        .map(|field| parse_quantity(&field.value, "vram_budget", block, diagnostics))
        .unwrap_or(0);
    let host = field(block, "host_ram_budget")
        .map(|field| parse_quantity(&field.value, "host_ram_budget", block, diagnostics))
        .unwrap_or(0);
    let storage_name = required_atom(block, "storage", diagnostics);
    let storage = parse_memory_tier(&storage_name, block, diagnostics);
    let capabilities = fields(block, "capability")
        .iter()
        .filter_map(|field| atom(&field.value).map(str::to_string))
        .collect();
    TargetSpec {
        name: block.name.clone(),
        gpu,
        gpu_arch,
        wave,
        vram_budget_bytes: vram,
        host_ram_budget_bytes: host,
        storage,
        capabilities,
    }
}

fn parse_model(block: &Block, diagnostics: &mut Diagnostics) -> ModelIdentity {
    check_known_fields(
        block,
        &[
            "identity",
            "model_sha256",
            "build_hash",
            "config_hash",
            "epoch",
            "size_gib",
            "context_length",
            "batch",
            "kv_cache_type",
            "nextn_predict_layers",
        ],
        diagnostics,
    );
    let identity = required_string(block, "identity", diagnostics);
    let model_sha256 = required_string(block, "model_sha256", diagnostics);
    let build_hash = required_string(block, "build_hash", diagnostics);
    let config_hash = required_string(block, "config_hash", diagnostics);
    let epoch = field(block, "epoch")
        .map(|field| Epoch(parse_u64(&field.value, "epoch", block, diagnostics)))
        .unwrap_or(Epoch(0));
    for (name, hash) in [
        ("model_sha256", &model_sha256),
        ("build_hash", &build_hash),
        ("config_hash", &config_hash),
    ] {
        let accepted_length = name == "build_hash" && (hash.len() == 40 || hash.len() == 64)
            || name != "build_hash" && hash.len() == 64;
        if !accepted_length || !hash.chars().all(|character| character.is_ascii_hexdigit()) {
            diagnostics.push(Diagnostic::error("S1031", format!("`{name}` must be a hexadecimal model/config identity (64 chars) or build identity (40/64 chars)"), Some(block.span)));
        }
    }
    let size_gib = field(block, "size_gib")
        .map(|field| parse_number(&field.value, "size_gib", block, diagnostics))
        .unwrap_or(0.0);
    let context_length = field(block, "context_length")
        .map(|field| parse_u64(&field.value, "context_length", block, diagnostics) as u32)
        .unwrap_or(0);
    let batch = field(block, "batch")
        .map(|field| parse_u64(&field.value, "batch", block, diagnostics) as u32)
        .unwrap_or(0);
    let kv_cache_type = field(block, "kv_cache_type")
        .and_then(|field| string_or_atom(&field.value))
        .unwrap_or("")
        .to_string();
    if !matches!(kv_cache_type.as_str(), "" | "q8_0" | "f16") {
        diagnostics.push(Diagnostic::error(
            "S1059",
            format!("unsupported KV cache type `{kv_cache_type}`; HAR V0 accepts q8_0 or f16"),
            Some(block.span),
        ));
    }
    let nextn_predict_layers = field(block, "nextn_predict_layers")
        .map(|field| parse_u64(&field.value, "nextn_predict_layers", block, diagnostics) as u32)
        .unwrap_or(0);
    ModelIdentity {
        name: block.name.clone(),
        identity,
        model_sha256,
        build_hash,
        config_hash,
        epoch,
        size_gib,
        context_length,
        batch,
        kv_cache_type,
        nextn_predict_layers,
    }
}

fn required_string(block: &Block, key: &str, diagnostics: &mut Diagnostics) -> String {
    match field(block, key).and_then(|field| string_or_atom(&field.value)) {
        Some(value) if !value.is_empty() => value.to_string(),
        _ => {
            diagnostics.push(Diagnostic::error(
                "S1032",
                format!("`{}` requires a non-empty string", key),
                Some(block.span),
            ));
            String::new()
        }
    }
}

fn parse_quality(block: &Block, diagnostics: &mut Diagnostics) -> QualitySpec {
    check_known_fields(block, &["authority", "fallback"], diagnostics);
    let authority = required_atom(block, "authority", diagnostics);
    let exact = block.name == "exact" || authority == "full_model";
    let fallback = required_atom(block, "fallback", diagnostics);
    let fallback_required = matches!(
        fallback.as_str(),
        "required" | "depth_zero" | "exact_target"
    );
    if exact && !fallback_required {
        diagnostics.push(Diagnostic::error(
            "S1033",
            "exact quality requires an explicit fallback",
            Some(block.span),
        ));
    }
    QualitySpec {
        name: block.name.clone(),
        authority,
        exact,
        fallback_required,
    }
}

fn parse_decode(block: &Block, diagnostics: &mut Diagnostics) -> DecodeSpec {
    check_known_fields(
        block,
        &[
            "horizon",
            "optimize",
            "require",
            "sampling",
            "placement",
            "gpu_layers",
            "gpu_layers_total",
            "topology_matched",
            "requires_model",
            "fallback",
            "requires_epoch",
            "required_nvme",
            "ram_to_vram",
            "vram_reserve",
            "verification_compute",
            "queue_slots",
        ],
        diagnostics,
    );
    // MTP acceptance is a pure discrete decision.  A decode policy that makes
    // acceptance depend on timing, elapsed time, deadlines, or measurement
    // windows is rejected before lowering (S1056).
    for field in &block.fields {
        let key = field.key.to_ascii_lowercase();
        if key.contains("timing")
            || key.contains("elapsed")
            || key.contains("deadline")
            || key.contains("after_ms")
        {
            diagnostics.push(Diagnostic::error(
                "S1056",
                format!(
                    "MTP acceptance must not be controlled by timing: `{}` is not a legal decode policy input",
                    field.key
                ),
                Some(field.span),
            ));
        }
    }
    let (min_horizon, max_horizon) = match field(block, "horizon").map(|field| &field.value) {
        Some(Value::Range { start, end }) => (
            start.parse::<u8>().unwrap_or(255),
            end.parse::<u8>().unwrap_or(255),
        ),
        _ => {
            diagnostics.push(Diagnostic::error(
                "S1034",
                "decode requires a horizon range such as `0..3`",
                Some(block.span),
            ));
            (0, 0)
        }
    };
    if min_horizon > max_horizon || max_horizon > 3 {
        diagnostics.push(Diagnostic::error(
            "S1035",
            "MTP horizon must be an ordered range inside 0..3",
            Some(block.span),
        ));
    }
    let objective = required_atom(block, "optimize", diagnostics);
    if objective != "accepted_tokens_per_complete_cost"
        && objective != "expected_accepted_tokens_per_complete_cost"
    {
        diagnostics.push(Diagnostic::error(
            "S1036",
            format!("unsupported decode objective `{objective}`"),
            Some(block.span),
        ));
    }
    let require_exact_acceptance = field(block, "require")
        .and_then(|field| atom(&field.value))
        .map(|value| value == "exact_acceptance")
        .unwrap_or(false);
    if !require_exact_acceptance {
        diagnostics.push(Diagnostic::error(
            "S1037",
            "decode requires `require exact_acceptance`",
            Some(block.span),
        ));
    }
    let sampling_location = required_atom(block, "sampling", diagnostics);
    let (sampling_location, sampling_policy) = match sampling_location.as_str() {
        "cpu" => ("cpu", "greedy"),
        "vulkan" => ("vulkan", "greedy"),
        "backend" => ("backend", "greedy"),
        "deterministic_greedy" => ("cpu", "deterministic_greedy"),
        "seeded_stochastic" => ("cpu", "seeded_stochastic"),
        _ => {
            diagnostics.push(Diagnostic::error(
                "S1038",
                format!("unsupported sampling policy `{sampling_location}`"),
                Some(block.span),
            ));
            ("cpu", "greedy")
        }
    };
    let placement = field(block, "placement")
        .and_then(|field| atom(&field.value))
        .unwrap_or("natural")
        .to_string();
    if !matches!(placement.as_str(), "natural" | "topology_matched") {
        diagnostics.push(Diagnostic::error(
            "S1060",
            format!("unsupported placement policy `{placement}`; V0 accepts natural or topology_matched"),
            Some(block.span),
        ));
    }
    let gpu_layers = field(block, "gpu_layers")
        .map(|field| parse_u64(&field.value, "gpu_layers", block, diagnostics) as u32)
        .unwrap_or(0);
    let gpu_layers_total = field(block, "gpu_layers_total")
        .map(|field| parse_u64(&field.value, "gpu_layers_total", block, diagnostics) as u32)
        .unwrap_or(0);
    if gpu_layers_total != 0 && gpu_layers > gpu_layers_total {
        diagnostics.push(Diagnostic::error(
            "S1061",
            format!("gpu_layers {gpu_layers} exceeds gpu_layers_total {gpu_layers_total}"),
            Some(block.span),
        ));
    }
    let topology_matched = field(block, "topology_matched")
        .and_then(|field| atom(&field.value))
        .map(|value| value == "true")
        .unwrap_or(false);
    let requires_model = field(block, "requires_model")
        .and_then(|field| string_or_atom(&field.value))
        .unwrap_or("")
        .to_string();
    let fallback = required_atom(block, "fallback", diagnostics);
    if !matches!(
        fallback.as_str(),
        "depth_zero" | "exact_target" | "required"
    ) {
        diagnostics.push(Diagnostic::error(
            "S1039",
            "decode fallback must be `depth_zero` or `exact_target`",
            Some(block.span),
        ));
    }
    let required_epoch = field(block, "requires_epoch").map(|field| {
        Epoch(parse_u64(
            &field.value,
            "requires_epoch",
            block,
            diagnostics,
        ))
    });
    let required_nvme_bytes = field(block, "required_nvme")
        .map(|field| parse_quantity(&field.value, "required_nvme", block, diagnostics))
        .unwrap_or(0);
    let ram_to_vram_bytes = field(block, "ram_to_vram")
        .map(|field| parse_quantity(&field.value, "ram_to_vram", block, diagnostics))
        .unwrap_or(0);
    let vram_bytes = field(block, "vram_reserve")
        .map(|field| parse_quantity(&field.value, "vram_reserve", block, diagnostics))
        .unwrap_or(0);
    let verification_compute = field(block, "verification_compute")
        .map(|field| parse_number(&field.value, "verification_compute", block, diagnostics))
        .unwrap_or(0.0);
    let queue_slots = field(block, "queue_slots")
        .map(|field| parse_u64(&field.value, "queue_slots", block, diagnostics))
        .unwrap_or(0);
    DecodeSpec {
        name: block.name.clone(),
        min_horizon,
        max_horizon,
        objective,
        require_exact_acceptance,
        sampling_location: sampling_location.to_string(),
        sampling_policy: sampling_policy.to_string(),
        placement,
        gpu_layers,
        gpu_layers_total,
        topology_matched,
        requires_model,
        fallback,
        required_epoch,
        cost: ResourceCostSpec {
            required_nvme_bytes,
            ram_to_vram_bytes,
            vram_bytes,
            verification_compute,
            queue_slots,
        },
    }
}

fn parse_tensor(block: &Block, diagnostics: &mut Diagnostics) -> Option<TensorSpec> {
    check_known_fields(
        block,
        &[
            "format",
            "shape",
            "tier",
            "authority",
            "kernel",
            "vram_required",
            "generation",
            "model_root",
        ],
        diagnostics,
    );
    let format_name = required_atom(block, "format", diagnostics);
    let format = match format_name.as_str() {
        "Q4_K_S" => QuantFormat::Q4KS,
        "Q4_K_M" => QuantFormat::Q4KM,
        "Q8_0" => QuantFormat::Q8_0,
        "F16" => QuantFormat::F16,
        "F32" => QuantFormat::F32,
        _ => {
            diagnostics.push(Diagnostic::error(
                "S1040",
                format!("unsupported tensor format `{format_name}`"),
                Some(block.span),
            ));
            QuantFormat::F32
        }
    };
    let shape = match field(block, "shape").map(|field| &field.value) {
        Some(Value::List(values)) => Shape(
            values
                .iter()
                .map(|value| parse_u64(value, "shape", block, diagnostics))
                .collect(),
        ),
        _ => {
            diagnostics.push(Diagnostic::error(
                "S1041",
                "tensor requires a shape list",
                Some(block.span),
            ));
            Shape(Vec::new())
        }
    };
    let tier_name = required_atom(block, "tier", diagnostics);
    let tier = match tier_name.as_str() {
        "direct_nvme" | "nvme" | "ram_mapped" | "pinned_ram" | "vram" | "vram_slot" | "scratch" => {
            parse_memory_tier(&tier_name, block, diagnostics)
        }
        _ => MemoryTier::DirectNvme,
    };
    let authority_name = field(block, "authority")
        .and_then(|field| atom(&field.value))
        .unwrap_or("exact");
    let authority = match authority_name {
        "exact" | "full_model" => ValueAuthority::Exact,
        "approximate" | "predictive" => ValueAuthority::Approximate,
        _ => {
            diagnostics.push(Diagnostic::error(
                "S1042",
                format!("unknown tensor authority `{authority_name}`"),
                Some(block.span),
            ));
            ValueAuthority::Exact
        }
    };
    let kernel_name = field(block, "kernel")
        .and_then(|field| atom(&field.value))
        .unwrap_or("cpu");
    let kernel = match kernel_name {
        "cpu" => KernelRequirement::Cpu,
        "vulkan" => KernelRequirement::Vulkan,
        "mtp_verify" => KernelRequirement::MtpVerify,
        "sampling" => KernelRequirement::Sampling,
        "q4_k_matvec" | "q4_k_mul_mat" => KernelRequirement::Q4KMatVec,
        "q8_k_matvec" | "quantized_mul_mat" => KernelRequirement::QuantizedMulMat,
        "attention" => KernelRequirement::Attention,
        "embedding_lookup" => KernelRequirement::EmbeddingLookup,
        _ => {
            diagnostics.push(Diagnostic::error(
                "S1043",
                format!("unsupported kernel requirement `{kernel_name}`"),
                Some(block.span),
            ));
            KernelRequirement::Cpu
        }
    };
    let required_vram_bytes = field(block, "vram_required")
        .map(|field| parse_quantity(&field.value, "vram_required", block, diagnostics))
        .unwrap_or(0);
    let generation = Generation(
        field(block, "generation")
            .map(|field| parse_u64(&field.value, "generation", block, diagnostics))
            .unwrap_or(0),
    );
    let model_root = field(block, "model_root")
        .and_then(|field| string_or_atom(&field.value))
        .unwrap_or("")
        .to_string();
    Some(TensorSpec {
        name: block.name.clone(),
        format,
        shape,
        tier,
        tier_name,
        authority,
        kernel,
        required_vram_bytes,
        generation,
        model_root,
    })
}

fn parse_memory_tier(name: &str, block: &Block, diagnostics: &mut Diagnostics) -> MemoryTier {
    match name {
        "direct_nvme" | "nvme" | "nvme_cold" => MemoryTier::DirectNvme,
        "ram_mapped" | "ram_pageable" => MemoryTier::RamMapped,
        "pinned_ram" | "ram_pinned" => MemoryTier::PinnedRam,
        "vram" | "vram_page" => MemoryTier::Vram,
        "vram_slot" => MemoryTier::VramSlot,
        "scratch" => MemoryTier::Scratch,
        _ => {
            diagnostics.push(Diagnostic::error(
                "S1044",
                format!("unknown memory tier `{name}`"),
                Some(block.span),
            ));
            MemoryTier::DirectNvme
        }
    }
}

fn parse_tier(block: &Block, diagnostics: &mut Diagnostics) -> Option<TierSpec> {
    check_known_fields(block, &["hot", "warm", "cold"], diagnostics);
    let hot = parse_memory_tier(
        &required_atom(block, "hot", diagnostics),
        block,
        diagnostics,
    );
    let warm = parse_memory_tier(
        &required_atom(block, "warm", diagnostics),
        block,
        diagnostics,
    );
    let cold = parse_memory_tier(
        &required_atom(block, "cold", diagnostics),
        block,
        diagnostics,
    );
    Some(TierSpec {
        name: block.name.clone(),
        hot,
        warm,
        cold,
    })
}

fn parse_phase(block: &Block, diagnostics: &mut Diagnostics) -> ExecutionPhaseSpec {
    check_known_fields(block, &["depends_on"], diagnostics);
    let dependencies = fields(block, "depends_on")
        .iter()
        .filter_map(|field| atom(&field.value).map(str::to_string))
        .collect();
    ExecutionPhaseSpec {
        name: block.name.clone(),
        dependencies,
    }
}

fn parse_telemetry(block: &Block, diagnostics: &mut Diagnostics) -> TelemetrySpec {
    check_known_fields(block, &["require"], diagnostics);
    let requirements = fields(block, "require")
        .iter()
        .filter_map(|field| atom(&field.value).map(str::to_string))
        .collect::<Vec<_>>();
    if requirements.is_empty() {
        diagnostics.push(Diagnostic::error(
            "S1045",
            "telemetry declaration requires at least one `require`",
            Some(block.span),
        ));
    }
    TelemetrySpec {
        name: block.name.clone(),
        requirements,
    }
}

fn parse_fact(block: &Block, diagnostics: &mut Diagnostics) -> FactSpec {
    check_known_fields(block, &["metric", "value", "evidence"], diagnostics);
    let metric = required_atom(block, "metric", diagnostics);
    let value = field(block, "value")
        .map(|field| parse_number(&field.value, "value", block, diagnostics))
        .unwrap_or(0.0);
    let evidence = required_string(block, "evidence", diagnostics);
    FactSpec {
        name: block.name.clone(),
        metric,
        value,
        evidence,
    }
}

#[allow(clippy::manual_is_multiple_of)]
fn parse_operation(block: &Block, diagnostics: &mut Diagnostics) -> OperationSpec {
    check_known_fields(
        block,
        &[
            "model_root",
            "package_root",
            "payload_entry",
            "source_tensor",
            "kernel",
            "quant_format",
            "rows",
            "columns",
            "alignment",
            "source",
            "destination",
            "require",
            "reference_tolerance",
            "fallback",
            "generation",
        ],
        diagnostics,
    );
    let model_root = required_string(block, "model_root", diagnostics);
    let package_root = required_string(block, "package_root", diagnostics);
    let payload_entry = required_string(block, "payload_entry", diagnostics);
    let source_tensor = required_string(block, "source_tensor", diagnostics);
    let kernel_name = required_atom(block, "kernel", diagnostics);
    let kernel = match kernel_name.as_str() {
        "q4_k_matvec" | "q4_k_mul_mat" => KernelRequirement::Q4KMatVec,
        "q8_k_matvec" | "quantized_mul_mat" => KernelRequirement::QuantizedMulMat,
        "attention" => KernelRequirement::Attention,
        "mtp_verify" => KernelRequirement::MtpVerify,
        "sampling" => KernelRequirement::Sampling,
        "embedding_lookup" => KernelRequirement::EmbeddingLookup,
        "cpu" => KernelRequirement::Cpu,
        "vulkan" => KernelRequirement::Vulkan,
        _ => {
            diagnostics.push(Diagnostic::error(
                "S1043",
                format!("unsupported kernel requirement `{kernel_name}`"),
                Some(block.span),
            ));
            KernelRequirement::Cpu
        }
    };
    let format_name = required_atom(block, "quant_format", diagnostics);
    let quant_format = match format_name.as_str() {
        "Q4_K" | "Q4_K_S" => QuantFormat::Q4KS,
        "Q4_K_M" => QuantFormat::Q4KM,
        "Q8_0" => QuantFormat::Q8_0,
        "F16" => QuantFormat::F16,
        "F32" => QuantFormat::F32,
        _ => {
            diagnostics.push(Diagnostic::error(
                "S1040",
                format!("unsupported operation quant format `{format_name}`"),
                Some(block.span),
            ));
            QuantFormat::F32
        }
    };
    let rows = field(block, "rows")
        .map(|field| parse_u64(&field.value, "rows", block, diagnostics) as u32)
        .unwrap_or(0);
    let columns = field(block, "columns")
        .map(|field| parse_u64(&field.value, "columns", block, diagnostics) as u32)
        .unwrap_or(0);
    if rows == 0 || columns == 0 {
        diagnostics.push(Diagnostic::error(
            "S1063",
            "operation shape must declare positive rows and columns",
            Some(block.span),
        ));
    }
    if columns != 0 && columns % 256 != 0 {
        diagnostics.push(Diagnostic::error(
            "S1063",
            "operation columns must be a whole number of 256-element Q4_K super-blocks",
            Some(block.span),
        ));
    }
    let alignment = field(block, "alignment")
        .map(|field| parse_u64(&field.value, "alignment", block, diagnostics))
        .unwrap_or(0);
    if alignment == 0 {
        diagnostics.push(Diagnostic::error(
            "S1068",
            "operation must declare an explicit alignment",
            Some(block.span),
        ));
    }
    let source_tier = parse_memory_tier(
        &required_atom(block, "source", diagnostics),
        block,
        diagnostics,
    );
    let destination_tier = parse_memory_tier(
        &required_atom(block, "destination", diagnostics),
        block,
        diagnostics,
    );
    let requirements = fields(block, "require")
        .iter()
        .filter_map(|field| atom(&field.value).map(str::to_string))
        .collect::<Vec<_>>();
    let require_exact_checksum = requirements
        .iter()
        .any(|value| value == "exact_payload_checksum");
    let reference_tolerance = field(block, "reference_tolerance")
        .map(|field| parse_number(&field.value, "reference_tolerance", block, diagnostics))
        .unwrap_or(0.0);
    let fallback = required_atom(block, "fallback", diagnostics);
    let generation = Generation(
        field(block, "generation")
            .map(|field| parse_u64(&field.value, "generation", block, diagnostics))
            .unwrap_or(0),
    );
    OperationSpec {
        name: block.name.clone(),
        model_root,
        package_root,
        payload_entry,
        source_tensor,
        kernel,
        quant_format,
        rows,
        columns,
        alignment,
        source_tier,
        destination_tier,
        require_exact_checksum,
        reference_tolerance,
        fallback,
        generation,
    }
}

fn check_known_fields(block: &Block, known: &[&str], diagnostics: &mut Diagnostics) {
    for field in &block.fields {
        if !known.contains(&field.key.as_str()) {
            diagnostics.push(Diagnostic::error(
                "S1046",
                format!(
                    "unknown field `{}` in {} declaration",
                    field.key, block.kind
                ),
                Some(field.span),
            ));
        }
    }
}

fn check_phase_graph(phases: &[ExecutionPhaseSpec], diagnostics: &mut Diagnostics) {
    let names = phases
        .iter()
        .map(|phase| phase.name.as_str())
        .collect::<HashSet<_>>();
    for phase in phases {
        for dependency in &phase.dependencies {
            if !names.contains(dependency.as_str()) {
                diagnostics.push(Diagnostic::error(
                    "S1047",
                    format!(
                        "phase `{}` depends on missing phase `{dependency}`",
                        phase.name
                    ),
                    None,
                ));
            }
        }
    }
    let mut graph = HashMap::<&str, Vec<&str>>::new();
    for phase in phases {
        graph.insert(
            phase.name.as_str(),
            phase.dependencies.iter().map(String::as_str).collect(),
        );
    }
    fn visit<'a>(
        node: &'a str,
        graph: &HashMap<&'a str, Vec<&'a str>>,
        visiting: &mut HashSet<&'a str>,
        done: &mut HashSet<&'a str>,
    ) -> bool {
        if done.contains(node) {
            return false;
        }
        if !visiting.insert(node) {
            return true;
        }
        let cycle = graph.get(node).is_some_and(|dependencies| {
            dependencies
                .iter()
                .any(|dependency| visit(dependency, graph, visiting, done))
        });
        visiting.remove(node);
        done.insert(node);
        cycle
    }
    let mut visiting = HashSet::new();
    let mut done = HashSet::new();
    for phase in phases {
        if visit(phase.name.as_str(), &graph, &mut visiting, &mut done) {
            diagnostics.push(Diagnostic::error(
                "S1048",
                "execution phase dependency graph contains a cycle",
                None,
            ));
            break;
        }
    }
}

fn check_cross_constraints(
    target: &TargetSpec,
    model: &ModelIdentity,
    quality: &QualitySpec,
    decode: &DecodeSpec,
    tensors: &[TensorSpec],
    diagnostics: &mut Diagnostics,
) {
    // A physical plan requires a hardware identity (gpu name and wave size).
    // A missing identity is rejected separately from a missing budget so the
    // failure is unambiguous (S1058).
    if target.gpu.is_empty() || target.wave == 0 || target.gpu_arch.is_empty() {
        diagnostics.push(Diagnostic::error(
            "S1058",
            "physical plan requires a hardware identity: target must declare gpu, gpu_arch, and wave",
            None,
        ));
    }
    if target.vram_budget_bytes == 0 || target.host_ram_budget_bytes == 0 {
        diagnostics.push(Diagnostic::error(
            "S1049",
            "target is missing hard budgets (vram_budget and host_ram_budget)",
            None,
        ));
    }
    let tensor_vram = tensors
        .iter()
        .map(|tensor| tensor.required_vram_bytes)
        .sum::<u64>();
    let total_vram = tensor_vram.saturating_add(decode.cost.vram_bytes);
    if total_vram > target.vram_budget_bytes {
        diagnostics.push(Diagnostic::error(
            "S1050",
            format!(
                "hard VRAM capacity exceeded: required {total_vram} bytes > budget {} bytes",
                target.vram_budget_bytes
            ),
            None,
        ));
    }
    if quality.exact && !quality.fallback_required {
        diagnostics.push(Diagnostic::error(
            "S1051",
            "approximation cannot satisfy exact authority without fallback",
            None,
        ));
    }
    if quality.exact
        && tensors
            .iter()
            .any(|tensor| tensor.authority == ValueAuthority::Approximate)
        && !quality.fallback_required
    {
        diagnostics.push(Diagnostic::error(
            "S1052",
            "approximate tensor flow cannot terminate at exact authority without fallback",
            None,
        ));
    }
    if let Some(required_epoch) = &decode.required_epoch {
        if required_epoch != &model.epoch {
            diagnostics.push(Diagnostic::error(
                "S1053",
                format!(
                    "stale epoch: decode requires {} but model identity is {}",
                    required_epoch.0, model.epoch.0
                ),
                None,
            ));
        }
    }
    // Stale generation: every non-zero tensor generation must match the model
    // epoch that the decode policy is pinned to (S1055).
    if let Some(required_epoch) = &decode.required_epoch {
        for tensor in tensors {
            if tensor.generation.0 != 0 && tensor.generation.0 != required_epoch.0 {
                diagnostics.push(Diagnostic::error(
                    "S1055",
                    format!(
                        "stale generation: tensor `{}` has generation {} but the decode epoch is {}",
                        tensor.name, tensor.generation.0, required_epoch.0
                    ),
                    None,
                ));
            }
        }
    }
    // Model-root mismatch: any declared tensor model_root or decode
    // requires_model must equal the model identity root (S1057).
    for tensor in tensors {
        if !tensor.model_root.is_empty() && tensor.model_root != model.model_sha256 {
            diagnostics.push(Diagnostic::error(
                "S1057",
                format!(
                    "model-root mismatch: tensor `{}` roots to {} but model identity is {}",
                    tensor.name, tensor.model_root, model.model_sha256
                ),
                None,
            ));
        }
    }
    if !decode.requires_model.is_empty() && decode.requires_model != model.model_sha256 {
        diagnostics.push(Diagnostic::error(
            "S1057",
            format!(
                "model-root mismatch: decode requires {} but model identity is {}",
                decode.requires_model, model.model_sha256
            ),
            None,
        ));
    }
}
