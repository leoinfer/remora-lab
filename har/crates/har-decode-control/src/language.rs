//! HAR language V0's immutable, parser-free decode plan join.
//!
//! This module is additive to the existing compiler decode descriptor.  It has
//! no dependency on syntax/AST types and exposes no source evaluator.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryTier {
    DirectNvme,
    RamMapped,
    PinnedRam,
    Vram,
    VramSlot,
    Scratch,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingLocation {
    Cpu,
    Vulkan,
    Backend,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FallbackPolicy {
    DepthZero,
    ExactTarget,
    Required,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HorizonMode {
    Fixed,
    Elastic,
    FailSafe,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeObjective {
    AcceptedTokensPerCompleteCost,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Epoch(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Generation(pub u64);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelIdentity {
    model_sha256: String,
    build_hash: String,
    config_hash: String,
    epoch: Epoch,
}
impl ModelIdentity {
    pub fn new(
        model_sha256: impl Into<String>,
        build_hash: impl Into<String>,
        config_hash: impl Into<String>,
        epoch: Epoch,
    ) -> Self {
        Self {
            model_sha256: model_sha256.into(),
            build_hash: build_hash.into(),
            config_hash: config_hash.into(),
            epoch,
        }
    }
    pub fn model_sha256(&self) -> &str {
        &self.model_sha256
    }
    pub fn build_hash(&self) -> &str {
        &self.build_hash
    }
    pub fn config_hash(&self) -> &str {
        &self.config_hash
    }
    pub fn epoch(&self) -> Epoch {
        self.epoch
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResourceCost {
    pub required_nvme_bytes: u64,
    pub ram_to_vram_bytes: u64,
    pub vram_bytes: u64,
    pub verification_compute: f64,
    pub queue_slots: u64,
}
impl ResourceCost {
    pub const ZERO: Self = Self {
        required_nvme_bytes: 0,
        ram_to_vram_bytes: 0,
        vram_bytes: 0,
        verification_compute: 0.0,
        queue_slots: 0,
    };
    pub fn complete_scalar(self) -> f64 {
        self.required_nvme_bytes as f64
            + self.ram_to_vram_bytes as f64
            + self.vram_bytes as f64
            + self.verification_compute
            + self.queue_slots as f64
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertUnionEstimate {
    expert_ids: Vec<u32>,
    union_bytes: u64,
    resident_overlap_bytes: u64,
}
impl ExpertUnionEstimate {
    pub fn new(
        expert_ids: Vec<u32>,
        union_bytes: u64,
        resident_overlap_bytes: u64,
    ) -> Option<Self> {
        (resident_overlap_bytes <= union_bytes).then_some(Self {
            expert_ids,
            union_bytes,
            resident_overlap_bytes,
        })
    }
    pub fn expert_ids(&self) -> &[u32] {
        &self.expert_ids
    }
    pub fn union_bytes(&self) -> u64 {
        self.union_bytes
    }
    pub fn resident_overlap_bytes(&self) -> u64 {
        self.resident_overlap_bytes
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceSlackSnapshot {
    pub snapshot_id: String,
    pub available_nvme_bytes: u64,
    pub available_ram_to_vram_bytes: u64,
    pub available_vram_bytes: u64,
    pub available_queue_slots: u64,
}
impl ResourceSlackSnapshot {
    pub fn can_cover(&self, cost: ResourceCost) -> bool {
        cost.required_nvme_bytes <= self.available_nvme_bytes
            && cost.ram_to_vram_bytes <= self.available_ram_to_vram_bytes
            && cost.vram_bytes <= self.available_vram_bytes
            && cost.queue_slots <= self.available_queue_slots
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecodePolicyPlan {
    name: String,
    min_horizon: u8,
    max_horizon: u8,
    mode: HorizonMode,
    objective: DecodeObjective,
    require_exact_acceptance: bool,
    sampling_location: SamplingLocation,
    fallback: FallbackPolicy,
    required_epoch: Option<Epoch>,
    base_cost: ResourceCost,
}
impl DecodePolicyPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        min_horizon: u8,
        max_horizon: u8,
        objective: DecodeObjective,
        require_exact_acceptance: bool,
        sampling_location: SamplingLocation,
        fallback: FallbackPolicy,
        required_epoch: Option<Epoch>,
        base_cost: ResourceCost,
    ) -> Option<Self> {
        (min_horizon <= max_horizon && max_horizon <= 3 && require_exact_acceptance).then_some(
            Self {
                name: name.into(),
                min_horizon,
                max_horizon,
                mode: if min_horizon == max_horizon {
                    HorizonMode::Fixed
                } else {
                    HorizonMode::Elastic
                },
                objective,
                require_exact_acceptance,
                sampling_location,
                fallback,
                required_epoch,
                base_cost,
            },
        )
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn min_horizon(&self) -> u8 {
        self.min_horizon
    }
    pub fn max_horizon(&self) -> u8 {
        self.max_horizon
    }
    pub fn mode(&self) -> HorizonMode {
        self.mode
    }
    pub fn objective(&self) -> DecodeObjective {
        self.objective
    }
    pub fn require_exact_acceptance(&self) -> bool {
        self.require_exact_acceptance
    }
    pub fn sampling_location(&self) -> SamplingLocation {
        self.sampling_location
    }
    pub fn fallback(&self) -> FallbackPolicy {
        self.fallback
    }
    pub fn required_epoch(&self) -> Option<Epoch> {
        self.required_epoch
    }
    pub fn base_cost(&self) -> ResourceCost {
        self.base_cost
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TelemetryRequirements {
    requirements: Vec<String>,
}
impl TelemetryRequirements {
    pub fn new(requirements: Vec<String>) -> Option<Self> {
        (!requirements.is_empty()).then_some(Self { requirements })
    }
    pub fn requirements(&self) -> &[String] {
        &self.requirements
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImmutableExecutionPlan {
    identity: ModelIdentity,
    required_memory_tiers: Vec<MemoryTier>,
    decode: DecodePolicyPlan,
    telemetry: TelemetryRequirements,
    exact_authority: String,
    target_vram_budget_bytes: u64,
    target_host_ram_budget_bytes: u64,
}
impl ImmutableExecutionPlan {
    pub fn new(
        identity: ModelIdentity,
        required_memory_tiers: Vec<MemoryTier>,
        decode: DecodePolicyPlan,
        telemetry: TelemetryRequirements,
        exact_authority: impl Into<String>,
        target_vram_budget_bytes: u64,
        target_host_ram_budget_bytes: u64,
    ) -> Option<Self> {
        (target_vram_budget_bytes > 0
            && target_host_ram_budget_bytes > 0
            && !required_memory_tiers.is_empty())
        .then_some(Self {
            identity,
            required_memory_tiers,
            decode,
            telemetry,
            exact_authority: exact_authority.into(),
            target_vram_budget_bytes,
            target_host_ram_budget_bytes,
        })
    }
    pub fn identity(&self) -> &ModelIdentity {
        &self.identity
    }
    pub fn required_memory_tiers(&self) -> &[MemoryTier] {
        &self.required_memory_tiers
    }
    pub fn decode(&self) -> &DecodePolicyPlan {
        &self.decode
    }
    pub fn telemetry(&self) -> &TelemetryRequirements {
        &self.telemetry
    }
    pub fn exact_authority(&self) -> &str {
        &self.exact_authority
    }
    pub fn target_vram_budget_bytes(&self) -> u64 {
        self.target_vram_budget_bytes
    }
    pub fn target_host_ram_budget_bytes(&self) -> u64 {
        self.target_host_ram_budget_bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HorizonInput<'a> {
    pub candidate_horizon: u8,
    pub mathematical_permission: bool,
    pub predicted_acceptance: f64,
    pub recent_acceptance: f64,
    pub costs_by_depth: &'a [ResourceCost],
    pub resource_slack: Option<&'a ResourceSlackSnapshot>,
    pub queue_backlog: u64,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HorizonDecision {
    pub depth: u8,
    pub predicted_acceptance: f64,
    pub expected_accepted_tokens: f64,
    pub incremental_cost: f64,
    pub fail_safe: bool,
}
pub fn choose_horizon(plan: &ImmutableExecutionPlan, input: HorizonInput<'_>) -> HorizonDecision {
    let zero = HorizonDecision {
        depth: 0,
        predicted_acceptance: input.predicted_acceptance,
        expected_accepted_tokens: 0.0,
        incremental_cost: 0.0,
        fail_safe: true,
    };
    if !input.mathematical_permission
        || input.candidate_horizon == 0
        || input.queue_backlog > 1_000_000
        || input.predicted_acceptance <= 0.0
    {
        return zero;
    }
    let max_depth = input
        .candidate_horizon
        .min(plan.decode().max_horizon())
        .min(3);
    if plan.decode().mode() == HorizonMode::Fixed && plan.decode().min_horizon() > 0 {
        let depth = plan.decode().min_horizon();
        let Some(cost) = input.costs_by_depth.get(depth as usize).copied() else {
            return zero;
        };
        if depth > input.candidate_horizon
            || input
                .resource_slack
                .is_some_and(|slack| !slack.can_cover(cost))
        {
            return zero;
        }
        return HorizonDecision {
            depth,
            predicted_acceptance: input.predicted_acceptance,
            expected_accepted_tokens: f64::from(depth) * input.predicted_acceptance,
            incremental_cost: cost.complete_scalar(),
            fail_safe: false,
        };
    }
    let mut best = zero;
    let mut best_score = 0.0;
    for depth in 1..=max_depth {
        let Some(cost) = input.costs_by_depth.get(depth as usize).copied() else {
            return zero;
        };
        if input
            .resource_slack
            .is_some_and(|slack| !slack.can_cover(cost))
        {
            continue;
        }
        let expected = f64::from(depth) * input.predicted_acceptance;
        let score = expected / cost.complete_scalar().max(f64::MIN_POSITIVE);
        if score > best_score {
            best = HorizonDecision {
                depth,
                predicted_acceptance: input.predicted_acceptance,
                expected_accepted_tokens: expected,
                incremental_cost: cost.complete_scalar(),
                fail_safe: false,
            };
            best_score = score;
        }
    }
    best
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerificationSpan {
    pub start_position: u64,
    pub candidate_token_count: u8,
    pub target_batch_id: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcceptedPrefix {
    pub token_ids: Vec<u32>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RejectedSuffix {
    pub token_ids: Vec<u32>,
    pub first_rejected_index: u8,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rollback {
    pub generation_before: Generation,
    pub generation_after: Generation,
    pub discarded_tokens: Vec<u32>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeculativeResult {
    pub accepted: AcceptedPrefix,
    pub rejected: Option<RejectedSuffix>,
    pub rollback: Option<Rollback>,
    pub authoritative: bool,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceQuery {
    pub required_tier: MemoryTier,
    pub generation: Generation,
    pub epoch: Epoch,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plan() -> ImmutableExecutionPlan {
        let identity = ModelIdentity::new("m", "b", "c", Epoch(1));
        let policy = DecodePolicyPlan::new(
            "q",
            0,
            3,
            DecodeObjective::AcceptedTokensPerCompleteCost,
            true,
            SamplingLocation::Cpu,
            FallbackPolicy::DepthZero,
            Some(Epoch(1)),
            ResourceCost::ZERO,
        )
        .unwrap();
        ImmutableExecutionPlan::new(
            identity,
            vec![MemoryTier::VramSlot],
            policy,
            TelemetryRequirements::new(vec!["accepted_tokens".into()]).unwrap(),
            "full_model",
            1,
            1,
        )
        .unwrap()
    }
    #[test]
    fn horizon_fails_safe_when_slack_is_insufficient() {
        let costs = [
            ResourceCost::ZERO,
            ResourceCost {
                vram_bytes: 10,
                verification_compute: 1.0,
                ..ResourceCost::ZERO
            },
            ResourceCost {
                vram_bytes: 20,
                verification_compute: 2.0,
                ..ResourceCost::ZERO
            },
            ResourceCost {
                vram_bytes: 30,
                verification_compute: 3.0,
                ..ResourceCost::ZERO
            },
        ];
        let slack = ResourceSlackSnapshot {
            snapshot_id: "s".into(),
            available_nvme_bytes: 0,
            available_ram_to_vram_bytes: 0,
            available_vram_bytes: 15,
            available_queue_slots: 2,
        };
        let decision = choose_horizon(
            &plan(),
            HorizonInput {
                candidate_horizon: 3,
                mathematical_permission: true,
                predicted_acceptance: 0.8,
                recent_acceptance: 0.8,
                costs_by_depth: &costs,
                resource_slack: Some(&slack),
                queue_backlog: 0,
            },
        );
        assert_eq!(decision.depth, 1);
        assert!(!decision.fail_safe);
    }
}
