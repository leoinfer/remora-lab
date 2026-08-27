use crate::certificate::KVQualityCertificate;
use crate::identity::{KVDependencyClosure, StableDigest};
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KVFormat {
    ExactQ8,
    ReconstructedF16,
    ReconstructedQ8,
    ContextFoldLatent,
    LosslessQ8,
    TokenArchive,
}

impl KVFormat {
    pub const F16: Self = Self::ReconstructedF16;
    pub const Q8_0: Self = Self::ExactQ8;
    pub const INT4: Self = Self::ContextFoldLatent;
    pub const INT2: Self = Self::ContextFoldLatent;
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KVStorageTier {
    ExactHot,
    ReconstructedWarm,
    LatentCold,
    LosslessCold,
    TokenArchive,
    ReconstructionScratch,
}

impl KVStorageTier {
    pub const EXACT_HOT: Self = Self::ExactHot;
    pub const RECONSTRUCTED_WARM: Self = Self::ReconstructedWarm;
    pub const LATENT_COLD: Self = Self::LatentCold;
    pub const LOSSLESS_COLD: Self = Self::LosslessCold;
    pub const TOKEN_ARCHIVE: Self = Self::TokenArchive;
    pub const RECONSTRUCTION_SCRATCH: Self = Self::ReconstructionScratch;
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RepresentationKind {
    ExactHotQ8,
    ReconstructedWarm,
    ContextFoldLatent,
    LosslessColdPage,
    ExactTokenArchive,
    ReconstructionScratch,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct KVPageId {
    pub prefix_node_id: crate::graph::PrefixNodeId,
    pub ordinal: u32,
    pub token_start: u64,
    pub token_end: u64,
    pub layer_start: u32,
    pub layer_end: u32,
    pub head_start: u32,
    pub head_end: u32,
    pub logical_generation: u64,
}

impl KVPageId {
    pub fn token_count(&self) -> u64 {
        self.token_end.saturating_sub(self.token_start)
    }

    pub fn layer_count(&self) -> u32 {
        self.layer_end.saturating_sub(self.layer_start)
    }

    pub fn head_count(&self) -> u32 {
        self.head_end.saturating_sub(self.head_start)
    }

    pub fn key(&self) -> String {
        format!(
            "{}:{}:{}-{}:L{}-{}:H{}-{}:g{}",
            self.prefix_node_id,
            self.ordinal,
            self.token_start,
            self.token_end,
            self.layer_start,
            self.layer_end,
            self.head_start,
            self.head_end,
            self.logical_generation,
        )
    }

    pub fn valid(&self) -> bool {
        self.token_end > self.token_start
            && self.layer_end > self.layer_start
            && self.head_end >= self.head_start
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhysicalPageLocation {
    pub store_id: String,
    pub offset: u64,
    pub length: u64,
    pub generation: u64,
}

/// A representation is a physical choice for one logical page.  The logical
/// page ID is held by `KVPageRecord`; moving a representation never changes it.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum KVRepresentation {
    ExactHotQ8 {
        bytes: u64,
        location: Option<PhysicalPageLocation>,
    },
    ReconstructedWarm {
        bytes: u64,
        source_codec: String,
        location: Option<PhysicalPageLocation>,
    },
    ContextFoldLatent {
        codec: String,
        bytes: u64,
        quality: KVQualityCertificate,
        location: Option<PhysicalPageLocation>,
    },
    LosslessCold {
        bytes: u64,
        location: Option<PhysicalPageLocation>,
    },
    ExactTokenArchive {
        bytes: u64,
        archive_ref: String,
        location: Option<PhysicalPageLocation>,
    },
    ReconstructionScratch {
        bytes: u64,
        ticket_id: String,
    },
}

impl KVRepresentation {
    pub fn kind(&self) -> RepresentationKind {
        match self {
            Self::ExactHotQ8 { .. } => RepresentationKind::ExactHotQ8,
            Self::ReconstructedWarm { .. } => RepresentationKind::ReconstructedWarm,
            Self::ContextFoldLatent { .. } => RepresentationKind::ContextFoldLatent,
            Self::LosslessCold { .. } => RepresentationKind::LosslessColdPage,
            Self::ExactTokenArchive { .. } => RepresentationKind::ExactTokenArchive,
            Self::ReconstructionScratch { .. } => RepresentationKind::ReconstructionScratch,
        }
    }

    pub fn tier(&self) -> KVStorageTier {
        match self {
            Self::ExactHotQ8 { .. } => KVStorageTier::ExactHot,
            Self::ReconstructedWarm { .. } => KVStorageTier::ReconstructedWarm,
            Self::ContextFoldLatent { .. } => KVStorageTier::LatentCold,
            Self::LosslessCold { .. } => KVStorageTier::LosslessCold,
            Self::ExactTokenArchive { .. } => KVStorageTier::TokenArchive,
            Self::ReconstructionScratch { .. } => KVStorageTier::ReconstructionScratch,
        }
    }

    pub fn bytes(&self) -> u64 {
        match self {
            Self::ExactHotQ8 { bytes, .. }
            | Self::ReconstructedWarm { bytes, .. }
            | Self::ContextFoldLatent { bytes, .. }
            | Self::LosslessCold { bytes, .. }
            | Self::ExactTokenArchive { bytes, .. }
            | Self::ReconstructionScratch { bytes, .. } => *bytes,
        }
    }

    pub fn is_exact(&self) -> bool {
        matches!(
            self,
            Self::ExactHotQ8 { .. } | Self::ExactTokenArchive { .. } | Self::LosslessCold { .. }
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KVPageRecord {
    pub id: KVPageId,
    pub closure: KVDependencyClosure,
    pub representations: Vec<KVRepresentation>,
    pub reference_count: u64,
    pub active_leases: u64,
    pub last_access_epoch: u64,
}

impl KVPageRecord {
    pub fn new(id: KVPageId, closure: KVDependencyClosure) -> Self {
        Self {
            id,
            closure,
            representations: Vec::new(),
            reference_count: 0,
            active_leases: 0,
            last_access_epoch: 0,
        }
    }

    pub fn add_representation(&mut self, representation: KVRepresentation) {
        let kind = representation.kind();
        self.representations
            .retain(|candidate| candidate.kind() != kind);
        self.representations.push(representation);
    }

    pub fn find(&self, kind: RepresentationKind) -> Option<&KVRepresentation> {
        self.representations
            .iter()
            .find(|representation| representation.kind() == kind)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KVPageLease {
    pub lease_id: String,
    pub page_id: KVPageId,
    pub representation: RepresentationKind,
    pub generation: u64,
    pub owner: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReconstructionTicket {
    pub ticket_id: String,
    pub page_id: KVPageId,
    pub codec: String,
    pub source_kind: RepresentationKind,
    pub requested_format: KVFormat,
    pub scratch_bytes: u64,
    pub output_lifetime: String,
    pub synchronization: String,
    pub generation: u64,
    pub source_digest: StableDigest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FallbackStep {
    LatentDecode,
    HigherResidual { codec: String },
    WidenPage { extra_tokens: u32 },
    TokenArchiveReplay { archive_ref: String },
    FullReference { authority_root: StableDigest },
}

#[derive(Clone, Debug, PartialEq)]
pub struct KVFallbackPlan {
    pub page_id: KVPageId,
    pub steps: Vec<FallbackStep>,
    pub reason: String,
    pub required_quality: Option<KVQualityCertificate>,
}

impl KVFallbackPlan {
    pub fn strict(
        page_id: KVPageId,
        archive_ref: impl Into<String>,
        authority_root: StableDigest,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            page_id,
            steps: vec![
                FallbackStep::LatentDecode,
                FallbackStep::HigherResidual {
                    codec: "pca_residual_v0_high".to_string(),
                },
                FallbackStep::WidenPage { extra_tokens: 32 },
                FallbackStep::TokenArchiveReplay {
                    archive_ref: archive_ref.into(),
                },
                FallbackStep::FullReference { authority_root },
            ],
            reason: reason.into(),
            required_quality: None,
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.page_id.valid() || self.steps.is_empty() || self.reason.is_empty() {
            return Err("fallback plan is incomplete");
        }
        Ok(())
    }
}

impl fmt::Display for KVPageId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.key())
    }
}
