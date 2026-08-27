use crate::policy::CompiledContextPolicy;
use crate::store::{KVPageStore, StoreError};
use har_kv::{
    KVFallbackPlan, KVFormat, KVPageId, KVPageLease, KVPageRecord, KVQualityCertificate,
    KVRepresentation, KVTelemetry, MetricBounds, PageEventKind, PageSelection,
    PhysicalPageLocation, PrefixGraph, PrefixGraphError, ReconstructionTicket, RepresentationKind,
    StableDigest,
};
use std::collections::HashSet;
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContextFoldError {
    PolicyNotAdmitted,
    Graph(PrefixGraphError),
    Store(StoreError),
    MissingLocation,
    MissingRepresentation,
    InvalidTicket,
}
impl fmt::Display for ContextFoldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PolicyNotAdmitted => {
                formatter.write_str("context policy is not admitted by its tail-probe gate")
            }
            Self::Graph(error) => write!(formatter, "graph error: {error}"),
            Self::Store(error) => write!(formatter, "store error: {error}"),
            Self::MissingLocation => {
                formatter.write_str("representation has no physical store location")
            }
            Self::MissingRepresentation => formatter.write_str("representation not registered"),
            Self::InvalidTicket => formatter.write_str("reconstruction ticket is stale or invalid"),
        }
    }
}
impl std::error::Error for ContextFoldError {}
impl From<PrefixGraphError> for ContextFoldError {
    fn from(value: PrefixGraphError) -> Self {
        Self::Graph(value)
    }
}
impl From<StoreError> for ContextFoldError {
    fn from(value: StoreError) -> Self {
        Self::Store(value)
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum ContextFoldSelection {
    Representation {
        kind: RepresentationKind,
        ticket: ReconstructionTicket,
    },
    Fallback(KVFallbackPlan),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContextFoldManifest {
    pub schema: String,
    pub policy_id: StableDigest,
    pub graph_generation: u64,
    pub store_id: String,
    pub page_count: usize,
    pub telemetry_events: usize,
    pub fallback_rate: f64,
    pub exact_tail_retrieval_rate: f64,
    pub global_materialization: bool,
}

impl ContextFoldManifest {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"schema\":\"{}\",\"policy_id\":\"{}\",\"graph_generation\":{},\"store_id\":\"{}\",\"page_count\":{},\"telemetry_events\":{},\"fallback_rate\":{},\"exact_tail_retrieval_rate\":{},\"global_materialization\":false}}",
            self.schema, self.policy_id, self.graph_generation, self.store_id, self.page_count, self.telemetry_events, self.fallback_rate, self.exact_tail_retrieval_rate
        )
    }
}

/// Rust owner of logical prefix/page semantics. Physical storage is injected
/// through residency layer's `KVPageStore` trait.
pub struct ContextFoldController<S: KVPageStore> {
    pub graph: PrefixGraph,
    pub store: S,
    pub policy: CompiledContextPolicy,
    pub telemetry: KVTelemetry,
    pub archive_ref: String,
    pub authority_root: StableDigest,
    hot_pages: HashSet<KVPageId>,
}

impl<S: KVPageStore> ContextFoldController<S> {
    pub fn new(
        store: S,
        policy: CompiledContextPolicy,
        archive_ref: impl Into<String>,
        authority_root: StableDigest,
    ) -> Result<Self, ContextFoldError> {
        if !policy.admitted() {
            return Err(ContextFoldError::PolicyNotAdmitted);
        }
        Ok(Self {
            graph: PrefixGraph::new(),
            store,
            policy,
            telemetry: KVTelemetry::default(),
            archive_ref: archive_ref.into(),
            authority_root,
            hot_pages: HashSet::new(),
        })
    }

    pub fn register_reference_page(
        &mut self,
        page: KVPageRecord,
        exact_q8_bytes: u64,
        archive_bytes: u64,
    ) -> Result<(), ContextFoldError> {
        let page_id = page.id.clone();
        self.graph.attach_page(page)?;
        let generation = self.graph.generation();
        self.graph.add_representation(
            &page_id,
            KVRepresentation::ExactHotQ8 {
                bytes: exact_q8_bytes,
                location: None,
            },
            generation,
        )?;
        self.hot_pages.insert(page_id.clone());
        self.graph.add_representation(
            &page_id,
            KVRepresentation::ExactTokenArchive {
                bytes: archive_bytes,
                archive_ref: self.archive_ref.clone(),
                location: None,
            },
            generation,
        )?;
        self.telemetry.page_event(
            PageEventKind::PageSelected,
            &page_id,
            Some(RepresentationKind::ExactHotQ8),
            generation,
            exact_q8_bytes,
        );
        Ok(())
    }

    pub fn persist_latent_page(
        &mut self,
        page_id: &KVPageId,
        codec: impl Into<String>,
        payload: Vec<u8>,
        quality: KVQualityCertificate,
        source_format: KVFormat,
    ) -> Result<PhysicalPageLocation, ContextFoldError> {
        let codec = codec.into();
        let generation = self.graph.generation();
        let location = self.store.put(page_id, payload, generation)?;
        let bytes = location.length;
        let representation = KVRepresentation::ContextFoldLatent {
            codec: codec.clone(),
            bytes,
            quality,
            location: Some(location.clone()),
        };
        if let Err(error) = self
            .graph
            .add_representation(page_id, representation, generation)
        {
            let _ = self.store.remove(page_id, &location);
            return Err(error.into());
        }
        self.telemetry.record(har_kv::PageEvent {
            sequence: 0,
            kind: PageEventKind::Promotion,
            page_id: Some(page_id.clone()),
            prefix_id: Some(page_id.prefix_node_id.to_string()),
            representation: Some(RepresentationKind::ContextFoldLatent),
            generation,
            codec: Some(codec),
            bytes,
            quality_error: None,
            fallback: false,
            value_credit: 0.0,
        });
        let _ = source_format; // recorded by the external codec descriptor/manifest
        Ok(location)
    }

    pub fn persist_lossless_page(
        &mut self,
        page_id: &KVPageId,
        payload: Vec<u8>,
    ) -> Result<PhysicalPageLocation, ContextFoldError> {
        let generation = self.graph.generation();
        let location = self.store.put(page_id, payload, generation)?;
        let representation = KVRepresentation::LosslessCold {
            bytes: location.length,
            location: Some(location.clone()),
        };
        if let Err(error) = self
            .graph
            .add_representation(page_id, representation, generation)
        {
            let _ = self.store.remove(page_id, &location);
            return Err(error.into());
        }
        Ok(location)
    }

    /// Apply the simple recent-page policy without changing logical IDs.
    /// Pages intersecting the half-open recent window are exact-hot only when
    /// an exact reference representation exists; all other pages are cold.
    pub fn set_recent_hot(
        &mut self,
        current_token: u64,
        hot_window_tokens: u64,
    ) -> Result<(), ContextFoldError> {
        if hot_window_tokens == 0 {
            return Err(ContextFoldError::Graph(PrefixGraphError::InvalidPage));
        }
        let cutoff = current_token.saturating_sub(hot_window_tokens);
        for page_id in self.graph.page_ids() {
            let is_recent = page_id.token_end > cutoff;
            let has_exact = self
                .graph
                .page(&page_id)
                .and_then(|page| page.find(RepresentationKind::ExactHotQ8))
                .is_some();
            if is_recent && has_exact {
                self.hot_pages.insert(page_id);
            } else {
                self.hot_pages.remove(&page_id);
            }
        }
        Ok(())
    }

    pub fn mark_page_hot(&mut self, page_id: &KVPageId) -> Result<(), ContextFoldError> {
        if self.graph.page(page_id).is_none() {
            return Err(ContextFoldError::Graph(PrefixGraphError::UnknownPage));
        }
        self.hot_pages.insert(page_id.clone());
        Ok(())
    }

    pub fn mark_page_cold(&mut self, page_id: &KVPageId) -> Result<(), ContextFoldError> {
        if self.graph.page(page_id).is_none() {
            return Err(ContextFoldError::Graph(PrefixGraphError::UnknownPage));
        }
        self.hot_pages.remove(page_id);
        Ok(())
    }

    pub fn promote_warm(
        &mut self,
        page_id: &KVPageId,
        reconstructed_bytes: u64,
        codec: impl Into<String>,
    ) -> Result<(), ContextFoldError> {
        let generation = self.graph.generation();
        self.graph.add_representation(
            page_id,
            KVRepresentation::ReconstructedWarm {
                bytes: reconstructed_bytes,
                source_codec: codec.into(),
                location: None,
            },
            generation,
        )?;
        self.telemetry.page_event(
            PageEventKind::Promotion,
            page_id,
            Some(RepresentationKind::ReconstructedWarm),
            generation,
            reconstructed_bytes,
        );
        Ok(())
    }

    pub fn select_for_attention(
        &mut self,
        page_id: &KVPageId,
        required: Option<&MetricBounds>,
    ) -> Result<ContextFoldSelection, ContextFoldError> {
        let generation = self.graph.generation();
        // Exact recent pages remain usable even when the compressed cold
        // policy is not admitted by its tail probe. The gate protects only
        // approximate ContextFold selection, never the reference authority.
        if self.hot_pages.contains(page_id)
            && self
                .graph
                .page(page_id)
                .and_then(|page| page.find(RepresentationKind::ExactHotQ8))
                .is_some()
        {
            let ticket = self.ticket_for(page_id, RepresentationKind::ExactHotQ8)?;
            self.telemetry.page_event(
                PageEventKind::PageSelected,
                page_id,
                Some(RepresentationKind::ExactHotQ8),
                generation,
                0,
            );
            return Ok(ContextFoldSelection::Representation {
                kind: RepresentationKind::ExactHotQ8,
                ticket,
            });
        }
        if self.policy.require_tail_probe_pass && !self.policy.tail_probe_pass {
            let fallback = KVFallbackPlan::strict(
                page_id.clone(),
                self.archive_ref.clone(),
                self.authority_root.clone(),
                "tail_probe_pass is required but absent",
            );
            self.telemetry
                .fallback(page_id, generation, "tail_probe_gate");
            return Ok(ContextFoldSelection::Fallback(fallback));
        }
        let selection = self.graph.select_with_hot(
            page_id,
            self.hot_pages.contains(page_id),
            required,
            &self.archive_ref,
            self.authority_root.clone(),
        )?;
        match selection {
            PageSelection::Representation(kind) => {
                let ticket = self.ticket_for(page_id, kind)?;
                self.telemetry.record(har_kv::PageEvent {
                    sequence: 0,
                    kind: PageEventKind::ReconstructionRequested,
                    page_id: Some(page_id.clone()),
                    prefix_id: Some(page_id.prefix_node_id.to_string()),
                    representation: Some(kind),
                    generation,
                    codec: Some(ticket.codec.clone()),
                    bytes: 0,
                    quality_error: None,
                    fallback: false,
                    value_credit: 0.0,
                });
                Ok(ContextFoldSelection::Representation { kind, ticket })
            }
            PageSelection::Fallback(plan) => {
                self.telemetry
                    .fallback(page_id, generation, "quality_or_representation_gate");
                Ok(ContextFoldSelection::Fallback(plan))
            }
        }
    }

    pub fn ticket_for(
        &self,
        page_id: &KVPageId,
        kind: RepresentationKind,
    ) -> Result<ReconstructionTicket, ContextFoldError> {
        let page = self
            .graph
            .page(page_id)
            .ok_or(ContextFoldError::Graph(PrefixGraphError::UnknownPage))?;
        let representation = page
            .find(kind)
            .ok_or(ContextFoldError::MissingRepresentation)?;
        let codec = match representation {
            KVRepresentation::ExactHotQ8 { .. } => "native_q8".to_string(),
            KVRepresentation::ReconstructedWarm { source_codec, .. } => source_codec.clone(),
            KVRepresentation::ContextFoldLatent { codec, .. } => codec.clone(),
            KVRepresentation::LosslessCold { .. } => "lossless_q8".to_string(),
            KVRepresentation::ExactTokenArchive { .. } => "token_archive_replay".to_string(),
            KVRepresentation::ReconstructionScratch { ticket_id, .. } => ticket_id.clone(),
        };
        let ticket_id = StableDigest::from_parts(&[
            page_id.key().as_bytes(),
            codec.as_bytes(),
            &self.graph.generation().to_le_bytes(),
        ]);
        Ok(ReconstructionTicket {
            ticket_id: ticket_id.to_string(),
            page_id: page_id.clone(),
            codec,
            source_kind: kind,
            requested_format: KVFormat::ReconstructedF16,
            scratch_bytes: representation.bytes(),
            output_lifetime: "CALLBACK_ONLY".to_string(),
            synchronization: "CALLBACK_FENCE".to_string(),
            generation: self.graph.generation(),
            source_digest: page.closure.root(),
        })
    }

    pub fn acquire_lease(
        &mut self,
        page_id: &KVPageId,
        kind: RepresentationKind,
        owner: impl Into<String>,
    ) -> Result<KVPageLease, ContextFoldError> {
        let generation = self.graph.generation();
        Ok(self.graph.lease(page_id, kind, owner, generation)?)
    }

    pub fn release_lease(&mut self, lease: KVPageLease) -> Result<(), ContextFoldError> {
        self.graph.release_lease(&lease.lease_id)?;
        self.telemetry.page_event(
            PageEventKind::LeaseReleased,
            &lease.page_id,
            Some(lease.representation),
            self.graph.generation(),
            0,
        );
        Ok(())
    }

    pub fn load_representation_payload(
        &self,
        page_id: &KVPageId,
        kind: RepresentationKind,
    ) -> Result<Vec<u8>, ContextFoldError> {
        let page = self
            .graph
            .page(page_id)
            .ok_or(ContextFoldError::Graph(PrefixGraphError::UnknownPage))?;
        let representation = page
            .find(kind)
            .ok_or(ContextFoldError::MissingRepresentation)?;
        let location = match representation {
            KVRepresentation::ContextFoldLatent { location, .. }
            | KVRepresentation::LosslessCold { location, .. } => {
                location.as_ref().ok_or(ContextFoldError::MissingLocation)?
            }
            _ => return Err(ContextFoldError::MissingLocation),
        };
        Ok(self.store.get(page_id, location)?)
    }

    pub fn safe_evict(
        &mut self,
        page_id: &KVPageId,
        kind: RepresentationKind,
    ) -> Result<(), ContextFoldError> {
        let generation = self.graph.generation();
        let representation = self.graph.evict(page_id, kind, generation)?;
        let location = match &representation {
            KVRepresentation::ContextFoldLatent { location, .. }
            | KVRepresentation::LosslessCold { location, .. } => location.clone(),
            _ => None,
        };
        if let Some(location) = location {
            self.store.remove(page_id, &location)?;
        }
        self.telemetry.page_event(
            PageEventKind::Eviction,
            page_id,
            Some(kind),
            generation,
            representation.bytes(),
        );
        Ok(())
    }

    pub fn manifest(&self) -> ContextFoldManifest {
        ContextFoldManifest {
            schema: "har-contextfold-rust-manifest-v1".to_string(),
            policy_id: self.policy.policy_id.clone(),
            graph_generation: self.graph.generation(),
            store_id: self.store.store_id().to_string(),
            page_count: self.graph.page_count(),
            telemetry_events: self.telemetry.events.len(),
            fallback_rate: self.telemetry.fallback_rate(),
            exact_tail_retrieval_rate: self.telemetry.exact_tail_retrieval_rate(),
            global_materialization: false,
        }
    }
}
