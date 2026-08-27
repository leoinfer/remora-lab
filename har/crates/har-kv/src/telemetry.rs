use crate::identity::StableDigest;
use crate::page::{KVPageId, RepresentationKind};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PageEventKind {
    PrefixLookup,
    PageSelected,
    LeaseAcquired,
    LeaseReleased,
    Promotion,
    Eviction,
    ReconstructionRequested,
    FallbackEscalated,
    ExactTailRetrieved,
    StaleGenerationRejected,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PageEvent {
    pub sequence: u64,
    pub kind: PageEventKind,
    pub page_id: Option<KVPageId>,
    pub prefix_id: Option<String>,
    pub representation: Option<RepresentationKind>,
    pub generation: u64,
    pub codec: Option<String>,
    pub bytes: u64,
    pub quality_error: Option<f64>,
    pub fallback: bool,
    pub value_credit: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KVTelemetry {
    next_sequence: u64,
    pub events: Vec<PageEvent>,
}

impl KVTelemetry {
    pub fn record(&mut self, mut event: PageEvent) {
        self.next_sequence += 1;
        event.sequence = self.next_sequence;
        self.events.push(event);
    }

    pub fn page_event(
        &mut self,
        kind: PageEventKind,
        page_id: &KVPageId,
        representation: Option<RepresentationKind>,
        generation: u64,
        bytes: u64,
    ) {
        self.record(PageEvent {
            sequence: 0,
            kind,
            page_id: Some(page_id.clone()),
            prefix_id: Some(page_id.prefix_node_id.to_string()),
            representation,
            generation,
            codec: None,
            bytes,
            quality_error: None,
            fallback: false,
            value_credit: 0.0,
        });
    }

    pub fn fallback(
        &mut self,
        page_id: &KVPageId,
        generation: u64,
        reason_codec: impl Into<String>,
    ) {
        self.record(PageEvent {
            sequence: 0,
            kind: PageEventKind::FallbackEscalated,
            page_id: Some(page_id.clone()),
            prefix_id: Some(page_id.prefix_node_id.to_string()),
            representation: None,
            generation,
            codec: Some(reason_codec.into()),
            bytes: 0,
            quality_error: None,
            fallback: true,
            value_credit: 0.0,
        });
    }

    pub fn fallback_rate(&self) -> f64 {
        let requests = self
            .events
            .iter()
            .filter(|event| {
                matches!(
                    event.kind,
                    PageEventKind::ReconstructionRequested | PageEventKind::PageSelected
                )
            })
            .count();
        let fallbacks = self
            .events
            .iter()
            .filter(|event| {
                event.fallback || matches!(event.kind, PageEventKind::FallbackEscalated)
            })
            .count();
        if requests == 0 {
            0.0
        } else {
            fallbacks as f64 / requests as f64
        }
    }

    pub fn exact_tail_retrieval_rate(&self) -> f64 {
        let tail = self
            .events
            .iter()
            .filter(|event| matches!(event.kind, PageEventKind::ExactTailRetrieved))
            .count();
        let requests = self
            .events
            .iter()
            .filter(|event| matches!(event.kind, PageEventKind::ReconstructionRequested))
            .count();
        if requests == 0 {
            0.0
        } else {
            tail as f64 / requests as f64
        }
    }

    /// OP-09 accounting: a value credit is only emitted by a realized event;
    /// this method detects duplicate event IDs/sequence values and rejects
    /// malformed imported telemetry.
    pub fn validate_value_ledger(&self) -> Result<(), &'static str> {
        for (index, event) in self.events.iter().enumerate() {
            if event.sequence != (index as u64 + 1) {
                return Err("telemetry sequence is not monotonic");
            }
            if !event.value_credit.is_finite() || event.value_credit < 0.0 {
                return Err("invalid negative/non-finite value credit");
            }
            if event.fallback && event.value_credit != 0.0 {
                return Err("fallback cannot receive an unobserved value credit");
            }
        }
        Ok(())
    }
}

#[allow(dead_code)]
fn _stable_digest_for_event(event: &PageEvent) -> StableDigest {
    StableDigest::from_text(&format!(
        "{}:{}:{}",
        event.sequence, event.generation, event.bytes
    ))
}
