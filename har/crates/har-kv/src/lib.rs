//! HAR's Rust KV/prefix semantics.
//!
//! This crate owns logical identity, page representation choice, leases,
//! dependency closure, certificates, fallback, and telemetry.  It does not
//! execute attention and contains no foreign execution dependency.  Native-kernel registry
//! may consume the stable request/representation descriptors from this crate.

mod certificate;
mod codec;
mod geometry;
mod graph;
mod identity;
mod page;
mod telemetry;

pub use certificate::{CertificateError, KVQualityCertificate, MetricBounds, QualityScope};
pub use codec::{
    CodecError, CodecPayload, KVCodec, LosslessXorRleCodec, MeasuredCodec, MeasuredCodecCatalog,
};
pub use geometry::{ByteProjection, GeometryError, KVGeometry, KVNativeFormat};
pub use graph::{
    CausalClosureError, PageSelection, PrefixEdge, PrefixGraph, PrefixGraphError, PrefixNode,
    PrefixNodeId, PrefixRoot,
};
pub use identity::{KVDependencyClosure, PrefixIdentity, StableDigest};
pub use page::{
    FallbackStep, KVFallbackPlan, KVFormat, KVPageId, KVPageLease, KVPageRecord, KVRepresentation,
    KVStorageTier, PhysicalPageLocation, ReconstructionTicket, RepresentationKind,
};
pub use telemetry::{KVTelemetry, PageEvent, PageEventKind};

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(tokens: Vec<u32>) -> PrefixIdentity {
        PrefixIdentity {
            model_root: StableDigest::from_text("model"),
            tokenizer_root: StableDigest::from_text("tokenizer"),
            token_sequence: tokens,
            rope_config_root: StableDigest::from_text("rope"),
            layer_start: 0,
            layer_end: 16,
            head_start: 0,
            head_end: 4,
            kv_type: "q8_0".to_string(),
            codec_version: "contextfold-v0".to_string(),
            runtime_generation: 1,
            authority_state_root: StableDigest::from_text("authority"),
        }
    }

    #[test]
    fn qwen_geometry_is_exact() {
        let geometry = KVGeometry::qwen36_27b("model", "metadata");
        assert_eq!(geometry.q8_row_bytes(true), 1088);
        assert_eq!(geometry.q8_row_bytes(false), 1088);
        assert_eq!(
            geometry
                .project(262_144, KVNativeFormat::Q8_0, KVNativeFormat::Q8_0)
                .total_bytes,
            9_126_805_504
        );
        assert_eq!(geometry.q8_scales_per_row(true), 64);
    }

    #[test]
    fn prefix_identity_changes_when_any_causal_leaf_changes() {
        let base = identity(vec![1, 2]);
        let mut changed_model = base.clone();
        changed_model.model_root = StableDigest::from_text("other-model");
        let mut changed_codec = base.clone();
        changed_codec.codec_version = "other-codec".to_string();
        assert_ne!(
            PrefixNodeId::from_identity(&base, 1),
            PrefixNodeId::from_identity(&changed_model, 1)
        );
        assert_ne!(
            PrefixNodeId::from_identity(&base, 1),
            PrefixNodeId::from_identity(&changed_codec, 1)
        );
    }

    #[test]
    fn radix_lookup_shares_logical_prefix_and_page_has_no_physical_location() {
        let mut graph = PrefixGraph::new();
        let root = graph.insert_root(identity(vec![])).unwrap();
        let (one, edge) = graph.extend(&root.id, 42, identity(vec![42])).unwrap();
        assert_eq!(edge.parent, root.id);
        assert_eq!(graph.lookup(&root.id, &[42]).unwrap(), Some(one.clone()));
        assert_eq!(graph.lookup(&root.id, &[42, 43]).unwrap(), None);
        let closure = identity(vec![42]).closure(graph.generation(), 0);
        let page_id = KVPageId {
            prefix_node_id: one,
            ordinal: 0,
            token_start: 0,
            token_end: 1,
            layer_start: 0,
            layer_end: 16,
            head_start: 0,
            head_end: 4,
            logical_generation: 1,
        };
        let mut page = KVPageRecord::new(page_id.clone(), closure);
        page.add_representation(KVRepresentation::ExactHotQ8 {
            bytes: 69_632,
            location: None,
        });
        graph.attach_page(page).unwrap();
        assert!(
            graph
                .page(&page_id)
                .unwrap()
                .find(RepresentationKind::ExactHotQ8)
                .unwrap()
                .bytes()
                > 0
        );
        assert!(graph.validate_causal_closure(&page_id).is_ok());
    }

    #[test]
    fn lease_blocks_eviction_and_stale_generation_rejects_mutation() {
        let mut graph = PrefixGraph::new();
        let root = graph.insert_root(identity(vec![])).unwrap();
        let (child, _) = graph.extend(&root.id, 7, identity(vec![7])).unwrap();
        let closure = identity(vec![7]).closure(graph.generation(), 0);
        let page_id = KVPageId {
            prefix_node_id: child,
            ordinal: 0,
            token_start: 0,
            token_end: 1,
            layer_start: 0,
            layer_end: 16,
            head_start: 0,
            head_end: 4,
            logical_generation: 1,
        };
        let mut page = KVPageRecord::new(page_id.clone(), closure);
        page.add_representation(KVRepresentation::ExactHotQ8 {
            bytes: 100,
            location: None,
        });
        graph.attach_page(page).unwrap();
        let lease = graph
            .lease(
                &page_id,
                RepresentationKind::ExactHotQ8,
                "attention",
                graph.generation(),
            )
            .unwrap();
        assert_eq!(
            graph.evict(&page_id, RepresentationKind::ExactHotQ8, graph.generation()),
            Err(PrefixGraphError::ActiveLease)
        );
        graph.release_lease(&lease.lease_id).unwrap();
        let old_generation = graph.generation();
        graph.advance_generation();
        assert!(matches!(
            graph.evict(&page_id, RepresentationKind::ExactHotQ8, old_generation),
            Err(PrefixGraphError::StaleGeneration { .. })
        ));
    }

    #[test]
    fn lossless_codec_roundtrips_page_bytes() {
        let codec = LosslessXorRleCodec;
        let source = b"aaaaabbbbb\0\0\0contextfold";
        let payload = codec.encode(source).unwrap();
        assert_eq!(codec.decode(&payload).unwrap(), source);
    }

    #[test]
    fn measured_mse_without_state_and_attention_does_not_authorize() {
        let dependency = StableDigest::from_text("closure");
        let certificate = KVQualityCertificate::measured_only(
            "int4",
            dependency.clone(),
            dependency.clone(),
            MetricBounds {
                reconstruction_mse: Some(0.0),
                attention_score_mse: None,
                attention_distribution_jsd: None,
                attention_output_mse: None,
                exact_tail_retrieval: None,
            },
        );
        assert!(!certificate.authorize(&MetricBounds::none(), &dependency, &dependency));
    }
}
