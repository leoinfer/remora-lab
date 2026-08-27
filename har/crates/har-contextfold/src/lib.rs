//! Rust ContextFold orchestration.
//!
//! Numeric page encode/decode remains an external codec implementation (the
//! existing offline capture artifacts are evidence, not reinterpreted as
//! quality certificates). This crate owns immutable policy compilation,
//! logical graph selection, residency layer storage handoff, leases, fallback, and
//! telemetry.

mod orchestrator;
mod policy;
mod store;

pub use orchestrator::{
    ContextFoldController, ContextFoldError, ContextFoldManifest, ContextFoldSelection,
};
pub use policy::{compile_policy, CompiledContextPolicy, PolicyError};
pub use store::{FilePageStore, KVPageStore, MemoryPageStore, StoreError};

pub use har_kv;

#[cfg(test)]
mod tests {
    use super::*;
    use har_kv::{
        KVPageId, KVPageRecord, KVQualityCertificate, MetricBounds, PrefixIdentity,
        RepresentationKind, StableDigest,
    };

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
    fn policy_compiles_to_immutable_data() {
        let source = "context { hot exact_q8 window 4096; cold contextfold codec pca_residual_v0; fallback token_archive; require tail_probe_pass; }";
        let policy = compile_policy(source).expect("policy");
        assert_eq!(policy.hot_window_tokens, 4096);
        assert!(!policy.admitted());
        assert!(policy.with_tail_probe_result(true).admitted());
        assert!(compile_policy("context { hot exact_q8 window 0; }").is_err());
    }

    #[test]
    fn controller_separates_logical_page_from_store_location() {
        let source = "context { hot exact_q8 window 4096; cold contextfold codec pca_residual_v0; fallback token_archive; }";
        let policy = compile_policy(source).unwrap();
        let authority = StableDigest::from_text("authority");
        let mut controller = ContextFoldController::new(
            MemoryPageStore::new("test-store"),
            policy,
            "archive:s",
            authority.clone(),
        )
        .unwrap();
        let root = controller.graph.insert_root(identity(vec![])).unwrap();
        let child_identity = identity(vec![11]);
        let (child, _) = controller
            .graph
            .extend(&root.id, 11, child_identity.clone())
            .unwrap();
        let closure = child_identity.closure(controller.graph.generation(), 0);
        let page_id = KVPageId {
            prefix_node_id: child.clone(),
            ordinal: 0,
            token_start: 0,
            token_end: 1,
            layer_start: 0,
            layer_end: 16,
            head_start: 0,
            head_end: 4,
            logical_generation: 1,
        };
        controller
            .register_reference_page(KVPageRecord::new(page_id.clone(), closure.clone()), 1088, 4)
            .unwrap();
        controller
            .graph
            .add_representation(
                &page_id,
                har_kv::KVRepresentation::ExactTokenArchive {
                    bytes: 4,
                    archive_ref: "archive:s".to_string(),
                    location: None,
                },
                controller.graph.generation(),
            )
            .unwrap();
        let quality = KVQualityCertificate::measured_only(
            "pca_residual_v0",
            closure.root(),
            closure.root(),
            MetricBounds::none(),
        );
        let location = controller
            .persist_latent_page(
                &page_id,
                "pca_residual_v0",
                vec![1, 2, 3, 4],
                quality,
                har_kv::KVFormat::ContextFoldLatent,
            )
            .unwrap();
        assert_eq!(location.store_id, "test-store");
        controller.mark_page_cold(&page_id).unwrap();
        let selection = controller.select_for_attention(&page_id, None).unwrap();
        assert!(matches!(
            selection,
            ContextFoldSelection::Representation {
                kind: RepresentationKind::ContextFoldLatent,
                ..
            }
        ));
        assert_eq!(
            controller
                .load_representation_payload(&page_id, RepresentationKind::ContextFoldLatent)
                .unwrap(),
            vec![1, 2, 3, 4]
        );
        assert!(!page_id.key().contains("test-store"));
    }
}
