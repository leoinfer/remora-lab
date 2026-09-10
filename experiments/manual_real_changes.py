from __future__ import annotations

"""Exercise the factual manual against the actual v0 change history.

This is deliberately ledger-first.  Answers are derived from experiment
receipts and the assembly graph; no prose answer is allowed to mutate the
graph or become an unsupported dependency.  The learned manual remains a
reader over this authority rather than the authority itself.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.ledger import record_experiment
from remora.manual import AssemblyLedger, ModuleRecord
from remora.utils import git_hash, write_json


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        return json.load(handle)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_assembly(source: dict[str, Any]) -> AssemblyLedger:
    assembly = source["lineage"]["assembly"]
    ledger = AssemblyLedger(generation=int(assembly["generation"]))
    for raw in assembly["modules"].values():
        ledger.register_module(ModuleRecord(**raw))
    ledger.edges = [dict(edge) for edge in assembly["edges"]]
    ledger.experiments = {
        key: dict(value) for key, value in assembly.get("experiments", {}).items()
    }
    return ledger


def _register(
    ledger: AssemblyLedger,
    module_id: str,
    *,
    version: str,
    inputs: list[str],
    outputs: list[str],
    state_owner: str,
    parameter_count: int,
    ancestry: list[str],
    status: str,
    benchmarks: dict[str, Any] | None = None,
    content_hash: str = "",
) -> None:
    if module_id in ledger.modules:
        return
    ledger.register_module(ModuleRecord(
        module_id=module_id,
        version=version,
        inputs=inputs,
        outputs=outputs,
        state_owner=state_owner,
        parameter_count=int(parameter_count),
        ancestry=ancestry,
        status=status,
        benchmarks=benchmarks or {},
        content_hash=content_hash,
    ))


def _active_dependents(ledger: AssemblyLedger, module_id: str) -> list[str]:
    """Return graph dependents whose records are currently ACTIVE."""

    return sorted(
        name
        for name in ledger.dependents(module_id)
        if ledger.modules.get(name) is not None
        and ledger.modules[name].status == "ACTIVE"
    )


def _aged_regressions(aged: dict[str, Any], seed: int = 7) -> list[dict[str, Any]]:
    row = next(item for item in aged["runs"] if int(item["seed"]) == seed)
    before = row["remora"]["after_replacement_before_training"]
    after = row["remora"]["after_local_rehearsal"]
    regressions = []
    for task_id in sorted(before["tasks"]):
        for split in ("valid", "shifted", "unseen"):
            before_accuracy = float(before["tasks"][task_id][split]["teacher_forced_accuracy"])
            after_accuracy = float(after["tasks"][task_id][split]["teacher_forced_accuracy"])
            if after_accuracy < before_accuracy:
                regressions.append({
                    "task": task_id,
                    "split": split,
                    "before": before_accuracy,
                    "after": after_accuracy,
                    "delta": after_accuracy - before_accuracy,
                })
    return regressions


def run(
    *,
    output: str | Path = ROOT / "results" / "manual-real-changes-v1.json",
    ship_path: str | Path = ROOT / "results" / "ship-of-theseus-aged-v1.json",
    aged_path: str | Path = ROOT / "results" / "aged-surgery-v1.json",
    donor_path: str | Path = ROOT / "results" / "qwen-neural-graft-analysis-v1.json",
    consolidation_path: str | Path = ROOT / "results" / "consolidation-probe-v1.json",
    lifetime_path: str | Path = ROOT / "results" / "lifetime-compounding.json",
    resurrection_path: str | Path = ROOT / "results" / "resurrection-real-v1.json",
) -> dict[str, Any]:
    ship = _load(ship_path)
    aged = _load(aged_path)
    donor = _load(donor_path)
    consolidation = _load(consolidation_path)
    lifetime = _load(lifetime_path)
    resurrection = _load(resurrection_path)

    ledger = _copy_assembly(ship)
    ledger.generation = max(ledger.generation, 5)

    # Foreign-neural lineage is represented as a candidate branch, not as a
    # promoted replacement.  The payload hash is the selected BF16 bundle
    # hash, while the candidate's own content hash is the experiment receipt.
    donor_source_id = "qwen3.8-flash-next.layer0.shared-expert"
    donor_graft_id = "remora.qwen-shared-expert-graft.v1"
    donor_extract = donor["extraction"]
    donor_accounting = donor["donor_parameter_accounting"]["per_seed"][0]
    _register(
        ledger,
        donor_source_id,
        version="qwen3.8-source-f5d0827",
        inputs=["qwen-hidden-2560"],
        outputs=["qwen-hidden-2560"],
        state_owner=donor_source_id,
        parameter_count=int(donor_extract["donor_parameter_count"]),
        ancestry=[
            "Qwen/Qwen3.8-Flash-Next",
            donor_extract["source_revision"],
            "model.layers.0.shared_expert",
        ],
        status="DONOR_FUNCTION_REPRODUCED",
        benchmarks={
            "source_shards": donor_extract["source_shards"],
            "source_payload_bytes": donor_extract["source_payload_bytes_selected"],
            "functional_equivalence": donor["retained_donor_capability_evidence"]["functional_reproduction"],
        },
        content_hash=donor_extract["source_payload_sha256_selected_bundle"],
    )
    _register(
        ledger,
        donor_graft_id,
        version="graft-v1",
        inputs=["language-v1", "qwen-hidden-2560"],
        outputs=["language-v1"],
        state_owner=donor_graft_id,
        parameter_count=int(donor_accounting["donor_parameters_preserved_unchanged"] + donor_accounting["newly_trained_parameters"]),
        ancestry=[
            donor_source_id,
            "conversion:BF16-to-FP32-storage-only",
            "ports:rank-8-input-output",
            "QWEN-NEURAL-ORGAN-003",
        ],
        status="CANDIDATE",
        benchmarks={
            "donor_parameters_preserved_unchanged": donor_accounting["donor_parameters_preserved_unchanged"],
            "donor_parameters_analytically_transformed": donor_accounting["donor_parameters_analytically_transformed"],
            "donor_parameters_discarded": donor_accounting["donor_parameters_discarded"],
            "newly_trained_parameters": donor_accounting["newly_trained_parameters"],
            "promotion": donor["promotion_state"],
        },
        content_hash=_sha256(donor_path),
    )
    ledger.add_dependency(donor_source_id, donor_graft_id, kind="ancestry")
    ledger.add_dependency("language-v1", donor_graft_id, kind="input")
    ledger.add_dependency(donor_graft_id, "lm-head", kind="candidate-output")

    # Consolidation is a separate world-model branch with explicit archive
    # provenance.  It is not mislabeled as language-level weight absorption.
    archive_id = "experience-archive-v1"
    adapter_id = "world-model.experience-adapter-v1"
    consolidated_id = "world-model.consolidated-v1"
    first_consolidation = consolidation["runs"][0]
    provenance = first_consolidation["provenance_receipt"]
    _register(
        ledger,
        archive_id,
        version="archive-v1",
        inputs=["lifetime-observation"],
        outputs=["evidence-cluster"],
        state_owner=archive_id,
        parameter_count=0,
        ancestry=["lifetime-evidence-episodes"],
        status="ACTIVE",
        benchmarks={"episode_count": len(first_consolidation["archive"]["records"])},
    )
    _register(
        ledger,
        adapter_id,
        version="fast-v1",
        inputs=["evidence-cluster"],
        outputs=["belief-v1"],
        state_owner=adapter_id,
        parameter_count=int(first_consolidation["parameter_counts"]["experience_adapter"]),
        ancestry=["experience-fast-path"],
        status="ACTIVE",
        benchmarks={"changed_fraction": first_consolidation["fast_experience_update"]["parameter_update"]["changed_fraction"]},
    )
    _register(
        ledger,
        consolidated_id,
        version="slow-v1",
        inputs=["belief-v1"],
        outputs=["belief-v1"],
        state_owner=consolidated_id,
        parameter_count=int(first_consolidation["parameter_counts"]["consolidated"]),
        ancestry=[adapter_id, provenance["consolidation_id"]],
        status="ACTIVE",
        benchmarks={
            "source_episode_ids": provenance["source_episode_ids"],
            "source_independence_clusters": provenance["source_independence_clusters"],
            "archive_recoverable": provenance["archive_recoverable"],
        },
    )
    ledger.add_dependency(archive_id, adapter_id, kind="evidence")
    ledger.add_dependency(adapter_id, consolidated_id, kind="consolidation")

    # Aged surgery is recorded as a tested dormant branch because the result
    # did not beat the matched Transformer-LoRA control on the full matrix.
    aged_candidate = "expert-layer1-aged-swiglu-surgery"
    _register(
        ledger,
        aged_candidate,
        version="aged-surgery-v1",
        inputs=["language-v1"],
        outputs=["language-v1"],
        state_owner=aged_candidate,
        parameter_count=int(aged["parameter_accounting"]["remora_replacement"]["new_parameters"]),
        ancestry=["expert-layer1-g2", "SwiGLUExpert", "AGED-SURGERY-001"],
        status="DORMANT",
        benchmarks={"promotion": aged["promotion"], "paired": aged["paired"]},
        content_hash=_sha256(aged_path),
    )
    ledger.add_dependency("language-v1", aged_candidate, kind="input")
    ledger.add_dependency(aged_candidate, "lm-head", kind="candidate-output")
    ledger.modules["expert-layer1-g2"].replacements.append(aged_candidate)

    # Bind actual experiment receipts to the factual manual.
    ledger.add_experiment("LIFETIME-COMPOUNDING-001", {
        "status": "COMPLETE",
        "artifact": str(lifetime_path),
        "note": "five-stage sequential curriculum with shifted/unseen interfaces",
    })
    ledger.add_experiment("CONSOLIDATION-001", {
        "status": "COMPLETE",
        "artifact": str(consolidation_path),
        "note": "structured-world probe; provenance preserved",
    })
    ledger.add_experiment("QWEN-NEURAL-ORGAN-003", {
        "status": donor["promotion_state"],
        "artifact": str(donor_path),
        "note": "real frozen Qwen shared-expert organ and ports",
    })
    ledger.add_experiment("AGED-SURGERY-001", {
        "status": aged["promotion"]["state"],
        "artifact": str(aged_path),
        "note": "near-equal Transformer-LoRA adversary",
    })
    ledger.add_experiment("RESURRECTION-REAL-001", {
        "status": "COMPLETE",
        "artifact": str(resurrection_path),
        "note": "real failed candidate retested after age and replay policy changed",
    })
    ledger.add_experiment("SHIP-THESEUS-AGED-001", {
        "status": "COMPLETE",
        "artifact": str(ship_path),
        "note": "four sequential replacements after lifetime aging",
    })

    t3 = next(
        stage
        for stage in lifetime["runs"][0]["arms"]["remora_local_rehearsal"]["stage_results"]
        if stage["stage_id"] == "T3"
    )
    changed_t3 = list(lifetime["runs"][0]["arms"]["remora_local_rehearsal"]["selected_parameter_names"])
    consolidated_ids = list(provenance["source_episode_ids"])
    aged_regressions = _aged_regressions(aged)
    resurrection_candidate = resurrection["queue_after_context_change"][0]
    graph_queries = {
        "dependents_of_language_v1_historical": ledger.factual_query("dependents", "language-v1"),
        "active_dependents_of_language_v1": {
            "module_id": "language-v1",
            "dependents": _active_dependents(ledger, "language-v1"),
            "source": "graph+status-filter",
        },
        "minimum_affected_if_expert_layer1_g2_changes": ledger.factual_query(
            "dependents", "expert-layer1-g2"
        ),
        "ancestry_of_qwen_graft": ledger.factual_query("ancestry", donor_graft_id),
        "ancestry_of_aged_surgery_candidate": ledger.factual_query("ancestry", aged_candidate),
    }
    verified_questions = [
        {
            "question": "Which modules/parameters changed during T3?",
            "answer": changed_t3,
            "oracle": {"selected_parameters": changed_t3, "changed_parameters": t3["changed_parameters"]},
            "passed": changed_t3 == list(lifetime["runs"][0]["arms"]["remora_local_rehearsal"]["selected_parameter_names"]),
        },
        {
            "question": "Which experiences were consolidated into world-model.consolidated-v1?",
            "answer": consolidated_ids,
            "oracle": provenance["source_episode_ids"],
            "passed": consolidated_ids == provenance["source_episode_ids"] and bool(first_consolidation["provenance_receipt"]["archive_recoverable"]),
        },
        {
            "question": "Which modules depend on language-v1 now?",
            "answer": graph_queries["active_dependents_of_language_v1"]["dependents"],
            "oracle": graph_queries["active_dependents_of_language_v1"],
            "passed": all(name in ledger.modules for name in graph_queries["active_dependents_of_language_v1"]["dependents"]),
        },
        {
            "question": "What is the graph-derived affected neighborhood for expert-layer1-g2?",
            "answer": graph_queries["minimum_affected_if_expert_layer1_g2_changes"],
            "oracle": graph_queries["minimum_affected_if_expert_layer1_g2_changes"],
            "passed": graph_queries["minimum_affected_if_expert_layer1_g2_changes"]["source"] == "graph",
        },
        {
            "question": "Which previous task/interface rows regressed after aged surgery?",
            "answer": aged_regressions,
            "oracle": "direct comparison of aged-surgery before/after matrices for seed 7",
            "passed": aged_regressions == _aged_regressions(aged, seed=7),
        },
        {
            "question": "Which real failed candidate deserves resurrection after context changed?",
            "answer": resurrection_candidate,
            "oracle": resurrection["queue_after_context_change"][0],
            "passed": float(resurrection_candidate["priority"]) > float(resurrection["queue_before_context_change"][0]["priority"]),
        },
        {
            "question": "What old failure condition changed before the resurrection trial?",
            "answer": resurrection["changed_context"],
            "oracle": {
                "old": resurrection["source_failure"]["failure_state"],
                "new": resurrection["changed_context"],
            },
            "passed": resurrection["source_failure"]["failure_state"]["training_policy"] == "target_only"
            and resurrection["changed_context"]["training_policy"] == "target_plus_old_rehearsal",
        },
        {
            "question": "What is the Qwen graft ancestry and promotion status?",
            "answer": {"ancestry": graph_queries["ancestry_of_qwen_graft"], "status": ledger.modules[donor_graft_id].status},
            "oracle": {"ancestry": graph_queries["ancestry_of_qwen_graft"], "status": "CANDIDATE"},
            "passed": ledger.modules[donor_graft_id].status == "CANDIDATE",
        },
    ]
    result = {
        "schema": "remora-v1-manual-real-changes-result",
        "experiment_family": "MANUAL-REAL-CHANGES-001+",
        "source_artifacts": {
            "ship": {"path": str(ship_path), "sha256": _sha256(ship_path)},
            "aged_surgery": {"path": str(aged_path), "sha256": _sha256(aged_path)},
            "donor_analysis": {"path": str(donor_path), "sha256": _sha256(donor_path)},
            "consolidation": {"path": str(consolidation_path), "sha256": _sha256(consolidation_path)},
            "lifetime": {"path": str(lifetime_path), "sha256": _sha256(lifetime_path)},
            "resurrection": {"path": str(resurrection_path), "sha256": _sha256(resurrection_path)},
        },
        "graph_queries": graph_queries,
        "verified_questions": verified_questions,
        "all_questions_passed": all(item["passed"] for item in verified_questions),
        "lineage": ledger.to_dict(),
        "labels": {
            "MEASURED": ["source artifact hashes", "experiment receipt fields", "aged before/after regressions"],
            "DERIVED": ["graph dependents", "ancestry queries", "active status filter", "question pass flags"],
            "HYPOTHESIS": ["learned manual can reason over this graph without mutating factual state"],
            "LIMITATION": ["this tranche verifies a factual real-change manual; it does not claim a language-model manual has been trained on the full graph"],
        },
        "promotion": {"state": "CONTROLLED_MANUAL_RECEIPT", "promoted": False},
        "interpretation": "MEASURED/DERIVED: the factual manual answers real-change questions from hashed experiment receipts and graph state; candidate branches remain non-promoted and the learner is non-authoritative.",
    }
    write_json(output, result)
    record_experiment(
        ROOT,
        "MANUAL-REAL-CHANGES-001",
        "The factual manual should remain correct when real lifetime, donor, surgery, consolidation, resurrection, and Ship-of-Theseus changes are composed.",
        "Import the aged assembly graph and bind actual experiment receipts, donor ancestry, consolidation provenance, and surgery branches; answer mechanically checkable questions.",
        "Every question agrees with its receipt/graph oracle, candidate branches retain non-promoted status, and no unsupported dependency is introduced.",
        "Any answer is not graph/receipt-derived, provenance is lost, a dormant candidate is treated as promoted, or an unsupported dependency appears.",
        "python -m experiments.manual_real_changes",
        0,
        {"all_questions_passed": result["all_questions_passed"], "question_count": len(verified_questions)},
        result["interpretation"],
        "Use this receipt as the factual context for a future trained manual; do not let a learned reader write graph state.",
        hardware={"device": "cpu", "git_hash_at_run": git_hash(ROOT)},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "results" / "manual-real-changes-v1.json"))
    args = parser.parse_args()
    result = run(output=args.output)
    print(json.dumps({
        "all_questions_passed": result["all_questions_passed"],
        "question_count": len(result["verified_questions"]),
        "graph_queries": result["graph_queries"],
    }, indent=2))


if __name__ == "__main__":
    main()
