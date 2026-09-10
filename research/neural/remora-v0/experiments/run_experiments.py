from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lifetime_learning import run as run_lifetime
from experiments.manual_lineage import run as run_manual
from experiments.module_replacement import run as run_replacement
from experiments.overfit import run as run_overfit
from experiments.resurrection import run as run_resurrection
from experiments.donor_inspection import run as run_donor_inspection


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the non-pretraining Remora-v0 experiment suite.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--remora-checkpoint")
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--donor-path", help="Optional resident donor directory; inspection is header-only.")
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "overfit": run_overfit(17, 60 if args.quick else 120, "cpu", out / "overfit.json"),
        "lifetime": run_lifetime(31, "cpu", out / "lifetime-evidence.json"),
        "manual": run_manual(41, out / "manual-lineage.json"),
        "resurrection": run_resurrection(out / "resurrection.json"),
    }
    if args.remora_checkpoint:
        results["replacement"] = run_replacement(
            args.remora_checkpoint,
            args.baseline_checkpoint,
            seed=53,
            steps=40 if args.quick else 120,
            device_name="auto",
            output=out / "module-replacement.json",
        )
    if args.donor_path:
        results["donor_inspection"] = run_donor_inspection(
            args.donor_path,
            out / "donor-inspection.json",
        )
    print(json.dumps({k: {"schema": v.get("schema")} for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
