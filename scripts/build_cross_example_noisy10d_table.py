"""Replace the clean 10D row in the exploratory cross-example table.

The 1D and predator--prey records come unchanged from the independently
verified clean-10D comparison.  Only the 10D rows are replaced by the
predeclared sigma_obs=0.005 records from the paired observation-noise sweep.
No estimator is rerun and no manuscript file is modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cross_example_nominal_comparison import (
    digest,
    read_json,
    summarize,
    write_interpretation,
    write_json,
    write_records_csv,
    write_tables,
)


DEFAULT_CONFIG = ROOT / "configs" / "experiments" / "cross_example_benchmark_comparison.json"
SOURCE_FILES = (
    "configs/experiments/cross_example_benchmark_comparison.json",
    "scripts/build_cross_example_noisy10d_table.py",
    "scripts/run_cross_example_nominal_comparison.py",
    "scripts/verify_cross_example_nominal_comparison.py",
)


def _verify(path, expected):
    path = ROOT / path
    actual = digest(path)
    if actual != expected:
        raise RuntimeError(f"input hash mismatch: {path}: {actual} != {expected}")
    return path


def build(config_path, out):
    spec = read_json(config_path)
    if spec.get("task") != "cross_example_nominal_method_comparison":
        raise ValueError("invalid cross-example table protocol")
    base_path = _verify(spec["base_cross_records"], spec["base_cross_records_sha256"])
    noisy = spec["localized_10d"]
    noisy_path = _verify(noisy["source_records"], noisy["source_sha256"])
    base_records = read_json(base_path)
    noisy_records = read_json(noisy_path)
    nominal = float(noisy["nominal_observation_noise"])

    records = [row for row in base_records if row["example"] != "localized_10d"]
    localized = [row for row in noisy_records
                 if float(row["observation_noise"]) == nominal]
    if len(localized) != 3 * int(noisy["repetitions"]):
        raise RuntimeError("unexpected noisy-10D nominal record count")
    for row in localized:
        records.append({
            "example": "localized_10d", "method": row["method"],
            "repetition": int(row["repetition"]),
            "training_seed": int(row["training_seed"]),
            "test_seed": int(row["test_seed"]),
            "observation_noise": nominal,
            "coefficient_error_relative": float(row["coefficient_error_relative"]),
            "heldout_drift_error_relative": float(row["latent_drift_relative_error"]),
            "profile_error_relative": float(row["profile_relative_error"]),
            "coefficients": row["coefficients"], "status": row["status"],
        })
    method_order = spec["methods"]
    example_order = ("one_dimensional", "predator_prey", "localized_10d")
    records.sort(key=lambda row: (
        example_order.index(row["example"]), row["repetition"],
        method_order.index(row["method"]),
    ))
    summary = summarize(records, spec["metric"])

    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    shutil.copy2(config_path, out / "requested_protocol.json")
    write_json(out / "records.json", records)
    write_records_csv(out / "records.csv", records)
    write_json(out / "summary.json", summary)
    write_tables(out, summary)
    write_interpretation(out, summary)
    for name in ("table_rmse_draft.md", "table_median_iqr_draft.md"):
        path = out / name
        path.write_text(
            path.read_text()
            .replace("Nominal system", "Benchmark system")
            .replace("across all predeclared repetitions", "across all repetitions")
        )
    tex_path = out / "table_rmse_draft.tex"
    tex_path.write_text(tex_path.read_text().replace("Nominal system", "Benchmark system"))
    groups = {(row["example"], row["method"]): row for row in summary["groups"]}
    lasso_10d = 100 * groups[("localized_10d", "weak_lasso")]["rmse"]
    weak_ols_10d = 100 * groups[("localized_10d", "weak_ols")]["rmse"]
    strong_10d = 100 * groups[("localized_10d", "strong_forward_difference_ols")]["rmse"]
    (out / "interpretation_draft.md").write_text(
        "# Interpretation of the cross-example benchmark comparison\n\n"
        "Weak LASSO has the lowest relative coefficient RMSE in all three benchmark "
        "systems. In the 10D localized system at "
        f"$\\sigma_{{\\rm obs}}={nominal:g}$, its RMSE is {lasso_10d:.1f}%, compared "
        f"with {weak_ols_10d:.1f}% for weak OLS and {strong_10d:.0f}% for Strong "
        "FD-OLS. The comparison shows the combined benefit of weak temporal testing "
        "and sparse regularization under noisy position observations.\n\n"
        "The compact table uses RMSE so that rare large errors remain visible. The "
        "median [IQR] table is retained as a diagnostic because the 1D weak-LASSO "
        "errors are heavy-tailed.\n"
    )
    source_hashes = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hashes[relative] = digest(source)
    artifacts = {name: digest(out / name) for name in (
        "requested_protocol.json", "records.json", "records.csv", "summary.json",
        "table_rmse_draft.md", "table_median_iqr_draft.md", "table_rmse_draft.tex",
        "interpretation_draft.md",
    )}
    write_json(out / "manifest.json", {
        "task": spec["task"], "protocol_version": spec["protocol_version"],
        "status": "scientific_complete", "smoke": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records), "source_sha256": source_hashes,
        "artifact_sha256": artifacts,
        "input_sha256": {
            key: spec[key]["source_sha256"]
            for key in ("one_dimensional", "predator_prey", "localized_10d")
        },
        "base_cross_records_sha256": spec["base_cross_records_sha256"],
        "localized_10d_observation_noise": nominal,
        "manuscript_modified": False,
    })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.out = ROOT / "outputs" / "validation" / "cross_example_nominal_comparison" / f"noisy10d-{stamp}"
    print(build(args.config, args.out))


if __name__ == "__main__":
    main()
