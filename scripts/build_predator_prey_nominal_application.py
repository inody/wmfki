"""Build the nominal predator--prey application artifact from a frozen sweep.

The fitted coefficients are copied from the predeclared observation-noise
condition of the paired robustness sweep. Only the representative display
rollout is newly simulated, so the recovery figure and comparison table use
exactly the same fits as the sweep rather than a separately simulated training
set.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_predator_prey_application import (
    create_display,
    summarize,
    validate_protocol as validate_application_protocol,
    write_records_csv,
)
from scripts.run_predator_prey_sweeps import validate_protocol as validate_sweep_protocol

DEFAULT_CONFIG = ROOT / "configs/experiments/predator_prey_nominal_from_sweep_common_noise_001.json"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_file(relative, expected):
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"artifact hash changed: {relative}: {actual} != {expected}")
    return path


def validate_protocol(requested, application, sweep):
    if requested.get("task") != "predator_prey_nominal_application_from_sweep" \
            or requested.get("protocol_version") != 1:
        raise ValueError("invalid nominal predator--prey artifact protocol")
    validate_application_protocol(application)
    validate_sweep_protocol(sweep)
    condition = requested.get("condition", {})
    sigma_obs = float(condition.get("observation_noise", -1.0))
    if condition != {"sweep": "observation_noise", "observation_noise": 0.01}:
        raise ValueError("nominal condition must be the predeclared sigma_obs=0.01 sweep point")
    if float(application["observation"]["observation_noise"]) != sigma_obs:
        raise ValueError("application and extracted sweep noise differ")
    if float(sweep["observation"]["observation_noise"]) != sigma_obs:
        raise ValueError("sweep baseline and extracted nominal noise differ")
    if int(requested.get("repetitions", -1)) != 20:
        raise ValueError("nominal artifact must retain all 20 repetitions")
    if int(requested.get("representative_repetition", -1)) != 0:
        raise ValueError("representative repetition must remain the first predeclared repetition")


def _method_equal(left, right):
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def extract_records(source, application, requested):
    sigma_obs = float(requested["condition"]["observation_noise"])
    selected = [
        row for row in source
        if row["condition"]["sweep"] == "observation_noise"
        and float(row["condition"]["value"]) == sigma_obs
    ]
    selected.sort(key=lambda row: int(row["repetition"]))
    if len(selected) != int(requested["repetitions"]):
        raise ValueError("nominal sweep extraction is incomplete")

    # The same baseline appears in all three one-factor panels. Check that the
    # cached fit is identical before selecting the observation-noise label.
    parity = []
    for chosen in selected:
        duplicates = [
            row for row in source
            if int(row["repetition"]) == int(chosen["repetition"])
            and int(row["condition"]["prey_particles"]) == 100
            and float(row["condition"]["observation_dt"]) == 0.01
            and float(row["condition"]["observation_noise"]) == sigma_obs
        ]
        passed = len(duplicates) == 3 and all(
            _method_equal(chosen["methods"], row["methods"]) for row in duplicates
        )
        parity.append({
            "repetition": int(chosen["repetition"]),
            "duplicate_count": len(duplicates),
            "passed": passed,
        })
    if not all(item["passed"] for item in parity):
        raise ValueError("the three labelled sweep baselines are not identical")

    dictionary = application["dictionary"]
    prey_truth = np.concatenate([
        np.asarray(dictionary["prey_prey_true"], dtype=float),
        np.asarray(dictionary["prey_predator_true"], dtype=float),
    ])
    predator_truth = np.asarray(dictionary["predator_prey_true"], dtype=float)
    converted = []
    for row in selected:
        methods = row["methods"]
        lasso = methods["lasso"]
        ols = methods["ols"]
        lasso_xx = np.asarray(lasso["channels"]["prey_prey"]["coefficients"], dtype=float)
        lasso_xz = np.asarray(lasso["channels"]["prey_predator"]["coefficients"], dtype=float)
        lasso_zx = np.asarray(lasso["channels"]["predator_prey"]["coefficients"], dtype=float)
        ols_xx = np.asarray(ols["channels"]["prey_prey"]["coefficients"], dtype=float)
        ols_xz = np.asarray(ols["channels"]["prey_predator"]["coefficients"], dtype=float)
        ols_zx = np.asarray(ols["channels"]["predator_prey"]["coefficients"], dtype=float)
        lasso_drift = lasso["held_out_drift"]
        ols_drift = ols["held_out_drift"]
        lasso_selection = lasso["selection"]
        converted.append({
            "index": int(row["repetition"]),
            "training_seed": int(row["seeds"]["training"]),
            "validation_seed": int(row["seeds"]["validation"]),
            "test_seed": int(row["seeds"]["test"]),
            "theta_xx": lasso_xx.tolist(),
            "theta_xz": lasso_xz.tolist(),
            "theta_zx": lasso_zx.tolist(),
            "ols_theta_xx": ols_xx.tolist(),
            "ols_theta_xz": ols_xz.tolist(),
            "ols_theta_zx": ols_zx.tolist(),
            "lasso_prey_coefficient_error": float(np.linalg.norm(
                np.concatenate([lasso_xx, lasso_xz]) - prey_truth
            ) / np.linalg.norm(prey_truth)),
            "lasso_predator_coefficient_error": float(
                np.linalg.norm(lasso_zx - predator_truth) / np.linalg.norm(predator_truth)
            ),
            "ols_prey_coefficient_error": float(np.linalg.norm(
                np.concatenate([ols_xx, ols_xz]) - prey_truth
            ) / np.linalg.norm(prey_truth)),
            "ols_predator_coefficient_error": float(
                np.linalg.norm(ols_zx - predator_truth) / np.linalg.norm(predator_truth)
            ),
            "lasso_drift_error": float(lasso_drift["species_balanced_relative"]),
            "lasso_prey_drift_error": float(lasso_drift["prey_relative"]),
            "lasso_predator_drift_error": float(lasso_drift["predator_relative"]),
            "ols_drift_error": float(ols_drift["species_balanced_relative"]),
            "ols_prey_drift_error": float(ols_drift["prey_relative"]),
            "ols_predator_drift_error": float(ols_drift["predator_relative"]),
            "lasso_profile_errors": [float(lasso["channels"][name]["profile_error_relative"])
                                     for name in ("prey_prey", "prey_predator", "predator_prey")],
            "ols_profile_errors": [float(ols["channels"][name]["profile_error_relative"])
                                   for name in ("prey_prey", "prey_predator", "predator_prey")],
            "prey_selected_alpha": float(lasso_selection["prey"]["alpha"]),
            "predator_selected_alpha": float(lasso_selection["predator"]["alpha"]),
            "prey_selected_at_boundary": bool(lasso_selection["prey"]["selected_at_grid_boundary"]),
            "predator_selected_at_boundary": bool(lasso_selection["predator"]["selected_at_grid_boundary"]),
            "prey_nonconverged_candidates": int(lasso_selection["prey"]["nonconverged_candidates"]),
            "predator_nonconverged_candidates": int(lasso_selection["predator"]["nonconverged_candidates"]),
            "prey_design_rank": int(row["systems"]["prey"]["training_rank"]),
            "prey_design_columns": int(row["systems"]["prey"]["columns"]),
            "predator_design_rank": int(row["systems"]["predator"]["training_rank"]),
            "predator_design_columns": int(row["systems"]["predator"]["columns"]),
        })
    return converted, parity


def build(config, output=None):
    config = Path(config).resolve()
    requested = read_json(config)
    application_path = _verify_file(
        requested["application_config"], requested["application_config_sha256"]
    )
    sweep_protocol_path = _verify_file(
        requested["sweep_protocol"], requested["sweep_protocol_sha256"]
    )
    sweep_records_path = _verify_file(
        requested["sweep_records"], requested["sweep_records_sha256"]
    )
    application = read_json(application_path)
    sweep = read_json(sweep_protocol_path)
    validate_protocol(requested, application, sweep)
    records, parity = extract_records(read_json(sweep_records_path), application, requested)

    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = ROOT / "outputs/validation/predator_prey" / f"application-from-sweep-{stamp}"
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    effective = json.loads(json.dumps(application))
    effective["fit_source"] = {
        "policy": "exact_extraction_from_paired_robustness_sweep",
        "condition": requested["condition"],
        "sweep_records": requested["sweep_records"],
        "sweep_records_sha256": requested["sweep_records_sha256"],
    }
    write_json(output / "requested_protocol.json", requested)
    write_json(output / "effective_protocol.json", effective)
    write_json(output / "records.json", records)
    write_records_csv(output / "records.csv", records)
    write_json(output / "summary.json", summarize(records))
    write_json(output / "baseline_parity.json", {
        "count": len(parity),
        "passed_count": sum(item["passed"] for item in parity),
        "passed": all(item["passed"] for item in parity),
        "comparisons": parity,
    })
    display = create_display(effective, records[int(requested["representative_repetition"])], output)
    write_json(output / "display_summary.json", display)

    source_files = [
        "scripts/build_predator_prey_nominal_application.py",
        "scripts/run_predator_prey_application.py",
        "scripts/run_predator_prey_sweeps.py",
        "predator_prey.py",
        requested["application_config"],
        requested["sweep_protocol"],
    ]
    source_hashes = {relative: digest(ROOT / relative) for relative in source_files}
    write_json(output / "source_manifest.json", {"sha256": source_hashes})
    artifacts = [
        "requested_protocol.json", "effective_protocol.json", "records.json",
        "records.csv", "summary.json", "baseline_parity.json",
        "display_trajectory.npz", "display_summary.json", "source_manifest.json",
    ]
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "observation_noise": requested["condition"]["observation_noise"],
        "representative_repetition": requested["representative_repetition"],
        "sweep_records_sha256": requested["sweep_records_sha256"],
        "baseline_parity_passed": all(item["passed"] for item in parity),
        "display": display,
        "artifact_sha256": {name: digest(output / name) for name in artifacts},
    }
    write_json(output / "manifest.json", manifest)
    shutil.copy2(config, output / "input_config.json")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(build(args.config, args.output))


if __name__ == "__main__":
    main()
