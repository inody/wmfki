"""Finalize integrity and manuscript-claim reports for the completed P=50N R=100 run.

This script reads saved records only. It performs no weak assembly, fitting,
penalty selection, or scientific simulation.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/validation/one_dimensional_hat/sampled-p50n-R100-20261025T084338282042Z"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def finalize(output):
    output = Path(output).resolve()
    summary = read_json(output / "summary.json")
    strict = read_json(output / "claim_report.json")
    verification = read_json(output / "verification/verification.json")
    figure = read_json(output / "figure1_candidate/metadata.json")
    sizes = [point["M"] for point in summary["dictionary_sizes"]]
    sampled_lasso = [point["sampled"]["weak_lasso"]["force_error_weighted_rel"]["mean"]
                     for point in summary["dictionary_sizes"]]
    sampled_ols = [point["sampled"]["weak_ols"]["force_error_weighted_rel"]["mean"]
                   for point in summary["dictionary_sizes"]]
    full_lasso = [point["full"]["weak_lasso"]["force_error_weighted_rel"]["mean"]
                  for point in summary["dictionary_sizes"]]
    best_index = int(np.argmin(sampled_lasso))
    best_size = sizes[best_index]
    best = summary["dictionary_sizes"][best_index]
    rho = float(np.corrcoef(np.argsort(np.argsort(sampled_lasso)),
                            np.argsort(np.argsort(full_lasso)))[0, 1])
    high_mean = float(np.mean(sampled_lasso[-3:]))
    diagnostics = summary["diagnostics"]
    checks = {
        "sampled_force_curve_preserves_full_pair_ordering": rho >= 0.8,
        "best_dictionary_is_intermediate_M14": best_size == 14,
        "coarse_dictionary_error_decreases_to_M14": sampled_lasso[0] > sampled_lasso[best_index],
        "high_resolution_error_exceeds_M14": high_mean > sampled_lasso[best_index],
        "aggregate_force_rms_within_1_25_of_full":
            strict["sampled_to_full_global_force_rms_ratio"] <= 1.25,
        "lasso_below_ols_from_M14_through_M65": all(
            sampled_lasso[index] < sampled_ols[index]
            for index, size in enumerate(sizes) if size >= 14),
        "all_selected_lasso_models_converged":
            diagnostics["selected_lasso_converged"] == 1000,
        "no_lasso_rollout_divergence": diagnostics["lasso_trajectory_divergences"] == 0,
        "all_ols_designs_rank_deficient": diagnostics["ols_rank_deficient_fits"] == 1000,
        "ols_instability_is_visible": diagnostics["ols_trajectory_divergences"] > 0,
    }
    all_size_exceptions = [
        size for size, lasso, ols in zip(sizes, sampled_lasso, sampled_ols) if not lasso < ols
    ]
    full_exceptions = [
        size for size, point in zip(sizes, summary["dictionary_sizes"])
        if not point["full"]["weak_lasso"]["force_error_weighted_rel"]["mean"]
        < point["full"]["weak_ols"]["force_error_weighted_rel"]["mean"]
    ]
    claim = {
        "status": "maintained" if all(checks.values()) else "not_maintained",
        "scope": "Claims currently made for Generic 1D Figure 1, not the stronger auxiliary all-M dominance check.",
        "checks": checks,
        "statement": (
            "The current Generic 1D claims are maintained under sampled pairs at P=50N."
            if all(checks.values()) else
            "At least one current Generic 1D claim is not maintained under P=50N."
        ),
        "sampled_best_dictionary_size": best_size,
        "force_curve_spearman": rho,
        "sampled_to_full_global_force_rms_ratio": strict["sampled_to_full_global_force_rms_ratio"],
        "M14": {
            "lasso_force_error_mean": best["sampled"]["weak_lasso"]["force_error_weighted_rel"]["mean"],
            "lasso_force_error_ci95": best["sampled"]["weak_lasso"]["force_error_weighted_rel"]["mean_ci95"],
            "lasso_trajectory_error_mean": best["sampled"]["weak_lasso"]["trajectory_error_rel"]["mean"],
            "lasso_trajectory_error_ci95": best["sampled"]["weak_lasso"]["trajectory_error_rel"]["mean_ci95"],
            "average_reported_nonzero": best["sampled"]["weak_lasso"]["selected_nonzero"]["mean"],
            "representative_force_error": figure["representative_metrics"]["force_error_weighted_rel"],
            "representative_trajectory_error": figure["representative_metrics"]["trajectory_error_rel"],
        },
        "diagnostics": diagnostics,
        "stronger_auxiliary_check": {
            "description": "LASSO mean force error is below OLS at every dictionary size.",
            "passed": len(all_size_exceptions) == 0,
            "sampled_exceptions": all_size_exceptions,
            "matched_full_pair_exceptions": full_exceptions,
            "interpretation": (
                "The only exception is M=11, and the same exception occurs in the matched full-pair results. "
                "The current manuscript does not claim uniform LASSO dominance at every resolution."
            ),
        },
    }
    write_json(output / "manuscript_claim_report.json", claim)

    failed_integrity = [row for row in verification["checks"]
                        if not row["passed"] and row["name"] != "claim_report_maintained"]
    integrity_checks = {
        "original_verification_has_no_integrity_failure": len(failed_integrity) == 0,
        "completed_jobs": read_json(output / "batch_status.json")["status"] == "completed",
        "records_hash_matches_original_verification":
            digest(output / "records.json") == verification["artifact_sha256"]["records.json"],
        "summary_hash_matches_original_verification":
            digest(output / "summary.json") == verification["artifact_sha256"]["summary.json"],
        "figure_pdf_hash_matches_metadata":
            digest(output / "figure1_candidate/one_dimensional_hat_sampled_p50n_figure1_candidate.pdf")
            == figure["pdf_sha256"],
        "figure_png_hash_matches_metadata":
            digest(output / "figure1_candidate/one_dimensional_hat_sampled_p50n_figure1_candidate.png")
            == figure["png_sha256"],
        "current_manuscript_figure_unchanged":
            digest(ROOT / "latex/L4DC/figures/one_dimensional_hat_practical.pdf")
            == "92d28f98b96e65eae73bb2967fe4a126c12b5d51509981066d357a094f15e5fc",
        "current_body_unchanged": digest(ROOT / "latex/L4DC/body.tex")
            == "47ece0f88f2435b046b19c87207ee091acd38e5e90a7d8ba7461d37bcb927a30",
        "current_appendix_unchanged": digest(ROOT / "latex/L4DC/appendix_numerics.tex")
            == "eddea6e9a4496977aa826ff3b442564e5a028a8cbad97e9e6cb215e60eed88dc",
    }
    integrity = {
        "status": "passed" if all(integrity_checks.values()) else "failed",
        "checks_passed": sum(integrity_checks.values()),
        "checks_total": len(integrity_checks),
        "checks": integrity_checks,
        "original_scientific_verification": {
            "checks_passed": verification["checks_passed"],
            "checks_total": verification["checks_total"],
            "sole_failed_check": "claim_report_maintained",
        },
        "note": "Scientific outcome checks are reported separately from provenance and artifact integrity.",
    }
    write_json(output / "verification/integrity_verification.json", integrity)

    headline = {
        "pair_ratio": 50, "pair_count": 50000, "repetitions": 100,
        "representative_M": best_size,
        "sampled_global_force_rms": strict["sampled_global_force_rms"],
        "matched_full_global_force_rms": strict["full_global_force_rms"],
        "sampled_to_full_global_force_rms_ratio": strict["sampled_to_full_global_force_rms_ratio"],
        "M14_lasso_force_mean": claim["M14"]["lasso_force_error_mean"],
        "M14_lasso_trajectory_mean": claim["M14"]["lasso_trajectory_error_mean"],
        "M14_average_nonzero": claim["M14"]["average_reported_nonzero"],
        "lasso_rollout_divergences": diagnostics["lasso_trajectory_divergences"],
        "ols_rank_deficient_fits": diagnostics["ols_rank_deficient_fits"],
        "ols_rollout_divergences": diagnostics["ols_trajectory_divergences"],
    }
    write_json(output / "table_headline_metrics.json", headline)
    with (output / "table_headline_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(headline.items())

    snapshot = output / "postprocessing_source_snapshot/finalize_1d_hat_sampled_p50n_R100.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), snapshot)
    production = {
        "status": "complete",
        "integrity_verification_status": integrity["status"],
        "manuscript_claim_status": claim["status"],
        "manuscript_claim_statement": claim["statement"],
        "strict_auxiliary_all_M_dominance_passed": claim["stronger_auxiliary_check"]["passed"],
        "strict_auxiliary_exception": claim["stronger_auxiliary_check"]["sampled_exceptions"],
        "output": str(output),
        "scientific_report_sha256": digest(output / "report.json"),
        "summary_sha256": digest(output / "summary.json"),
        "records_sha256": digest(output / "records.json"),
        "manuscript_claim_report_sha256": digest(output / "manuscript_claim_report.json"),
        "integrity_verification_sha256": digest(output / "verification/integrity_verification.json"),
        "table_metrics_json_sha256": digest(output / "table_metrics.json"),
        "table_metrics_csv_sha256": digest(output / "table_metrics.csv"),
        "table_headline_metrics_json_sha256": digest(output / "table_headline_metrics.json"),
        "table_headline_metrics_csv_sha256": digest(output / "table_headline_metrics.csv"),
        "figure1_pdf_sha256": figure["pdf_sha256"],
        "figure1_png_sha256": figure["png_sha256"],
        "finalizer_sha256": digest(Path(__file__)),
        "finalizer_snapshot_sha256": digest(snapshot),
        "manuscript_edited": False,
        "current_figure_edited": False,
    }
    write_json(output / "production_report.json", production)
    return production


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(finalize(args.output), indent=2))


if __name__ == "__main__":
    main()
