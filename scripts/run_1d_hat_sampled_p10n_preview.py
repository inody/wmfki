"""Preview the generic 1D dictionary experiment with sampled pairs at P=10N.

The script reuses the frozen training, validation, and test trajectories from the
current 100-repetition dictionary-grid experiment.  It changes only weak-design
assembly: training and validation use independent sampled ordered pairs.  The
current production results and manuscript figures are never overwritten.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
from time import perf_counter
import traceback

import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hat_sweep_numerics import (
    assemble_sampled,
    evaluate_model,
    evaluation_statistics,
)
from scripts.prepare_1d_hat_experiment import array_hash, write_json
from scripts.run_1d_hat_dictionary_size_R100 import candidates, load_data
from scripts.run_1d_hat_lasso import select_lasso
from scripts.run_1d_hat_sweeps import test_family

DEFAULT_CONFIG = ROOT / "configs/experiments/one_dimensional_hat_sampled_p10n_preview.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def source_context(spec):
    source = ROOT / spec["source_batch"]
    frozen = read_json(source / "frozen_protocol.json")
    status = read_json(source / "batch_status.json")
    if status["status"] != "completed":
        raise ValueError("source dictionary-grid batch is not complete")
    if digest(source / "frozen_protocol.json") != status["frozen_protocol_sha256"]:
        raise ValueError("source frozen protocol hash mismatch")
    if len(frozen["references"]) != spec["scientific_repetitions"]:
        raise ValueError("source reference count mismatch")
    base = deepcopy(frozen["base"])
    base["basis"]["sizes"] = spec["dictionary_sizes"]
    return source, frozen, base


def validate_spec(spec):
    if spec.get("task") != "one_dimensional_hat_sampled_p10n_preview":
        raise ValueError("wrong preview task")
    if spec.get("protocol_version") != 1 or spec.get("pair_ratio") != 10:
        raise ValueError("preview protocol changed")
    if spec.get("dictionary_sizes") != [9, 11, 14, 17, 22, 27, 33, 42, 52, 65]:
        raise ValueError("dictionary grid changed")
    required = {
        "ordered": True,
        "exclude_diagonal": True,
        "with_replacement": True,
        "resample_each_time": True,
        "shared_across_tests_and_atoms": True,
        "independent_training_validation": True,
    }
    if spec.get("pair_sampling") != required:
        raise ValueError("pair-sampling policy changed")


def validate_reference(reference):
    directory = Path(reference["directory"])
    report = directory / "report.json"
    if digest(report) != reference["report_sha256"]:
        raise ValueError("reference report hash mismatch")
    payload = read_json(report)
    for name, expected in payload["artifacts_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"reference artifact changed: {name}")
    return directory


def sampled_split(reference, split, base, pair_count, pair_seed):
    data = load_data(reference / split / "data.npz")
    if data["X"].shape[2] * 10 != pair_count:
        raise ValueError("P is not 10N")
    space, time = test_family(data["t"])
    designs = assemble_sampled(
        data["X"], data["t"], space, time, candidates(base), pair_count, pair_seed
    )
    response = np.load(reference / split / "response.npy")
    return designs, response, data


def fit_one(index, reference, base, spec, destination):
    started = perf_counter()
    out = Path(destination) / f"run-{index + 1:03d}"
    out.mkdir(parents=True, exist_ok=False)
    record = {"index": index, "status": "running"}
    write_json(out / "status.json", record)
    try:
        reference = validate_reference(reference)
        n_particles = int(base["simulation"]["N"])
        pair_count = int(spec["pair_ratio"] * n_particles)
        pair_seed_base = int(spec["pair_seed_start"] + spec["pair_seed_stride"] * index)
        with threadpool_limits(limits=1):
            train_A, train_b, _ = sampled_split(
                reference, "train", base, pair_count,
                pair_seed_base + spec["training_pair_seed_offset"],
            )
            val_A, val_b, _ = sampled_split(
                reference, "validation", base, pair_count,
                pair_seed_base + spec["validation_pair_seed_offset"],
            )
            test = load_data(reference / "test/data.npz")
            stats = evaluation_statistics(
                test["X"], test["t"], candidates(base), base["true_kernel"]
            )
            results = {}
            for name, params in candidates(base):
                selection, _ = select_lasso(
                    train_A[name], train_b, val_A[name], val_b, base["estimator"]
                )
                methods = {}
                if selection["status"] == "selected":
                    theta = np.asarray(selection["theta"], dtype=float)
                    metrics, rollout = evaluate_model(
                        test, params, theta, base["simulation"]["sigma"], stats[name]
                    )
                    methods["weak_lasso"] = {
                        "status": "evaluated",
                        "theta": theta.tolist(),
                        "selection": selection,
                        "metrics": metrics,
                    }
                    if rollout is not None:
                        methods["weak_lasso"]["rollout_sha256"] = array_hash(rollout)
                else:
                    methods["weak_lasso"] = {
                        "status": selection["status"], "selection": selection, "metrics": {}
                    }
                theta_ols, _, rank, singular_values = np.linalg.lstsq(
                    train_A[name], train_b, rcond=None
                )
                metrics_ols, rollout_ols = evaluate_model(
                    test, params, theta_ols, base["simulation"]["sigma"], stats[name]
                )
                methods["weak_ols"] = {
                    "status": "evaluated",
                    "theta": theta_ols.tolist(),
                    "rank": int(rank),
                    "singular_values": singular_values.tolist(),
                    "metrics": metrics_ols,
                }
                if rollout_ols is not None:
                    methods["weak_ols"]["rollout_sha256"] = array_hash(rollout_ols)
                results[name] = {"basis": params, "methods": methods}
        record.update(
            status="completed",
            reference_directory=str(reference),
            reference_report_sha256=reference.name and digest(reference / "report.json"),
            particles=n_particles,
            sampled_pairs=pair_count,
            pair_seeds={
                "training": pair_seed_base + spec["training_pair_seed_offset"],
                "validation": pair_seed_base + spec["validation_pair_seed_offset"],
            },
            results=results,
        )
    except Exception as error:
        record.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    record["elapsed_seconds"] = perf_counter() - started
    write_json(out / "report.json", record)
    write_json(out / "status.json", {
        "status": record["status"], "elapsed_seconds": record["elapsed_seconds"]
    })
    return {
        "index": index,
        "status": record["status"],
        "directory": str(out),
        "report_sha256": digest(out / "report.json"),
        "elapsed_seconds": record["elapsed_seconds"],
        "error": record.get("error"),
    }


def summarize(destination, records, sizes):
    rows = []
    for item in records:
        report = read_json(Path(item["directory"]) / "report.json")
        if report["status"] != "completed":
            continue
        for size in sizes:
            name = f"hat_M{size}"
            for method, result in report["results"][name]["methods"].items():
                metrics = result.get("metrics", {})
                rows.append({
                    "index": report["index"],
                    "M": size,
                    "method": method,
                    "force_error": metrics.get("force_error_weighted_rel"),
                    "trajectory_error": metrics.get("trajectory_error_rel"),
                    "trajectory_status": metrics.get("trajectory_status"),
                })
    summary = {"records": rows, "aggregate": {}}
    for size in sizes:
        for method in ("weak_lasso", "weak_ols"):
            values = [r["force_error"] for r in rows
                      if r["M"] == size and r["method"] == method and r["force_error"] is not None]
            summary["aggregate"][f"M{size}:{method}"] = {
                "count": len(values),
                "mean_force_error": float(np.mean(values)) if values else None,
            }
    write_json(Path(destination) / "summary.json", summary)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = {"weak_lasso": "tab:red", "weak_ols": "tab:blue"}
    labels = {"weak_lasso": "weak LASSO", "weak_ols": "weak OLS"}
    for method in ("weak_lasso", "weak_ols"):
        means = [summary["aggregate"][f"M{size}:{method}"]["mean_force_error"] for size in sizes]
        ax.plot(sizes, means, marker="o", color=colors[method], label=labels[method])
        for row in rows:
            if row["method"] == method and row["force_error"] is not None:
                ax.scatter(row["M"], row["force_error"], color=colors[method], alpha=0.25, s=12)
    ax.set_xlabel("dictionary size M")
    ax.set_ylabel("held-out relative force error")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.set_title("SMOKE PREVIEW: sampled weak systems; not a manuscript result")
    fig.tight_layout()
    fig.savefig(Path(destination) / "sampled_p10n_smoke_preview.pdf")
    fig.savefig(Path(destination) / "sampled_p10n_smoke_preview.png", dpi=180)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.smoke == args.run:
        parser.error("choose exactly one of --smoke or --run")
    spec = read_json(args.config)
    validate_spec(spec)
    _, frozen, base = source_context(spec)
    count = spec["smoke_repetitions"] if args.smoke else spec["scientific_repetitions"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = (args.out_dir or ROOT / "outputs/validation/one_dimensional_hat/sampledp10n-preview" /
                   (("smoke-" if args.smoke else "scientific-") + stamp)).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "protocol.json", {
        "requested": spec,
        "smoke_only": args.smoke,
        "repetitions": count,
        "source_frozen_protocol_sha256": digest(ROOT / spec["source_batch"] / "frozen_protocol.json"),
        "source_config_sha256": digest(args.config),
    })
    jobs = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {}
        for index in range(count):
            reference = frozen["references"][f"run-{index}"]
            future = pool.submit(fit_one, index, reference, base, spec, destination)
            futures[future] = index
        for future in as_completed(futures):
            item = future.result()
            jobs.append(item)
            print(f"{len(jobs)}/{count}: run-{item['index'] + 1:03d} {item['status']} "
                  f"{item['elapsed_seconds']:.1f}s", flush=True)
    jobs.sort(key=lambda x: x["index"])
    summary = summarize(destination, jobs, spec["dictionary_sizes"])
    report = {
        "status": "completed" if all(x["status"] == "completed" for x in jobs) else "completed_with_failures",
        "destination": str(destination),
        "jobs": jobs,
        "summary_sha256": digest(destination / "summary.json"),
        "preview_pdf_sha256": digest(destination / "sampled_p10n_smoke_preview.pdf"),
        "aggregate": summary["aggregate"],
    }
    write_json(destination / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
