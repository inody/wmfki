"""Select LASSO on independent validation trajectories and evaluate the frozen coefficients on unused trajectories."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import make_trapz_weights
from kernels import make_kernel_basis, make_kernel_evaluator, make_radial_basis
from scripts.prepare_1d_hat_experiment import array_hash, validate_spec, write_json
from simulate import make_true_kernel, simulate_with_kernel


def select_lasso(A, b, validation_A, validation_b, settings):
    """Use only training and validation matrices for selection; do not accept the true function or final evaluation data."""
    A, b = np.asarray(A), np.asarray(b)
    validation_A, validation_b = np.asarray(validation_A), np.asarray(validation_b)
    if (A.ndim != 2 or b.shape != (A.shape[0],) or validation_A.ndim != 2
            or validation_A.shape[1] != A.shape[1]
            or validation_b.shape != (validation_A.shape[0],)):
        raise ValueError("incompatible training/validation shapes")
    if any(not np.all(np.isfinite(x)) for x in (A, b, validation_A, validation_b)):
        raise ValueError("nonfinite training/validation arrays")
    alpha_max = float(np.max(np.abs(A.T @ b)) / A.shape[0])
    if alpha_max <= 0:
        raise ValueError("zero training score: no positive alpha grid can be defined")
    ratios = np.geomspace(1.0, settings["alpha_min_ratio"], settings["alpha_count"])
    model = Lasso(
        alpha=alpha_max, fit_intercept=False, warm_start=True,
        max_iter=settings["max_iter"], tol=settings["tol"], selection="cyclic",
    )
    rows, coefficients = [], []
    for index, ratio in enumerate(ratios):
        alpha = alpha_max * float(ratio)
        model.alpha = alpha
        started = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(A, b)
        theta = model.coef_.copy()
        finite = bool(np.all(np.isfinite(theta)) and np.isfinite(model.dual_gap_))
        warning_count = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        train_mse = float(np.mean((A @ theta - b) ** 2)) if finite else None
        validation_mse = float(np.mean((validation_A @ theta - validation_b) ** 2)) if finite else None
        if finite and (not np.isfinite(train_mse) or not np.isfinite(validation_mse)):
            finite, train_mse, validation_mse = False, None, None
        row = {
            "index": index, "alpha_sk": alpha, "alpha_over_max": float(ratio),
            "lambda_summed_loss": A.shape[0] * alpha,
            "train_mse": train_mse, "validation_mse": validation_mse,
            "converged": finite and warning_count == 0,
            "convergence_warnings": warning_count, "iterations": int(model.n_iter_),
            "dual_gap": float(model.dual_gap_) if finite else None,
            "nonzero_exact": int(np.count_nonzero(theta)) if finite else None,
            "nonzero_reported": int(np.count_nonzero(np.abs(theta) > settings["report_nonzero_threshold"])) if finite else None,
            "fit_seconds": perf_counter() - started,
        }
        rows.append(row)
        coefficients.append(theta)
    valid = [row for row in rows if row["converged"]]
    if not valid:
        return {"status": "no_converged_candidate", "alpha_max": alpha_max, "path": rows}, np.asarray(coefficients)
    # The search proceeds from larger alpha values, so ties in validation error select the larger alpha.
    selected = min(valid, key=lambda row: row["validation_mse"])
    return {
        "status": "selected", "alpha_max": alpha_max, "path": rows,
        "selected": selected, "selection_used": "independent_validation_weak_residual_mse",
        "selected_at_grid_boundary": selected["index"] in (0, len(rows) - 1),
        "theta": coefficients[selected["index"]].tolist(),
    }, np.asarray(coefficients)


def load_split(assembly, manifest, split, candidate):
    matrices, responses = [], []
    for entry in manifest["splits"][split]:
        folder = assembly / split / str(entry["seed"])
        A, b = np.load(folder / f"{candidate}.npy"), np.load(folder / "response.npy")
        if array_hash(A) != entry["designs"][candidate]["sha256"] or array_hash(b) != entry["response_sha256"]:
            raise ValueError("saved matrix or response hash mismatch")
        matrices.append(A)
        responses.append(b)
    return np.vstack(matrices), np.concatenate(responses)


def relative_error(numerator, denominator):
    if denominator <= 0 or not np.isfinite(numerator + denominator):
        return None
    return float(np.sqrt(numerator / denominator))


def force_and_drift_errors(X, t, true_kernel, learned_kernel):
    """Compute force errors over all ordered pairs and observation times, and drift errors over all particles."""
    force_error = force_norm = drift_error = drift_norm = 0.0
    particles = X.shape[2]
    for path in X:
        for x, weight in zip(path, make_trapz_weights(t)):
            z = x[:, None, :] - x[None, :, :]
            truth = true_kernel(z.reshape(-1, X.shape[-1])).reshape(z.shape)
            fitted = learned_kernel(z.reshape(-1, X.shape[-1])).reshape(z.shape)
            # Both kernels satisfy K(0)=0; also exclude self-interaction explicitly.
            truth[np.arange(particles), np.arange(particles)] = 0
            fitted[np.arange(particles), np.arange(particles)] = 0
            force_error += weight * float(np.sum((fitted - truth) ** 2))
            force_norm += weight * float(np.sum(truth ** 2))
            true_drift, fitted_drift = truth.sum(axis=1) / particles, fitted.sum(axis=1) / particles
            drift_error += weight * float(np.sum((fitted_drift - true_drift) ** 2))
            drift_norm += weight * float(np.sum(true_drift ** 2))
    return {
        "force_error_weighted_rel": relative_error(force_error, force_norm),
        "drift_error_rel": relative_error(drift_error, drift_norm),
        "force_measure": "all off-diagonal pair distances weighted by trapezoidal time quadrature",
    }


def make_evaluation_kernel(params, theta, chunk_size=16384):
    """Evaluate the same estimated function without constructing an N^2*M array, preserving the coefficients and extension rule."""
    theta = np.asarray(theta, dtype=float)
    basis, _ = make_kernel_basis(params)
    if theta.shape != (params["M"],) or not np.all(np.isfinite(theta)):
        raise ValueError("coefficients must be finite and match the basis")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    generic = make_kernel_evaluator(theta, basis)

    def kernel(z):
        z = np.asarray(z, dtype=float)
        if z.ndim == 0:
            z = z[None]
        if z.ndim == 1:
            z = z[:, None]
        if params["basis"] == "hat":
            radius = np.linalg.norm(z, axis=-1)
            # np.interp uses constant extension outside the endpoints, giving the same function as the linear combination of nodal hats.
            profile = np.interp(radius.ravel(), params["knots"], theta).reshape(radius.shape)
            return z * profile[..., None]
        flat = z.reshape(-1, z.shape[-1])
        result = np.empty_like(flat)
        for start in range(0, len(flat), chunk_size):
            result[start:start + chunk_size] = generic(flat[start:start + chunk_size])
        return result.reshape(z.shape)

    return kernel


def evaluate_selected(data_path, params, theta, simulation, true_params, radius_max):
    with np.load(data_path) as data:
        X, t, X0, noise = (data[key].copy() for key in ("X", "t", "X0", "driving_noise"))
    true_kernel, _, _, true_profile = make_true_kernel(true_params)
    learned_kernel = make_evaluation_kernel(params, theta)
    metrics = force_and_drift_errors(X, t, true_kernel, learned_kernel)
    radial_basis, _ = make_radial_basis(params)
    radius = np.linspace(0.02, radius_max, 800)
    profile_truth, profile_fit = true_profile(radius), radial_basis(radius) @ theta
    metrics["profile_error_grid_rel"] = relative_error(
        float(np.sum((profile_fit - profile_truth) ** 2)), float(np.sum(profile_truth ** 2)),
    )
    metrics["profile_grid_interval"] = [0.02, radius_max]
    metrics["coefficient_l1"] = float(np.linalg.norm(theta, ord=1))
    metrics["trajectory_status"] = "not_run"
    predicted = None

    def guarded_kernel(z):
        if np.any(~np.isfinite(z)) or np.max(np.abs(z)) > 2e3:
            raise FloatingPointError("rollout left the numerical safety range")
        return learned_kernel(z)

    try:
        with np.errstate(over="raise", invalid="raise"):
            predicted = simulate_with_kernel(guarded_kernel, X0, t, sigma=simulation["sigma"], noise=noise)
        if np.any(~np.isfinite(predicted)) or np.max(np.abs(predicted)) > 1e3:
            raise FloatingPointError("nonfinite or divergent rollout")
        difference = predicted - X
        metrics.update({
            "trajectory_status": "finite", "trajectory_error_rel": relative_error(
                float(np.sum(difference ** 2)), float(np.sum(X ** 2))),
            "trajectory_rmse": float(np.sqrt(np.mean(difference ** 2))),
            "trajectory_max_pair_distance": float(np.max(np.ptp(predicted[..., 0], axis=2))),
            "trajectory_exits_basis_interval": bool(np.max(np.ptp(predicted[..., 0], axis=2)) > radius_max),
        })
    except FloatingPointError as error:
        predicted = None
        metrics.update({"trajectory_status": "diverged", "trajectory_error_rel": None,
                        "trajectory_rmse": None, "trajectory_failure": str(error)})
    return metrics, predicted


def make_plots(out_dir, results, spec, test_data, distances):
    successful = {name: item for name, item in results.items() if item["status"] == "evaluated"}
    if not successful:
        return
    _, _, _, true_profile = make_true_kernel(spec["true_kernel"])
    radius = np.linspace(0.02, spec["basis"]["radius_max"], 800)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(radius, true_profile(radius), color="black", label="True")
    axes[0, 1].plot(radius, radius * true_profile(radius), color="black", label="True")
    for name, result in successful.items():
        phi, _ = make_radial_basis(result["basis"])
        fitted = phi(radius) @ np.asarray(result["theta"])
        label = name.replace("hat_M", "Hat M=").replace("baseline_six_atoms", "6-atom dictionary")
        axes[0, 0].plot(radius, fitted, label=label)
        axes[0, 1].plot(radius, radius * fitted, label=label)
        path = result["selection"]["path"]
        converged = [row for row in path if row["converged"]]
        axes[1, 0].plot([row["nonzero_reported"] for row in converged],
                        [row["validation_mse"] for row in converged], ".-", label=label, alpha=0.65)
        chosen = result["selection"]["selected"]
        axes[1, 0].plot(chosen["nonzero_reported"], chosen["validation_mse"], "ko", markersize=4)
    axes[0, 0].set(xlabel="r", ylabel="Radial profile", title="Selected by validation, not truth")
    axes[0, 1].set(xlabel="r", ylabel="r × radial profile", title="Interaction force")
    axes[1, 0].set(xlabel="Nonzero coefficients (|coefficient| > 1e-8)", ylabel="Validation weak-residual MSE", yscale="log")
    edges = np.asarray(distances["edges"])
    axes[1, 1].stairs(np.asarray(distances["bin_probability"]) / np.diff(edges), edges, fill=True, alpha=0.5)
    axes[1, 1].set(xlabel="Pair distance r", ylabel="Density", title="Held-out distance distribution")
    axes[0, 1].legend(fontsize=8)
    fig.savefig(out_dir / "pilot_summary.png", dpi=160)
    plt.close(fig)

    with np.load(test_data) as data:
        X, t = data["X"], data["t"]
    indices = np.linspace(0, X.shape[2] - 1, min(15, X.shape[2]), dtype=int)
    indices = np.argsort(X[0, 0, :, 0])[indices]
    fig, axes = plt.subplots(1, len(successful), figsize=(3 * len(successful), 3.5), squeeze=False, constrained_layout=True)
    for ax, (name, result) in zip(axes[0], successful.items()):
        ax.plot(t, X[0, :, indices, 0].T, color="black", linewidth=0.7, alpha=0.5)
        rollout_path = out_dir / name / "rollout_0.npz"
        if rollout_path.exists():
            with np.load(rollout_path) as data:
                prediction = data["X_hat"]
            ax.plot(t, prediction[0, :, indices, 0].T, color="tab:orange", linewidth=0.8, alpha=0.8)
        ax.set(title=name, xlabel="t", ylabel="Position")
    fig.suptitle("Held-out initial state; common Brownian input (black: true, orange: learned)")
    fig.savefig(out_dir / "heldout_trajectories.png", dpi=160)
    plt.close(fig)


def run(assembly, out_dir, *, allow_smoke=False):
    assembly, out_dir = Path(assembly), Path(out_dir)
    manifest_path = assembly / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] not in ("assembled_not_fitted", "smoke_passed"):
        raise ValueError("assembly is incomplete")
    if manifest["smoke_only"] and not allow_smoke:
        raise ValueError("smoke data require --allow-smoke; they are not a scientific experiment")
    for name, digest in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"assembly source has changed: {name}; rebuild the assembly")
    spec = manifest["effective_protocol"]
    validate_spec(manifest["requested_protocol"])
    settings = spec["estimator"]
    if settings is None or settings["method"] != "lasso":
        raise ValueError("LASSO settings must be specified before assembling data")
    out_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "selecting", "smoke_only": manifest["smoke_only"],
        "assembly_directory": str(assembly.resolve()),
        "assembly_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("scripts/run_1d_hat_lasso.py", "kernels.py", "simulate.py")},
        "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
        "protocol": spec, "ols_performed": False, "results": {},
        "kernel_evaluation": "hat interpolation / chunked other bases; same fitted function",
    }
    write_json(out_dir / "report.json", report)
    started = perf_counter()
    try:
        names = list(manifest["splits"]["train"][0]["designs"])
        selections = {}
        # Freeze the coefficients for every M before loading the final evaluation data.
        for name in names:
            A, b = load_split(assembly, manifest, "train", name)
            validation_A, validation_b = load_split(assembly, manifest, "validation", name)
            selection, coefficients = select_lasso(A, b, validation_A, validation_b, settings)
            folder = out_dir / name
            folder.mkdir()
            np.save(folder / "path_coefficients.npy", coefficients)
            write_json(folder / "selection.json", selection)
            params = manifest["splits"]["train"][0]["designs"][name]["basis"]
            selections[name] = {"basis": params, **selection}
            print(name, "selection:", selection.get("selected", selection["status"]), flush=True)
        write_json(out_dir / "frozen_selections.json", selections)
        report["frozen_selections_sha256"] = hashlib.sha256((out_dir / "frozen_selections.json").read_bytes()).hexdigest()
        report["status"] = "evaluating_frozen_models"
        write_json(out_dir / "report.json", report)
        for name, selection in selections.items():
            if selection["status"] != "selected":
                report["results"][name] = {"status": selection["status"], "selection": selection}
                continue
            theta = np.asarray(selection["theta"])
            test_A, test_b = load_split(assembly, manifest, "test", name)
            result = {
                "status": "evaluated", "basis": selection["basis"], "theta": theta.tolist(),
                "selection": selection,
                "test_weak_residual_mse": float(np.mean((test_A @ theta - test_b) ** 2)),
                "test_metrics": [],
            }
            for index, entry in enumerate(manifest["splits"]["test"]):
                data_path = assembly / "test" / str(entry["seed"]) / "data.npz"
                with np.load(data_path) as data:
                    if array_hash(data["Y"]) != entry["observation_sha256"] or array_hash(data["t"]) != entry["time_sha256"]:
                        raise ValueError("evaluation data hash mismatch")
                metrics, rollout = evaluate_selected(data_path, selection["basis"], theta,
                                                      spec["simulation"], spec["true_kernel"], spec["basis"]["radius_max"])
                result["test_metrics"].append({"seed": entry["seed"], **metrics})
                if rollout is not None:
                    np.savez_compressed(out_dir / name / f"rollout_{index}.npz", X_hat=rollout)
            report["results"][name] = result
            write_json(out_dir / name / "evaluation.json", result)
            write_json(out_dir / "report.json", report)
        first_test = manifest["splits"]["test"][0]
        make_plots(out_dir, report["results"], spec,
                   assembly / "test" / str(first_test["seed"]) / "data.npz", first_test["distances"])
        all_evaluated = all(item["status"] == "evaluated" for item in report["results"].values())
        report["status"] = "completed" if all_evaluated else "completed_with_fit_failures"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = perf_counter() - started
        write_json(out_dir / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.out_dir or ROOT / "outputs/validation/one_dimensional_hat" / f"lasso-{timestamp}"
    with threadpool_limits(limits=1):
        report = run(args.assembly, destination, allow_smoke=args.allow_smoke)
    print(json.dumps({"status": report["status"], "output": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
