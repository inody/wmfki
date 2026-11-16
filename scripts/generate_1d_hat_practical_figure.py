"""Generate the three-panel practical hat-dictionary figure without editing the manuscript."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from features import make_trapz_weights
from kernels import make_kernel_basis
from scripts.manuscript_plot_style import (
    ESTIMATE_COLOR, GRID_COLOR, REFERENCE_COLOR, TERTIARY_COLOR, TRUE_COLOR,
    apply_manuscript_style, finish_axes,
)
from scripts.prepare_1d_hat_experiment import array_hash
from scripts.run_1d_hat_dictionary_size_R100 import digest, read_json
from scripts.hat_sweep_numerics import simulate_hat
from scripts.run_1d_hat_sampled_pair_ratio_sweep_R5 import candidates
from scripts.run_1d_hat_sweeps import load_data, test_family
from simulate import make_true_kernel
from utils import make_pair_sampler

PANEL_BOX_ASPECT = 3 / 4


def sampled_pair_quantile(data, pair_count, seed, quantile=0.99):
    """Quantile of the exact sampled-pair stream used by the weak design."""
    path = data["X"][0, :, :, 0]
    time = data["t"]
    _, temporal = test_family(time)
    sampler = make_pair_sampler(
        pair_count,
        np.random.default_rng(seed),
        with_replacement=True,
        exclude_diagonal=True,
        resample_each_call=True,
    )
    edges = np.linspace(0.0, 4.5, 9001)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    total = 0
    for index, weight in enumerate(make_trapz_weights(time)):
        if not np.any(weight * temporal["psi"][:, index]):
            continue
        source, target = sampler(path.shape[1])
        radius = np.abs(path[index, source] - path[index, target])
        counts += np.histogram(radius, bins=edges)[0]
        total += len(radius)
    cumulative = np.cumsum(counts)
    bin_index = min(np.searchsorted(cumulative, quantile * total), len(counts) - 1)
    return float((edges[bin_index] + edges[bin_index + 1]) / 2), int(total)


def resolve_representative(batch, summary, selection):
    if selection == "fixed":
        representative = summary["representative"]
        report_path = Path(representative["source_report"])
        expected_hash = representative["source_report_sha256"]
        M = int(representative["dictionary"].removeprefix("hat_M"))
        policy = representative["policy"]
        post_hoc = False
    elif selection == "aggregate-force-minimum":
        point = min(
            summary["dictionary_sizes"],
            key=lambda row: row["total100"]["metrics"]["force_error_weighted_rel"]["mean"],
        )
        M = int(point["value"])
        if point.get("source") != "new":
            raise ValueError("aggregate minimum is not available in the new-model reports")
        source = summary["inputs"]["new"]["run-0"]
        report_path = batch / source["path"]
        expected_hash = source["sha256"]
        policy = (
            f"M={M} minimizes the reported 100-repetition mean held-out force error; "
            "within that size, use global repetition 0 without seed selection."
        )
        post_hoc = True
    else:
        raise ValueError("unknown representative selection")
    if digest(report_path) != expected_hash:
        raise ValueError("representative source report changed")
    report = read_json(report_path)
    directory = report_path.parent
    name = f"hat_M{M}"
    model = report["results"][name]
    if model["status"] != "evaluated" or model["basis"]["M"] != M:
        raise ValueError("requested representative model is unavailable")
    return {
        "M": M, "name": name, "policy": policy, "post_hoc": post_hoc,
        "report_path": report_path, "report_sha256": expected_hash,
        "report": report, "directory": directory, "model": model,
    }


def generate_p50_publication(
    batch, destination, strong_lasso_run=None, noisy_three_method_run=None
):
    summary = read_json(batch / "summary.json")
    frozen = read_json(batch / "frozen_protocol.json")
    records = read_json(batch / "records.json")["records"]
    size = int(min(
        summary["dictionary_sizes"],
        key=lambda point: point["sampled"]["weak_lasso"]["drift_error_rel"]["mean"],
    )["M"])
    representative = next(
        row for row in records
        if row.get("mode") == "sampled" and row.get("method") == "weak_lasso"
        and row.get("repetition") == 0 and row.get("M") == size
    )
    bundle = frozen["reference_bundles"]["run-0"]
    reference_directory = Path(bundle["reference_directory"])
    test = load_data(reference_directory / "test/data.npz")
    train_path = reference_directory / "train/data.npz"
    expected_train_hash = bundle["critical_artifacts_sha256"].get(str(train_path))
    if expected_train_hash is None or digest(train_path) != expected_train_hash:
        raise ValueError("representative training trajectory changed")
    train = load_data(train_path)
    pair_seed = (
        frozen["requested_protocol"]["pair_seed_start"]
        + frozen["requested_protocol"]["training_pair_seed_offset"]
    )
    support_q99, support_pair_count = sampled_pair_quantile(
        train, frozen["requested_protocol"]["pair_count"], pair_seed, 0.99
    )
    rollout_path = batch / "figure1_candidate/representative_rollout.npz"
    with np.load(rollout_path) as saved:
        rollout = saved["X_hat"]
    candidate_metadata = read_json(batch / "figure1_candidate/metadata.json")
    if array_hash(rollout) != candidate_metadata["rollout_sha256"]:
        raise ValueError("representative P50 rollout changed")
    params = dict(candidates(frozen["effective_base"]))[f"hat_M{size}"]
    basis, _ = make_kernel_basis(params)
    truth, *_ = make_true_kernel(frozen["effective_base"]["true_kernel"])

    if noisy_three_method_run is not None:
        noisy_three_method_run = Path(noisy_three_method_run).resolve()
        noisy_records = read_json(noisy_three_method_run / "records.json")
        aggregate_rows = sorted(
            (
                row for row in noisy_records
                if row.get("method") == "weak_lasso_validation_selected"
                and row.get("M") == size and row.get("status") == "finite"
            ),
            key=lambda row: row["repetition"],
        )
        aggregate_theta = np.asarray(
            [row["coefficients"] for row in aggregate_rows], dtype=float
        )
        if len(aggregate_rows) != frozen["requested_protocol"]["repetitions"]:
            raise ValueError("noisy representative aggregate curve is incomplete")
        representative_theta = aggregate_theta[0]
        rollout = simulate_hat(
            params, representative_theta, test["X0"], test["t"],
            test["driving_noise"], frozen["effective_base"]["simulation"]["sigma"],
        )
    else:
        aggregate_rows = sorted(
            (
                row for row in records
                if row.get("mode") == "sampled"
                and row.get("method") == "weak_lasso"
                and row.get("M") == size
                and row.get("status") == "evaluated"
            ),
            key=lambda row: row["repetition"],
        )
        if len(aggregate_rows) != frozen["requested_protocol"]["repetitions"]:
            raise ValueError("M=14 aggregate curve is missing repetitions")
        aggregate_theta = np.asarray([row["theta"] for row in aggregate_rows], dtype=float)

    apply_manuscript_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55), constrained_layout=True)
    displacement = np.linspace(-3.5, 3.5, 1201)
    vectors = basis(displacement[:, None])[..., 0]
    learned_curves = aggregate_theta @ vectors.T
    if noisy_three_method_run is not None:
        center_theta = np.median(aggregate_theta, axis=0)
        learned_center = np.median(learned_curves, axis=0)
        center_label = "weak LASSO median"
    else:
        center_theta = aggregate_theta.mean(axis=0)
        learned_center = learned_curves.mean(axis=0)
        center_label = "weak LASSO mean"
    center_components = vectors * center_theta[None, :]
    learned_lower = np.quantile(learned_curves, 0.1, axis=0)
    learned_upper = np.quantile(learned_curves, 0.9, axis=0)
    exact = truth(displacement[:, None])[:, 0]
    axes[0].axvspan(displacement[0], -support_q99, color=GRID_COLOR, alpha=.12)
    axes[0].axvspan(support_q99, displacement[-1], color=GRID_COLOR, alpha=.12)
    threshold = frozen["effective_base"]["estimator"]["report_nonzero_threshold"]
    for index in range(center_theta.size):
        if abs(center_theta[index]) > threshold:
            axes[0].plot(
                displacement, center_components[:, index],
                color=REFERENCE_COLOR, linewidth=.45, alpha=.24,
            )
    axes[0].fill_between(
        displacement, learned_lower, learned_upper,
        color=ESTIMATE_COLOR, alpha=.18, linewidth=0,
        label=r"weak LASSO 10--90\%",
    )
    axes[0].plot(displacement, exact, color=TRUE_COLOR, linewidth=1.5, label="true")
    axes[0].plot(displacement, learned_center, "--", color=ESTIMATE_COLOR,
                 linewidth=1.5, label=center_label)
    axes[0].axhline(0, color=GRID_COLOR, linewidth=.6)
    axes[0].axvline(0, color=GRID_COLOR, linewidth=.6)
    axes[0].set(xlabel=r"relative displacement $z$",
                ylabel=r"interaction $K(z)$",
                title="(a) Interaction recovery")
    axes[0].legend(loc="best", fontsize=6.7)

    true_trajectory = test["X"][0, :, :, 0]
    fitted_trajectory = rollout[0, :, :, 0]
    time = test["t"]
    axes[1].plot(time, true_trajectory, color=TRUE_COLOR, linewidth=.18,
                 alpha=.055, rasterized=True)
    axes[1].plot(time, fitted_trajectory, "--", color=ESTIMATE_COLOR,
                 linewidth=.18, alpha=.07, rasterized=True)
    axes[1].set(xlabel=r"time $t$", ylabel="particle position",
                title=rf"(b) All trajectories ($N={true_trajectory.shape[1]}$)")
    axes[1].legend(handles=[
        Line2D([0], [0], color=TRUE_COLOR, linewidth=1.0, label="true"),
        Line2D([0], [0], color=ESTIMATE_COLOR, linestyle="--",
               linewidth=1.0, label="learned"),
    ], loc="best", fontsize=7.5)

    rng = np.random.default_rng(970050)
    sizes = np.asarray([point["M"] for point in summary["dictionary_sizes"]], dtype=float)
    strong_source = None
    noisy_source = None
    if noisy_three_method_run is not None:
        if strong_lasso_run is not None:
            raise ValueError("choose either a strong-only or noisy three-method overlay")
        noisy_three_method_run = Path(noisy_three_method_run).resolve()
        noisy_summary_path = noisy_three_method_run / "summary.json"
        noisy_summary = read_json(noisy_summary_path)
        grouped = {
            (int(row["M"]), row["method"]): row["drift_error_rel"]
            for row in noisy_summary["groups"]
        }
        method_styles = (
            ("weak_lasso_validation_selected", "^--", ESTIMATE_COLOR,
             r"weak LASSO"),
            ("weak_ols", "o-", TRUE_COLOR, r"weak OLS"),
            ("strong_forward_difference_lasso_validation_selected", "s-.",
             TERTIARY_COLOR, r"strong FD--LASSO"),
        )
        expected = {(int(size), method) for size in sizes for method, *_ in method_styles}
        if set(grouped) != expected:
            raise ValueError("noisy three-method dictionary-size grid differs")
        for size in sizes:
            values = np.asarray(grouped[(int(size), method_styles[0][0])]["values"])
            axes[2].scatter(
                size * np.exp(rng.uniform(-.035, .035, len(values))), values,
                s=5.5, alpha=.12, color=REFERENCE_COLOR, linewidths=0,
            )
        for method, fmt, color, label in method_styles:
            stats = [grouped[(int(size), method)] for size in sizes]
            means = np.asarray([row["mean"] for row in stats], dtype=float)
            lower = np.asarray([row["mean_ci95"][0] for row in stats], dtype=float)
            upper = np.asarray([row["mean_ci95"][1] for row in stats], dtype=float)
            axes[2].errorbar(
                sizes, means, yerr=np.vstack([means - lower, upper - means]),
                fmt=fmt, color=color, linewidth=1.25, markersize=3.7,
                capsize=2.5, label=label,
            )
        noisy_source = {
            "path": str(noisy_three_method_run),
            "summary_sha256": digest(noisy_summary_path),
            "observation_noise": 0.01,
        }
        panel_c_title = r"(c) Dictionary size"
    else:
        for point in summary["dictionary_sizes"]:
            M = point["M"]
            values = np.asarray(
                point["sampled"]["weak_lasso"]["drift_error_rel"]["values"]
            )
            axes[2].scatter(
                M * np.exp(rng.uniform(-.035, .035, len(values))), values,
                s=5.5, alpha=.12, color=REFERENCE_COLOR, linewidths=0,
            )
        for method, fmt, color, label in (
            ("weak_lasso", "^--", ESTIMATE_COLOR, r"weak LASSO mean (95\% CI)"),
            ("weak_ols", "o-", TRUE_COLOR, r"weak OLS mean (95\% CI)"),
        ):
            stats = [point["sampled"][method]["drift_error_rel"]
                     for point in summary["dictionary_sizes"]]
            means = np.asarray([row["mean"] for row in stats], dtype=float)
            lower = np.asarray([row["mean_ci95"][0] for row in stats], dtype=float)
            upper = np.asarray([row["mean_ci95"][1] for row in stats], dtype=float)
            axes[2].errorbar(
                sizes, means, yerr=np.vstack([means - lower, upper - means]),
                fmt=fmt, color=color, linewidth=1.35, markersize=4,
                capsize=2.5, label=label,
            )
        if strong_lasso_run is not None:
            strong_lasso_run = Path(strong_lasso_run).resolve()
            strong_summary_path = strong_lasso_run / "summary.json"
            strong_summary = read_json(strong_summary_path)
            strong_by_size = {int(row["M"]): row for row in strong_summary["groups"]}
            if set(strong_by_size) != set(map(int, sizes)):
                raise ValueError("strong FD--LASSO dictionary-size grid differs")
            strong_stats = [strong_by_size[int(size)]["drift_error_rel"] for size in sizes]
            strong_means = np.asarray([row["mean"] for row in strong_stats], dtype=float)
            strong_lower = np.asarray([row["mean_ci95"][0] for row in strong_stats], dtype=float)
            strong_upper = np.asarray([row["mean_ci95"][1] for row in strong_stats], dtype=float)
            axes[2].errorbar(
                sizes, strong_means,
                yerr=np.vstack([strong_means - strong_lower, strong_upper - strong_means]),
                fmt="s-.", color=TERTIARY_COLOR, linewidth=1.25, markersize=3.7,
                capsize=2.5, label=r"strong FD--LASSO mean (95\% CI)",
            )
            strong_source = {
                "path": str(strong_lasso_run),
                "summary_sha256": digest(strong_summary_path),
            }
        panel_c_title = r"(c) Dictionary size ($P=50N$)"
    axes[2].set(xscale="log", yscale="log", xlabel=r"dictionary size $M$",
                ylabel="drift error", title=panel_c_title)
    axes[2].set_xticks(sizes, [str(int(value)) for value in sizes])
    axes[2].minorticks_off()
    axes[2].legend(loc="best", fontsize=7.0)
    for axis in axes:
        axis.set_box_aspect(PANEL_BOX_ASPECT)
        finish_axes(axis)

    destination.mkdir(parents=True, exist_ok=False)
    stem = (
        "one_dimensional_hat_practical_noisy_three_method_overlay"
        if noisy_three_method_run is not None
        else (
            "one_dimensional_hat_practical_lasso_ols_strong_lasso_overlay"
            if strong_lasso_run is not None
            else "one_dimensional_hat_practical_lasso_ols_overlay"
        )
    )
    pdf = destination / f"{stem}.pdf"
    png = destination / f"{stem}.png"
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=220)
    plt.close(fig)
    metadata = {
        "batch": str(batch),
        "records_sha256": digest(batch / "records.json"),
        "summary_sha256": digest(batch / "summary.json"),
        "frozen_protocol_sha256": digest(batch / "frozen_protocol.json"),
        "representative_rollout_sha256": array_hash(rollout),
        "pair_policy": "P=50N ordered off-diagonal pairs sampled with replacement at each observation time",
        "repetitions": 100,
        "representative_M": size,
        "panel_a_quantity": "signed interaction K(z) against relative displacement z",
        "panel_a_summary": (
            "weak LASSO median and 10--90% pointwise band across 100 repetitions, with individual median fitted hat-function contributions"
            if noisy_three_method_run is not None else
            "weak LASSO mean and 10--90% pointwise band across 100 repetitions, with individual mean fitted hat-function contributions"
        ),
        "panel_ab_observation_noise": 0.01 if noisy_three_method_run is not None else 0.0,
        "panel_a_sparse_tail_start": support_q99,
        "panel_a_sparse_tail_definition": "upper 1% of the exact run-0 sampled training-pair stream",
        "panel_a_support_pair_count": support_pair_count,
        "panel_c_metric": "drift_error_rel",
        "panel_c_strong_fd_lasso": strong_source,
        "panel_c_noisy_three_method": noisy_source,
        "figure_pdf_sha256": digest(pdf),
        "figure_png_sha256": digest(png),
        "manuscript_target": "latex/L4DC/figures/one_dimensional_hat_practical.pdf",
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pdf)


def generate(
    batch,
    destination,
    representative_selection="fixed",
    display_particles=20,
    ols_batch=None,
    strong_lasso_run=None,
    noisy_three_method_run=None,
):
    batch, destination = Path(batch).resolve(), Path(destination).resolve()
    candidate = batch / "figure1_candidate/one_dimensional_hat_sampled_p50n_figure1_candidate.pdf"
    if candidate.is_file():
        generate_p50_publication(
            batch, destination, strong_lasso_run, noisy_three_method_run
        )
        return
    summary = read_json(batch / "summary/summary.json")
    frozen = read_json(batch / "frozen_protocol.json")
    ols_batch = None if ols_batch is None else Path(ols_batch).resolve()
    ols_summary = None if ols_batch is None else read_json(ols_batch / "summary/summary.json")
    representative = resolve_representative(batch, summary, representative_selection)
    report = representative["report"]
    directory = representative["directory"]
    model = representative["model"]
    M = representative["M"]
    test_spec = report.get("splits", {}).get("test", {})
    local_test_path = directory / "test/data.npz"
    reported_test_path = Path(test_spec.get("source_data", local_test_path))
    expected_test_hash = test_spec.get("source_data_sha256")
    if local_test_path.is_file() and (
        expected_test_hash is None or digest(local_test_path) == expected_test_hash
    ):
        test_path = local_test_path
    elif reported_test_path.is_file() and (
        expected_test_hash is None or digest(reported_test_path) == expected_test_hash
    ):
        test_path = reported_test_path
    else:
        raise FileNotFoundError(
            "representative test data are missing or fail their recorded hash: "
            f"{local_test_path} / {reported_test_path}"
        )
    test = load_data(test_path)
    rollout_path = directory / representative["name"] / "rollout.npz"
    with np.load(rollout_path) as saved:
        learned_trajectory = saved["X_hat"]
    if array_hash(learned_trajectory) != model["rollout_sha256"]:
        raise ValueError("representative rollout changed")
    theta = np.asarray(model["theta"], dtype=float)
    basis, _ = make_kernel_basis(model["basis"])
    truth, *_ = make_true_kernel(frozen["base"]["true_kernel"])

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55), constrained_layout=True)

    # (a) Radial interaction amplitude, with the fitted hat contributions visible.
    radius = np.linspace(0.0, 3.5, 601)
    vectors = basis(radius[:, None])[..., 0]
    components = vectors * theta[None, :]
    learned = components.sum(axis=1)
    exact = truth(radius[:, None])[:, 0]
    for index in range(theta.size):
        if abs(theta[index]) > 1e-8:
            axes[0].plot(radius, components[:, index], color=REFERENCE_COLOR,
                         linewidth=.45, alpha=.24)
    axes[0].plot(radius, exact, color=TRUE_COLOR, linewidth=1.5, label="true")
    axes[0].plot(radius, learned, "--", color=ESTIMATE_COLOR, linewidth=1.5,
                 label=rf"learned ($M={M}$)")
    axes[0].axhline(0, color=GRID_COLOR, linewidth=.6)
    axes[0].set(xlabel=r"pairwise distance $r$",
                ylabel=r"radial interaction $r\phi(r)$",
                title="(a) Learned interaction")
    axes[0].legend(loc="best", fontsize=7.5)

    # (b) Coupled test trajectories; all particles can be requested explicitly.
    true_trajectory = test["X"][0, :, :, 0]
    fitted_trajectory = learned_trajectory[0, :, :, 0]
    time = test["t"]
    particle_count = true_trajectory.shape[1]
    if display_particles == "all":
        chosen = np.arange(particle_count)
        linewidth, true_alpha, learned_alpha, rasterized = .18, .055, .07, True
        trajectory_title = rf"(b) All trajectories ($N={particle_count}$)"
    else:
        requested = int(display_particles)
        if not 1 <= requested <= particle_count:
            raise ValueError("display-particles must be 'all' or an integer from 1 to N")
        order = np.argsort(true_trajectory[0])
        chosen = order[np.linspace(0, particle_count - 1, requested, dtype=int)]
        linewidth, true_alpha, learned_alpha, rasterized = .42, .45, .55, False
        trajectory_title = "(b) Coupled trajectories"
    axes[1].plot(time, true_trajectory[:, chosen], color=TRUE_COLOR, linewidth=linewidth,
                 alpha=true_alpha, rasterized=rasterized)
    axes[1].plot(time, fitted_trajectory[:, chosen], "--", color=ESTIMATE_COLOR, linewidth=linewidth,
                 alpha=learned_alpha, rasterized=rasterized)
    axes[1].set(xlabel=r"time $t$", ylabel="particle position", title=trajectory_title)
    axes[1].legend(handles=[
        Line2D([0], [0], color=TRUE_COLOR, linewidth=1.0, label="true"),
        Line2D([0], [0], color=ESTIMATE_COLOR, linestyle="--", linewidth=1.0, label="learned"),
    ], loc="best", fontsize=7.5)

    # (c) Paired 100-repetition dictionary-size comparison.
    points = summary["dictionary_sizes"]
    rng = np.random.default_rng(970099)
    for point in points:
        M = point["value"]
        values = np.asarray(point["total100"]["metrics"]["drift_error_rel"]["values"], dtype=float)
        jitter = np.exp(rng.uniform(-.035, .035, size=len(values)))
        axes[2].scatter(M * jitter, values, s=5.5, alpha=.12, color=REFERENCE_COLOR, linewidths=0)
    sizes = np.asarray([point["value"] for point in points], dtype=float)
    stats = [point["total100"]["metrics"]["drift_error_rel"] for point in points]
    means = np.asarray([row["mean"] for row in stats])
    lower = np.asarray([row["mean_ci95"][0] for row in stats])
    upper = np.asarray([row["mean_ci95"][1] for row in stats])
    lasso_label = r"weak LASSO mean (95\% CI)" if ols_summary is not None else r"mean and 95\% CI"
    axes[2].errorbar(sizes, means, yerr=np.vstack([means - lower, upper - means]), fmt="^--",
                     color=ESTIMATE_COLOR, linewidth=1.45, markersize=4, capsize=2.5,
                     label=lasso_label)
    if ols_summary is not None:
        ols_points = ols_summary["dictionary_sizes"]
        ols_sizes = np.asarray([point["value"] for point in ols_points], dtype=float)
        if not np.array_equal(ols_sizes, sizes):
            raise ValueError("LASSO and OLS dictionary-size grids differ")
        ols_stats = [
            point["total100"]["metrics"]["drift_error_rel"]
            for point in ols_points
        ]
        ols_means = np.asarray([row["mean"] for row in ols_stats])
        ols_lower = np.asarray([row["mean_ci95"][0] for row in ols_stats])
        ols_upper = np.asarray([row["mean_ci95"][1] for row in ols_stats])
        if not np.all(np.isfinite(np.r_[ols_means, ols_lower, ols_upper])):
            raise ValueError("OLS drift-error summary contains nonfinite values")
        if not np.all((ols_lower > 0) & (ols_means >= ols_lower) & (ols_upper >= ols_means)):
            raise ValueError("OLS confidence intervals are incompatible with the log axis")
        axes[2].errorbar(
            ols_sizes,
            ols_means,
            yerr=np.vstack([ols_means - ols_lower, ols_upper - ols_means]),
            fmt="o-",
            color=TRUE_COLOR,
            linewidth=1.25,
            markersize=3.8,
            capsize=2.5,
            label=r"weak OLS mean (95\% CI)",
        )
    axes[2].set(xscale="log", yscale="log", xlabel=r"dictionary size $M$",
                ylabel="drift error", title="(c) Dictionary size")
    axes[2].set_xticks(sizes, [str(int(value)) for value in sizes])
    axes[2].minorticks_off()
    axes[2].legend(loc="best", fontsize=7.2)

    for ax in axes:
        ax.set_box_aspect(PANEL_BOX_ASPECT)
        ax.grid(alpha=.16)
        finish_axes(ax)
    destination.mkdir(parents=True, exist_ok=False)
    stem = (
        "one_dimensional_hat_practical_lasso_ols_overlay"
        if ols_summary is not None
        else "one_dimensional_hat_practical_three_panel"
    )
    pdf = destination / f"{stem}.pdf"
    png = destination / f"{stem}.png"
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=220)
    plt.close(fig)
    shutil.copy2(Path(__file__), destination / "generate_figure.py")
    metadata = {
        "batch": str(batch), "summary_sha256": digest(batch / "summary/summary.json"),
        "representative_M": int(model["basis"]["M"]),
        "representative_policy": representative["policy"],
        "representative_selected_after_aggregate_results": representative["post_hoc"],
        "representative_source_report": str(representative["report_path"]),
        "representative_source_report_sha256": representative["report_sha256"],
        "representative_metrics": model["metrics"],
        "underlying_particle_count": int(true_trajectory.shape[1]),
        "displayed_particle_count": int(len(chosen)),
        "panel_a_quantity": "radial interaction r phi(r)",
        "panel_c_metric": "drift_error_rel",
        "figure_pdf_sha256": digest(pdf), "figure_png_sha256": digest(png),
        "ols_overlay": None if ols_summary is None else {
            "batch": str(ols_batch),
            "summary_sha256": digest(ols_batch / "summary/summary.json"),
            "statistic": "100-repetition arithmetic mean and percentile-bootstrap 95% CI",
            "individual_points_shown": False,
            "total_divergent_rollouts_retained_in_source": int(sum(
                point["total100"]["diagnostics"]["divergent_rollouts"]
                for point in ols_summary["dictionary_sizes"]
            )),
        },
        "manuscript_edited": False,
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pdf)


if __name__ == "__main__":
    apply_manuscript_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--representative", choices=("fixed", "aggregate-force-minimum"), default="fixed")
    parser.add_argument("--display-particles", default="20",
                        help="integer count selected by initial rank, or 'all'")
    parser.add_argument("--ols-batch", type=Path,
                        help="optional matching OLS batch whose mean and 95% CI are overlaid in panel (c)")
    parser.add_argument("--strong-lasso-run", type=Path,
                        help="optional validation-selected strong FD--LASSO run for panel (c)")
    parser.add_argument("--noisy-three-method-run", type=Path,
                        help="optional sigma_obs=0.01 weak LASSO, weak OLS, and strong FD--LASSO run for panel (c)")
    args = parser.parse_args()
    display = args.display_particles if args.display_particles == "all" else int(args.display_particles)
    generate(
        args.batch, args.destination, args.representative, display,
        args.ols_batch, args.strong_lasso_run, args.noisy_three_method_run,
    )
