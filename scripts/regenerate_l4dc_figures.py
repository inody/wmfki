"""Regenerate the eight frozen L4DC numerical figure PDFs.

Run from the repository root with::

    .venv/bin/python scripts/regenerate_l4dc_figures.py

All figures are first built in a temporary staging directory. Source-artifact and
output hashes must match the frozen protocol before the reference figures are
replaced, so a failed or partial run cannot leave a mixed figure set.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiments/l4dc_current_figures.json"
EXPECTED_OUTPUTS = {
    "one_dimensional_hat_practical.pdf",
    "validation_sweeps.pdf",
    "predator_prey_profiles.pdf",
    "predator_prey_dynamics.pdf",
    "predator_prey_robustness_sweeps.pdf",
    "localized10d_recovery.pdf",
    "localized10d_coupled_dynamics.pdf",
    "localized10d_robustness_sweeps.pdf",
}
CURRENT_RENDERING_POLICY = {
    "figure_text_font": "Times_via_LaTeX_times_package",
    "figure_math_font": "Computer_Modern",
    "figure1_observation_noise": 0.01,
    "figure1_panel_a": "M14_weak_lasso_median_with_10_90_band",
    "figure1_panel_c_metric": "drift_error",
    "figure1_panel_c_title": "dictionary_size",
    "figure2_panel_a_comparison": "weak_lasso_weak_ols_strong_fd_lasso",
    "nominal_overlay_comparison": "weak_lasso_weak_ols_strong_fd_lasso",
    "figure4_layout": "rows_true_weak_lasso_weak_ols_strong_fd_lasso_columns_t0_t1_t2_t3",
}


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def validate_config(spec: dict) -> None:
    protocol = (spec.get("task"), spec.get("protocol_version"))
    if protocol not in {
        ("regenerate_l4dc_current_figures", 1),
        ("reproduce_l4dc_figures_from_fresh_runs", 2),
    }:
        raise ValueError("invalid L4DC figure-regeneration protocol")
    outputs = set(spec.get("expected_figure_sha256", {}))
    if outputs != EXPECTED_OUTPUTS:
        raise ValueError(
            f"current figure set changed: expected {sorted(EXPECTED_OUTPUTS)}, "
            f"found {sorted(outputs)}"
        )
    if set(spec.get("sources", {})) != {
        "hat", "hat_noisy_comparison", "redundant6", "redundant6_time_step",
        "runtime_scaling", "qanr_strong_lasso_noise", "predator_sweeps",
        "predator_strong_lasso_noise", "localized10d", "regularized_baselines",
    }:
        raise ValueError("source-run keys changed")
    expected_hashes = spec.get("expected_figure_sha256", {})
    if protocol == ("regenerate_l4dc_current_figures", 1):
        if any(value is None for value in expected_hashes.values()):
            raise ValueError("the frozen numerical protocol requires pinned figure hashes")
        if spec.get("rendering_policy") != CURRENT_RENDERING_POLICY:
            raise ValueError("current Figure 1 / Figure 2(d) rendering policy changed")
        sources = spec["sources"]
        if sources["redundant6_time_step"] != sources["redundant6"]:
            raise ValueError(
                "current Figure 2(d) uses the deterministic production-bundle summary"
            )
        expected_runtime = str(Path(sources["redundant6"]) / "summary_f.json")
        if sources["runtime_scaling"] != expected_runtime:
            raise ValueError("current Figure 2(f) requires the production summary_f.json")


def configured_directories(spec: dict) -> tuple[Path, Path]:
    figure_value = spec.get("figure_directory")
    if not isinstance(figure_value, str) or not figure_value:
        raise ValueError("figure_directory must be a nonempty repository-relative path")
    build_value = spec.get("build_directory")
    if build_value is None:
        build_value = str(Path(figure_value).parent / "build")
    if not isinstance(build_value, str) or not build_value:
        raise ValueError("build_directory must be a nonempty repository-relative path")

    directories = []
    for label, value in (
        ("figure_directory", figure_value),
        ("build_directory", build_value),
    ):
        path = (ROOT / value).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError as error:
            raise ValueError(f"{label} must stay below {ROOT}: {value}") from error
        directories.append(path)
    return directories[0], directories[1]


def verify_required_artifacts(spec: dict) -> dict[str, str]:
    failures = []
    actual = {}
    for relative, expected in spec["required_artifact_sha256"].items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        value = digest(path)
        actual[relative] = value
        if value != expected:
            failures.append(f"hash mismatch: {relative}: {value} != {expected}")
    for key, relative in spec["sources"].items():
        path = ROOT / relative
        if key == "runtime_scaling":
            if not path.is_file():
                failures.append(f"missing runtime source: {relative}")
        elif not path.is_dir():
            failures.append(f"missing source run: {relative}")
    if failures:
        raise RuntimeError(
            "frozen figure inputs are unavailable or changed:\n  "
            + "\n  ".join(failures)
        )
    return actual


def command_plan(spec: dict, stage: Path) -> list[tuple[str, list[str]]]:
    py = sys.executable
    source = {key: str(ROOT / value) for key, value in spec["sources"].items()}
    return [
        (
            "hat",
            [
                py, str(ROOT / "scripts/generate_1d_hat_practical_figure.py"),
                source["hat"], str(stage / "hat"),
                "--representative", "aggregate-force-minimum",
                "--display-particles", "all",
                "--ols-batch", source["hat"],
                "--noisy-three-method-run", source["hat_noisy_comparison"],
            ],
        ),
        (
            "redundant6",
            [
                py, str(ROOT / "scripts/generate_qanr_matched_lasso_overlay.py"),
                source["redundant6"], str(stage / "redundant6"),
                "--time-step-run", source["redundant6_time_step"],
                "--runtime-csv", source["runtime_scaling"],
                "--strong-lasso-noise-run", source["qanr_strong_lasso_noise"],
            ],
        ),
        (
            "nominal_overlays",
            [
                py, str(ROOT / "scripts/generate_nominal_baseline_overlays.py"),
                "--regularized-run", source["regularized_baselines"],
                "--predator-run", source["predator_sweeps"],
                "--localized-run", source["localized10d"],
                "--output", str(stage / "nominal"),
            ],
        ),
        (
            "predator_robustness",
            [
                py, str(ROOT / "scripts/generate_multidimensional_sweep_figures.py"),
                "--predator-dir", source["predator_sweeps"],
                "--predator-strong-dir", source["predator_strong_lasso_noise"],
                "--morse-dir", source["localized10d"],
                "--output-dir", str(stage / "multidimensional"),
            ],
        ),
        (
            "localized10d",
            [
                py, str(ROOT / "scripts/generate_localized10d_figures.py"),
                source["localized10d"],
                "--output", str(stage / "localized10d"),
            ],
        ),
    ]


def staged_outputs(stage: Path) -> dict[str, Path]:
    return {
        "one_dimensional_hat_practical.pdf": (
            stage / "hat/one_dimensional_hat_practical_noisy_three_method_overlay.pdf"
        ),
        "validation_sweeps.pdf": (
            stage / "redundant6/validation_sweeps_lasso_ols_overlay.pdf"
        ),
        "predator_prey_profiles.pdf": stage / "nominal/predator_prey_profiles.pdf",
        "predator_prey_dynamics.pdf": stage / "nominal/predator_prey_dynamics.pdf",
        "predator_prey_robustness_sweeps.pdf": (
            stage / "multidimensional/predator_prey_robustness_sweeps.pdf"
        ),
        "localized10d_recovery.pdf": stage / "nominal/localized10d_recovery.pdf",
        "localized10d_coupled_dynamics.pdf": (
            stage / "nominal/localized10d_coupled_dynamics.pdf"
        ),
        "localized10d_robustness_sweeps.pdf": (
            stage / "localized10d/localized10d_robustness_sweeps.pdf"
        ),
    }


def verify_generated(outputs: dict[str, Path], expected: dict[str, str | None]) -> dict[str, str]:
    failures = []
    actual = {}
    for name, path in outputs.items():
        if not path.is_file():
            failures.append(f"missing generated figure: {name} ({path})")
            continue
        value = digest(path)
        actual[name] = value
        if expected[name] is not None and value != expected[name]:
            failures.append(f"figure hash mismatch: {name}: {value} != {expected[name]}")
    if failures:
        raise RuntimeError(
            "generated figures differ from the frozen numerical artifacts:\n  "
            + "\n  ".join(failures)
        )
    return actual


def verify_checked_in(destination: Path, expected: dict[str, str]) -> dict[str, str]:
    failures = []
    actual = {}
    for name, value in expected.items():
        path = destination / name
        if not path.is_file():
            failures.append(f"missing checked-in figure: {name}")
            continue
        actual[name] = digest(path)
        if actual[name] != value:
            failures.append(
                f"checked-in figure hash mismatch: {name}: {actual[name]} != {value}"
            )
    if failures:
        raise RuntimeError(
            "reference figures differ from the frozen numerical protocol:\n  "
            + "\n  ".join(failures)
        )
    return actual


def install_figures(outputs: dict[str, Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    temporary = []
    try:
        for name, source in outputs.items():
            target = destination / name
            staged_target = destination / f".{name}.regenerating"
            shutil.copy2(source, staged_target)
            temporary.append((staged_target, target))
        for staged_target, target in temporary:
            staged_target.replace(target)
    finally:
        for staged_target, _ in temporary:
            staged_target.unlink(missing_ok=True)


def relative_command(command: list[str]) -> list[str]:
    result = []
    for token in command:
        try:
            result.append(str(Path(token).resolve().relative_to(ROOT)))
        except (ValueError, OSError):
            result.append(token)
    return result


def regenerate(
    config_path: Path, *, check_only: bool = False, refresh_hashes: bool = False
) -> dict:
    config_path = config_path.resolve()
    spec = read_json(config_path)
    validate_config(spec)
    input_hashes = verify_required_artifacts(spec)
    figure_directory, build = configured_directories(spec)
    build.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["SOURCE_DATE_EPOCH"] = "0"

    executed = []
    with tempfile.TemporaryDirectory(prefix="current-figures-", dir=build) as temporary:
        stage = Path(temporary)
        for label, command in command_plan(spec, stage):
            print(f"[figures] {label}", flush=True)
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
            executed.append({"label": label, "command": relative_command(command)})
        outputs = staged_outputs(stage)
        expected = (
            {name: None for name in outputs}
            if refresh_hashes else spec["expected_figure_sha256"]
        )
        generated_hashes = verify_generated(outputs, expected)
        if check_only:
            installed = verify_checked_in(
                figure_directory, spec["expected_figure_sha256"]
            )
        else:
            install_figures(outputs, figure_directory)
            installed = {name: digest(figure_directory / name) for name in outputs}
            if installed != generated_hashes:
                raise RuntimeError("installed figures differ from staged figures")

    result = {
        "task": spec["task"],
        "status": (
            "verified" if check_only else
            "hashes_refreshed" if refresh_hashes else
            "regenerated"
        ),
        "check_only": check_only,
        "refresh_hashes": refresh_hashes,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": digest(config_path),
        "python": sys.executable,
        "source_date_epoch": environment["SOURCE_DATE_EPOCH"],
        "rendering_policy": spec.get("rendering_policy"),
        "commands": executed,
        "required_artifact_sha256": input_hashes,
        "figure_sha256": generated_hashes,
        "checked_in_figure_sha256": installed,
    }
    manifest = build / "current_figure_regeneration.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["manifest"] = str(manifest.relative_to(ROOT))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true",
        help="regenerate in staging and verify hashes without replacing reference figures",
    )
    mode.add_argument(
        "--refresh-hashes", action="store_true",
        help="install an intentional rendering update and report its new figure hashes",
    )
    args = parser.parse_args()
    result = regenerate(
        args.config, check_only=args.check, refresh_hashes=args.refresh_hashes
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
