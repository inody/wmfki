"""Small synthetic fixtures for public protocol unit tests.

These fixtures exercise protocol, seed, sampler, and failure-handling behavior.
They do not claim parity with private historical output archives.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def read_config(name: str) -> dict:
    return json.loads((ROOT / "configs/experiments" / name).read_text())


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_temporary_directory(test_case) -> Path:
    temporary = tempfile.TemporaryDirectory(dir=ROOT)
    test_case.addCleanup(temporary.cleanup)
    return Path(temporary.name)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def synthetic_hat_sweep_frozen() -> dict:
    return {
        "base": read_config("one_dimensional_hat_nested_20.json"),
        "protocol": read_config("one_dimensional_hat_sweeps.json"),
    }


def synthetic_dictionary_grid_frozen(reference_count: int = 100) -> dict:
    base = read_config("one_dimensional_hat_nested_20.json")
    base["basis"]["sizes"] = [9, 11, 14, 17, 22, 27, 33, 42, 52, 65]
    base["replication"]["count"] = reference_count
    references = {
        f"run-{index}": {
            "directory": f"synthetic-reference/run-{index + 1:03d}",
            "report_sha256": f"synthetic-{index:03d}",
        }
        for index in range(reference_count)
    }
    return {
        "base": base,
        "requested_protocol": read_config("one_dimensional_hat_dictionary_grid_R100.json"),
        "references": references,
    }


def materialize_matched_dictionary_sources(test_case, reference_count: int = 100):
    root = repository_temporary_directory(test_case)
    frozen = synthetic_dictionary_grid_frozen(reference_count)
    source = {}
    hashes = {}
    for method in ("lasso", "ols"):
        batch = root / method
        frozen_path = write_json(batch / "frozen_protocol.json", frozen)
        status = completed_status(frozen_path, reference_count, job_prefix="run")
        verification = {"status": "passed"}
        write_json(batch / "batch_status.json", status)
        write_json(batch / "verification/verification.json", verification)
        source.update({
            method: batch,
            f"{method}_frozen": deepcopy(frozen),
            f"{method}_status": status,
            f"{method}_verification": verification,
        })
        hashes[method] = digest(frozen_path)
    return source, hashes


def synthetic_hat_r100_frozen() -> dict:
    source = synthetic_hat_sweep_frozen()
    base = deepcopy(source["base"])
    count = 80
    base["basis"]["sizes"] = [17]
    base["replication"]["count"] = count
    base["seeds"] = {
        "train": [8_000_000 + 10 * index for index in range(count)],
        "validation": [8_000_002 + 10 * index for index in range(count)],
        "test": [8_000_004 + 10 * index for index in range(count)],
    }
    protocol = deepcopy(source["protocol"])
    protocol["observation_seed_offset"] = 10_000_000
    return {
        "base": base,
        "protocol": protocol,
        "requested_protocol": read_config("one_dimensional_hat_M17_R100.json"),
    }


def completed_status(frozen_path: Path, result_count: int, *, job_prefix: str = "job") -> dict:
    jobs = {f"{job_prefix}-{index}": {} for index in range(result_count)}
    return {
        "status": "completed",
        "jobs": jobs,
        "results": {key: {} for key in jobs},
        "frozen_protocol_sha256": digest(frozen_path),
    }


def synthetic_localized_frozen(observation_noise: float = 0.01) -> dict:
    base = read_config("localized10d_scientific.json")
    observation = read_config(
        "localized10d_observation_noise_001.json"
        if observation_noise == 0.01
        else "localized10d_observation_noise_003.json"
    )
    frozen = deepcopy(base)
    frozen.update({
        "task": "synthetic_localized10d_fixed_noise_source",
        "protocol_version": 1,
        "observation_noise": observation_noise,
        "observation_noise_seeds": deepcopy(observation["observation_noise_seeds"]),
        "spatial_test_construction": observation["spatial_test_construction"],
        "pair_sampling": {
            "baseline_pairs": 10000,
            "mode": "nested_resample_each_time",
            "with_replacement": True,
            "exclude_diagonal": True,
        },
        "pair_penalty_policy": (
            "absolute_alpha_selected_at_Pmax_by_independent_sampled_validation_"
            "then_fixed_across_all_P_within_repetition"
        ),
    })
    frozen["evaluation"]["rollout_particles"] = frozen["model"]["particles"]
    return frozen
