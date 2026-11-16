"""Resume aggregation after an external harness timeout using the frozen runner snapshot."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Preload the hash-matched live dependencies because the compact source
# snapshot intentionally contains only files listed in the frozen manifest.
import features  # noqa: F401
import kernels  # noqa: F401
import pipeline  # noqa: F401
import simulate  # noqa: F401
import scripts.r05_numerical_settings  # noqa: F401
import scripts.run_1d_hat_lasso  # noqa: F401
import scripts.run_full_pair_scaling  # noqa: F401
import scripts.run_random_pair_validation  # noqa: F401
import scripts.run_sindy_comparison  # noqa: F401


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_runner(batch):
    path = batch / "source/scripts/run_qanr_matched_lasso_sweeps.py"
    spec = importlib.util.spec_from_file_location("frozen_qanr_lasso_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def resume(batch):
    batch = Path(batch).resolve()
    report_path = batch / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("status") != "running":
        raise ValueError(f"batch status is not running: {report.get('status')}")
    module, frozen_runner = load_frozen_runner(batch)
    protocol = json.loads((batch / "frozen_protocol.json").read_text())
    module.validate_protocol(protocol)
    expected = {
        "observation": protocol["observation_noise"]["repetitions"],
        "stochastic": (
            len(protocol["process_noise"]["values"])
            * protocol["process_noise"]["repetitions"]
            + len(protocol["particle_number"]["values"])
            * protocol["particle_number"]["repetitions"]
        ),
        "time_step": len(protocol["time_step"]["values"]),
    }
    jobs = {}
    for kind, count in expected.items():
        paths = [batch / "jobs" / kind / f"job-{index:04d}.json" for index in range(count)]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise ValueError(f"missing completed jobs for {kind}: {missing[:3]}")
        jobs[kind] = [json.loads(path.read_text()) for path in paths]

    started = perf_counter()
    pair_path = batch / "jobs/pair/pair.json"
    if pair_path.exists():
        pair_result = json.loads(pair_path.read_text())
    else:
        pair_result = module.run_pair_sweep(protocol)
        module.write_json(pair_path, pair_result)

    observation = [row for result in jobs["observation"] for row in result["rows"]]
    stochastic = jobs["stochastic"]
    dt_rows = jobs["time_step"]
    pair_rows = pair_result["rows"]
    module.write_csv(batch / "observation_lasso.csv", observation)
    module.write_csv(batch / "stochastic_lasso.csv", stochastic)
    module.write_csv(batch / "time_step_lasso.csv", dt_rows)
    module.write_csv(batch / "pair_lasso.csv", pair_rows)
    module.write_json(batch / "pair_full_selection.json", {
        "selection": pair_result["selection"], "theta_full": pair_result["theta_full"]
    })
    summary = module.summarize(
        observation, stochastic, dt_rows, pair_rows, pair_result["selection"]
    )
    module.write_json(batch / "summary.json", summary)
    verification = module.verify_against_archives(
        protocol, observation, stochastic, dt_rows, pair_rows, False
    )
    module.write_json(batch / "verification.json", verification)

    report.update({
        "status": "completed",
        "completion_mode": "resumed_after_harness_timeout",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "resume_elapsed_seconds": perf_counter() - started,
        "harness_timeout_seconds": 7200,
        "resume_used_frozen_runner_snapshot": True,
        "frozen_runner_sha256_used_for_resume": digest(frozen_runner),
        "resume_script_sha256": digest(Path(__file__)),
        "counts": {
            "observation_models": len(observation),
            "stochastic_models": len(stochastic),
            "time_step_models": len(dt_rows),
            "pair_models": len(pair_rows),
        },
        "summary_sha256": digest(batch / "summary.json"),
        "verification_sha256": digest(batch / "verification.json"),
    })
    module.write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    print(json.dumps(resume(args.batch), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
