"""Run the three training maps as independent, serial, payload-free mapping trials."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import mkdtemp
import traceback

from .optimization.evaluation.resources import run_limited


REPO = Path(__file__).resolve().parents[4]
TOOLS = Path(__file__).resolve().parent
OUTPUT = REPO / "output/sf13"
DATA = REPO / "output/optimization/datasets/dataset_001"
WORKER = "swarm_rescue.solutions.devtools.payload_free_mapping_worker"
SEEDS = (2026111701, 2026111702)
MEMORY_BYTES = 4_000_000_000


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_plan(smoke):
    train = {r["id"] for r in read(DATA / "train.json")["maps"]}
    catalog = {r["name"]: r for r in read(DATA / "index.json")["maps"]}
    files = {Path(__file__), TOOLS / "payload_free_mapper_trial.py",
             TOOLS / "payload_free_mapping_worker.py", TOOLS / "windows_pressure.py",
             TOOLS / "optimization/evaluation/resources.py", TOOLS / "evaluate_red.py",
             REPO / "src/swarm_rescue/tools/map_exploration_scoring.py"}
    files.update((REPO / "src/swarm_rescue/simulation").rglob("*.py"))
    schedule = []
    for n in (72, 24, 37):
        row = catalog[f"yh_map_fs_{n}"]
        if row["id"] not in train:
            raise ValueError(f"Map{n} is not in the training split")
        for name, expected in row["files"].items():
            file = REPO / name
            if digest(file) != expected:
                raise ValueError(f"Training file differs from the dataset: {name}")
            files.add(file)
        module = ("swarm_rescue.solutions.devtools.optimization.generated_maps."
                  f"g_{row['geometry_sha256']}.map_generated")
        for number, seed in enumerate(SEEDS, 1):
            if smoke and (n, number) != (72, 1):
                continue
            schedule.append(dict(map=f"Map{n}", round=number, seed=seed, module=module,
                                 run_dir=f"Map{n}_r{number}", map_file=row["map_path"],
                                 size=row["size"], limit_steps=12 if smoke else 5400,
                                 limit_seconds=180))
    return dict(smoke=smoke, controller="PayloadFreeMapperTrial", drones=5, payload=0,
                memory_bytes=MEMORY_BYTES, cpu_count=1, simultaneous_flights=1,
                outer_timeout_seconds=240, zones=["NO_COM_ZONE", "NO_GPS_ZONE"],
                git_head=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                                 text=True).strip(),
                source_sha256={f.relative_to(REPO).as_posix(): digest(f) for f in sorted(files)},
                schedule=schedule)


def changed_sources(plan):
    return [p for p, expected in plan["source_sha256"].items()
            if not (REPO / p).is_file() or digest(REPO / p) != expected]


def validate(run, item, resources):
    import numpy as np

    issues = []
    if resources["status"] != "exited" or resources["exit_code"] != 0:
        issues.append(f"Worker {resources['status']} exit={resources['exit_code']}")
    log = (run / "run.log").read_text(encoding="utf-8", errors="replace")
    if any(marker in log for marker in ("Exception ignored", "Traceback (most recent call last)",
                                         "Fatal Python error")):
        issues.append("Worker or native callback exception in run.log")
    result = read(run / "result.json")
    if any(result.get(key) != item[key] for key in ("map", "round", "seed")):
        issues.append("Map, round, or seed mismatch")
    if result.get("status") != "ok" or result.get("bombs") != 0:
        issues.append("Unsuccessful or non-empty flight")
    if result.get("steps") != item["limit_steps"] or result.get("stop_reason") != "step_limit":
        issues.append("Did not reach the designated step limit")
    for key in ("coverage_percent", "map_score"):
        value = result.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            issues.append(f"Missing or invalid {key}")
    with np.load(run / "trial.npz", allow_pickle=False) as saved:
        truth, pred, positions = (saved[key] for key in ("truth", "pred", "positions"))
        if (truth.shape != tuple(reversed(item["size"])) or pred.shape != truth.shape
                or not truth.any() or not np.isin(truth, [0, 1]).all()
                or not np.isin(pred, [0, 1]).all()):
            issues.append("Invalid map arrays")
        if (positions.shape != (result.get("steps"), 5, 2)
                or not np.isfinite(positions[0]).all()
                or np.isinf(positions).any()
                or (np.isnan(positions[:, :, 0]) != np.isnan(positions[:, :, 1])).any()):
            issues.append("Invalid trajectory frames")
    return issues


def execute(plan, output):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONHASHSEED="0", G3_MEMORY_BYTES=str(MEMORY_BYTES))
    outcomes = []
    for item in plan["schedule"]:
        run = output / item["run_dir"]
        run.mkdir()
        write(run / "request.json", item)
        issues = []
        drift = changed_sources(plan)
        if drift:
            issues.append(f"Source drift before flight: {drift}")
        else:
            try:
                print(f"START {item['run_dir']}", flush=True)
                command = [sys.executable, "-B", "-u", "-m", WORKER, str(run)]
                with (run / "run.log").open("xb") as log:
                    resources = run_limited(command, env, log, plan["outer_timeout_seconds"],
                                            memory_bytes=MEMORY_BYTES)
                write(run / "resources.json", resources)
                issues.extend(validate(run, item, resources))
            except (OSError, ValueError, RuntimeError, KeyError, IndexError) as error:
                issues.append(f"{type(error).__name__}: {error}")
                write(run / "exception.json", dict(traceback=traceback.format_exc()))
            drift = changed_sources(plan)
            if drift:
                issues.append(f"Source drift during flight: {drift}")
        verdict = dict(valid=not issues, reasons=issues)
        write(run / "validity.json", verdict)
        write(run / "integrity.json", {p.name: digest(p) for p in sorted(run.iterdir()) if p.is_file()})
        outcomes.append(dict(map=item["map"], round=item["round"], **verdict))
        print(f"DONE {item['run_dir']} {verdict}", flush=True)
        if drift or (not (run / "trial.npz").exists() and len(outcomes) == 1):
            break
    summary = dict(rounds=outcomes, expected=len(plan["schedule"]),
                   valid=sum(row["valid"] for row in outcomes))
    write(output / "summary.json", summary)
    return 0 if summary["valid"] == summary["expected"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run one 12-step wiring check")
    args = parser.parse_args()
    plan = make_plan(args.smoke)
    output = Path(mkdtemp(prefix="payload_free_smoke_" if args.smoke else "payload_free_six_", dir=OUTPUT))
    print(str(output.relative_to(REPO)), flush=True)
    write(output / "plan.json", plan)
    return execute(plan, output)


if __name__ == "__main__":
    sys.exit(main())
