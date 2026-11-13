r"""Run the current red entry on isolated rounds; never change engine/config files.

From the repository root:
    .venv\Scripts\python.exe -B -m swarm_rescue.solutions.devtools.evaluate_red
Outputs are local development artifacts, not competition submissions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime

ROOT = Path(__file__).resolve().parents[4]
MAPS = ("Map01", "Map02", "Map03", "Map04", "Map05", "Map07")
METRICS = ("score_map", "score_wall", "score_time", "round_score")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def evaluation_config(map_name, drones=5, bombs=2, recording=True, zones=None):
    if map_name not in MAPS or drones < 1 or bombs < 1:
        raise ValueError("Unsupported map or nonpositive team budget")
    return {
        "stat_saving_enabled": False,
        "state_recording_enabled": recording,
        "video_capture_enabled": False,
        "evaluation_plan": [{"map_name": map_name, "team_mode": "place",
                             "nb_rounds": 1, "config_weight": 1,
                             "number_drones": drones,
                             "initial_bombs_per_drone": bombs,
                             "zones_config": list(zones or [])}],
    }


def summarize(rows):
    """Keep failed/missing rounds visible; do not silently turn them into zero."""
    result = {}
    for name in dict.fromkeys(row["map"] for row in rows):
        selected = [row for row in rows if row["map"] == name]
        scored = [row for row in selected if row.get("phase1") is not None]
        item = {
            "attempted": len(selected), "scored": len(scored),
            "unscored": len(selected) - len(scored),
            "run_failures": sum(row["status"] != "ok" for row in selected),
            "submitted": sum(bool(row.get("submitted")) for row in selected),
            "valid_shape": sum(bool(row.get("valid_shape")) for row in selected),
            "all_bombs_placed": sum(row.get("placed_count") == row.get("expected_count", -1)
                                    for row in selected),
            "walltime_limited": sum(bool(row.get("walltime_limited")) for row in selected),
        }
        for metric in METRICS:
            values = [row["phase1"][metric] for row in scored]
            item[metric] = ({"mean": statistics.mean(values), "min": min(values),
                             "max": max(values), "stdev": statistics.stdev(values)
                             if len(values) > 1 else 0.0} if values else None)
        result[name] = item
    return result


def configure_ray_backend(backend):
    """Select the existing engine CPU implementation in this evaluation process."""
    if backend != "cpu":
        return
    from swarm_rescue.simulation.ray_sensors.ray_compute import RayCompute
    if getattr(RayCompute, "_red_cpu_selected", False):
        return
    original_init = RayCompute.__init__

    def cpu_ray_init(self, *args, **kwargs):
        kwargs["use_shader"] = False
        original_init(self, *args, **kwargs)

    RayCompute.__init__ = cpu_ray_init
    RayCompute._red_cpu_selected = True


def run_worker(request_path):
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    run_dir = Path(request["run_dir"])
    result = dict(map=request["map"], round=request["round"], seed=request["seed"],
                  status="error", phase1=None)
    started = time.monotonic()
    try:
        os.environ["SWARM_RESCUE_RESULT_PATH"] = request["engine_dir"]
        sys.path.insert(0, str(ROOT / "src"))
        import numpy as np
        import yaml
        from dataclasses import asdict
        from swarm_rescue.launcher import Launcher
        from swarm_rescue.simulation.reporting.bombs_io import exploration_map_output_path

        # Development-only selection of the engine's existing CPU ray backend.
        # Both versions must use the same backend for a fair comparison.
        configure_ray_backend(request.get("ray_backend", "engine"))
        if request.get("visual"):
            from swarm_rescue.solutions.devtools.replay_window import configure_replay_window
            configure_replay_window()
        if request.get("strategy", "current") != "current":
            import swarm_rescue.launcher as launcher_module
            from swarm_rescue.solutions.devtools.evaluate_verified_revision import controller_identity, load_controller
            selected = load_controller(controller_identity(request["strategy"]))
            entry = getattr(launcher_module, "drone_entry", launcher_module)
            original_selector = entry.drone_class_for_mode
            entry.drone_class_for_mode = lambda mode: selected if mode == "place" else original_selector(mode)

        random.seed(request["seed"])
        np.random.seed(request["seed"])
        config = evaluation_config(request["map"], request["drones"],
                                   request["bombs"], request["recording"],request.get('zones'))
        # The checked-in config is never edited. Keep a JSON copy for audit.
        write_json(run_dir / "conditions.json", config)
        with tempfile.TemporaryDirectory(prefix="red-eval-") as temporary:
            config_path = Path(temporary) / "eval.yml"
            config_path.write_text(yaml.safe_dump(config), encoding="ascii")
            launcher = Launcher(config_path=str(config_path))
            if not launcher.eval_plan_ok:
                raise RuntimeError("Launcher rejected evaluation plan")
            scenario = launcher.eval_plan.list_eval_config[0]
            data = launcher.one_round(scenario, request["round"],
                                      headless=not request.get("visual", False), competition_mode=True)
        if data is None or data[14] is None:
            raise RuntimeError("Launcher returned no red phase1 score")
        submission = Path(exploration_map_output_path(
            launcher.result_path, launcher.team_info, scenario, request["round"]))
        expected_shape = [int(launcher.size_area[1]), int(launcher.size_area[0])]
        submitted = submission.is_file()
        shape = None
        valid_shape = False
        if submitted:
            grid = json.loads(submission.read_text(encoding="utf-8"))["grid"]
            shape = [len(grid), len(grid[0]) if grid else 0]
            valid_shape = shape == expected_shape and all(len(row) == shape[1] for row in grid)
        result.update(
            status="controller_crash" if data[9] else "ok",
            elapsed_steps=int(data[2]), elapsed_walltime=float(data[7]),
            walltime_limited=bool(data[8]), controller_crashed=bool(data[9]),
            max_steps=int(launcher.max_timestep_limit),
            max_walltime=float(launcher.max_walltime_limit),
            placed_count=int(data[11]), expected_count=request["drones"] * request["bombs"],
            remaining_inventory=int(data[13]),
            percent_destroyed=float(data[0]), mean_health=float(data[1]),
            trajectory_exploration=float(data[4]),
            submitted=submitted, submission_shape=shape, expected_shape=expected_shape,
            valid_shape=valid_shape, phase1=asdict(data[14]),
            comparison_images=len(list(Path(launcher.result_path).glob("*_exploration_diff.png"))),
            wall_images=len(list(Path(launcher.result_path).glob("*_wall_destruction.png"))),
        )
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    result["process_seconds"] = time.monotonic() - started
    write_json(run_dir / "result.json", result)
    return 0 if result["status"] == "ok" else 1


def environment_manifest(args):
    packages = {}
    for package in ("swarm-rescue", "arcade", "pymunk", "numpy", "opencv-contrib-python",
                    "PyYAML", "cerberus", "scipy", "pytest"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    files = sorted((ROOT / "src").rglob("*.py"))
    files += sorted((ROOT / "src" / "swarm_rescue" / "solutions").glob("*.yml"))
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": packages, "arguments": vars(args),
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in files},
        "score_scope": "evaluated contribution: 0.5*map + 0.2*wall + 0.1*time; difficulty pending; final total out of 100 pending",
        "seed_scope": "Python random and NumPy legacy RNG; engine RNG/real time may vary",
    }


def save_reports(output, rows):
    write_json(output / "results.json", rows)
    summary = summarize(rows)
    write_json(output / "summary.json", summary)
    fields = ["map", "round", "seed", "status", "submitted", "valid_shape", "placed_count",
              "expected_count", "elapsed_steps", "elapsed_walltime", "walltime_limited", *METRICS]
    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fields}
            flat.update({key: (row.get("phase1") or {}).get(key) for key in METRICS})
            writer.writerow(flat)
    lines = ["# 红队评测结果", "", "各分项满分 100，最终总分满分 100。已评测部分 = 地图分×50% + 墙体分×20% + 时间分×10%。参考蓝队相关分待评，最终总分待定；不对已评测部分重新归一化。",
             "", "| 地图 | 有成绩/尝试 | 提交 | 放满 | 异常 | 地图均分 / 100 | 墙体均分 / 100 | 时间均分 / 100 | 已评测部分均分 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, item in summary.items():
        means = [f"{item[key]['mean']:.2f}" if item[key] else "缺失" for key in METRICS]
        lines.append(f"| {name} | {item['scored']}/{item['attempted']} | {item['submitted']} | "
                     f"{item['all_bombs_placed']} | {item['run_failures']} | " + " | ".join(means) + " |")
    lines += ["", "详细单轮数据见 results.csv / results.json；极值和标准差见 summary.json。",
              "每轮目录保存日志、条件、评分明细及引擎输出。未提交不等于进程异常；两者分别统计。",
              "固定种子不保证逐帧一致。source_sha256 用于核对运行版本，非源代码备份。", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def worker_command(request_path=None, *, replay_path=None, ray_backend="engine"):
    # On Windows/Python 3.10 an editable-install .pth may be GBK, while the
    # team's YAML is UTF-8. Reuse this already initialized interpreter's paths,
    # then start the worker in UTF-8 mode without rereading those .pth files.
    # This changes no environment files and does not patch any engine APIs.
    arguments = (["--worker", str(request_path)] if replay_path is None else
                 ["--replay-worker", str(replay_path), "--ray-backend", ray_backend])
    bootstrap = (
        "import sys, runpy; "
        f"sys.path[:] = {sys.path!r}; "
        f"sys.prefix = {sys.prefix!r}; sys.exec_prefix = {sys.exec_prefix!r}; "
        f"sys.argv = {[str(Path(__file__).resolve()), *arguments]!r}; "
        f"runpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')"
    )
    return [sys.executable, "-u", "-B", "-S", "-X", "utf8", "-X", "faulthandler", "-c", bootstrap]


def execute_round(request, timeout, environment, scratch_dir):
    run_dir = Path(request["run_dir"])
    # OpenCV imwrite on this Windows build fails on Unicode output paths.
    # Let the engine write to an ASCII temporary path and copy all artifacts
    # using Python's Unicode-aware filesystem operations after the child exits.
    with tempfile.TemporaryDirectory(prefix="red-artifacts-", dir=scratch_dir) as temporary:
        if os.name == "nt" and not temporary.isascii():
            raise ValueError("Use --scratch-dir with a writable ASCII-only directory")
        request["engine_dir"] = str(Path(temporary) / "artifacts")
        write_json(run_dir / "request.json", request)
        with (run_dir / "run.log").open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(
                    worker_command(run_dir / "request.json"), cwd=ROOT, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                exit_code = process.returncode
                fallback = "process_error"
            except subprocess.TimeoutExpired:
                exit_code = None
                fallback = "process_timeout"
        artifacts = Path(request["engine_dir"])
        if artifacts.exists():
            shutil.copytree(artifacts, run_dir / "artifacts")
    result_path = run_dir / "result.json"
    if result_path.exists():
        row = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        row = dict(map=request["map"], round=request["round"], seed=request["seed"],
                   status=fallback, phase1=None)
    row["exit_code"] = exit_code
    if exit_code != 0 and row["status"] == "ok":
        row["status"] = fallback
    write_json(result_path, row)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", nargs="+", choices=MAPS, default=list(MAPS))
    parser.add_argument("--rounds", type=positive_int, default=3)
    parser.add_argument("--drones", type=positive_int, default=5)
    parser.add_argument("--bombs", type=positive_int, default=2)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--timeout", type=positive_int, default=600,
                        help="Per-process safety timeout, not simulation time budget")
    parser.add_argument("--no-recording", action="store_true")
    parser.add_argument("--output", help="New output directory; an existing path is rejected")
    parser.add_argument("--scratch-dir", help="Existing ASCII-only temporary directory on Windows")
    parser.add_argument("--ray-backend", choices=("engine", "cpu"), default="engine")
    parser.add_argument("--strategy", choices=("advanced2", "advanced5_narrow", "current"), default="current")
    parser.add_argument("--visual", action="store_true", help="Show the live simulation window")
    parser.add_argument("--replay", help="Play an existing NPZ with the selected ray backend")
    parser.add_argument("--replay-worker", help=argparse.SUPPRESS)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return run_worker(args.worker)
    if args.replay_worker:
        import runpy
        from swarm_rescue.solutions.devtools.replay_window import configure_replay_window
        configure_ray_backend(args.ray_backend)
        configure_replay_window()
        sys.argv = ["launcher", "--replay", args.replay_worker]
        runpy.run_module("swarm_rescue.launcher", run_name="__main__")
        return 0
    if args.replay:
        recording = Path(args.replay).resolve()
        if not recording.is_file():
            parser.error(f"Recording not found: {recording}")
        return subprocess.run(worker_command(replay_path=recording, ray_backend=args.ray_backend),
                              cwd=ROOT).returncode
    if not 0 <= args.seed <= 2**32 - 1 - 500000:
        parser.error("seed outside supported range")
    output = (Path(args.output) if args.output else ROOT / "output" / "red_baseline" /
              datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", environment_manifest(args))
    print(f"Output: {output}", flush=True)
    rows = []
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                       PYTHONDONTWRITEBYTECODE="1")
    for name in dict.fromkeys(args.maps):
        for round_number in range(1, args.rounds + 1):
            seed = (args.seed + MAPS.index(name) * 10000 + round_number - 1) % 2**32
            run_dir = output / f"{name}_r{round_number:02}"
            run_dir.mkdir()
            request = dict(map=name, round=round_number, seed=seed, run_dir=str(run_dir),
                           drones=args.drones, bombs=args.bombs, recording=not args.no_recording,
                           ray_backend=args.ray_backend, strategy=args.strategy, visual=args.visual)
            print(f"Starting {name} round {round_number}/{args.rounds}, seed={seed}", flush=True)
            row = execute_round(request, args.timeout, environment, args.scratch_dir)
            rows.append(row)
            save_reports(output, rows)
            score = (row.get("phase1") or {}).get("round_score")
            print(f"Finished {name} r{round_number}: {row['status']}, partial={score}, "
                  f"submitted={row.get('submitted')}", flush=True)
    print(f"Report: {output / 'summary.md'}", flush=True)
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

