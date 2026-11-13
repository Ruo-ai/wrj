"""Explicit, isolated controller workers and audited resumable evaluations.

The controller is carried by each request; the submission entry is untouched.
Every output is new. Reuse copies verified completed rounds into that new output.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from swarm_rescue.solutions.devtools.optimization.evaluation.identities import RED_CONTROLLERS

from swarm_rescue.solutions.devtools import evaluate_red as red
from swarm_rescue.solutions.devtools import evaluate_provided_blue as blue

ROOT = red.ROOT
SCOPE = "local_provided_blue_only; reinforcement matched simulation, not platform score"
STRATEGY = "reinforced_evaluation"
CONTROLLERS = RED_CONTROLLERS


def controller_identity(selector):
    """Serializable identity, with no lazy-import or shared selector side effect."""
    try:
        module, name = CONTROLLERS[selector]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown explicit controller: {selector!r}") from exc
    return dict(controller_selector=selector, controller_class=name,
                controller_module=f"swarm_rescue.solutions.{module}")


def verify_controller(record, expected=None):
    identity = controller_identity(record.get("controller_selector"))
    if any(record.get(key) != value for key, value in identity.items()):
        raise ValueError("Controller selector, module and class provenance disagree")
    if expected is not None and identity != controller_identity(expected):
        raise ValueError("Different controller in provenance")
    return identity


def load_controller(request):
    identity = verify_controller(request)
    return getattr(importlib.import_module(identity["controller_module"]),
                   identity["controller_class"])


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_sources(manifest):
    for name, sha in manifest["source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Frozen source changed: {name}")


def command(request_path, phase):
    script = str(Path(__file__).resolve())
    argv = [script, "--worker", str(request_path), "--phase", phase]
    bootstrap = (f"import sys,runpy; sys.path[:] = {sys.path!r}; "
                 f"sys.prefix={sys.prefix!r}; sys.exec_prefix={sys.exec_prefix!r}; "
                 f"sys.argv={argv!r}; runpy.run_path({script!r},run_name='__main__')")
    return [sys.executable, "-u", "-B", "-S", "-X", "utf8", "-X", "faulthandler", "-c", bootstrap]


def worker(path, phase):
    # All monkey patches are confined to this one-round child process.
    request = read(path)
    verify_controller(request)
    red.write_json = atomic_json
    if phase == "blue":
        return blue.worker(path)
    import swarm_rescue.launcher as launcher
    controller = load_controller(request)
    entry = getattr(launcher, "drone_entry", launcher)
    original = entry.drone_class_for_mode
    entry.drone_class_for_mode = lambda mode: controller if mode == "place" else original(mode)
    # run_worker already supports the current selector. Keep the actual selector
    # and immutable public request explicit for provenance.
    request["strategy"] = "current"
    internal = Path(path).with_name("engine_request.json")
    atomic_json(internal, request)
    return red.run_worker(internal)


def execute(request, phase, timeout=600, scratch_dir=None):
    """Same ASCII staging as evaluate_red, with explicit per-call worker command."""
    identity = verify_controller(request)
    run = Path(request["run_dir"])
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                       PYTHONDONTWRITEBYTECODE="1", SWARM_BLUE_DEBUG="1")
    with tempfile.TemporaryDirectory(prefix=f"reinforced-{phase}-", dir=scratch_dir) as temporary:
        if os.name == "nt" and not temporary.isascii():
            raise ValueError("--scratch-dir must provide an ASCII temporary path")
        request = dict(request, engine_dir=str(Path(temporary) / "artifacts"))
        atomic_json(run / "request.json", request)
        with (run / "run.log").open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(command(run / "request.json", phase), cwd=ROOT,
                    env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                exit_code, fallback = process.returncode, "process_error"
            except subprocess.TimeoutExpired:
                exit_code, fallback = None, "process_timeout"
        artifacts = Path(request["engine_dir"])
        if artifacts.exists():
            shutil.copytree(artifacts, run / "artifacts")
    result = (read(run / "result.json") if (run / "result.json").is_file() else
              dict(map=request["map"], round=request["round"], seed=request["seed"], status=fallback, phase1=None))
    result["exit_code"] = exit_code
    result.update(identity)
    if exit_code != 0 and result["status"] == "ok":
        result["status"] = fallback
    atomic_json(run / "result.json", result)
    return result


def round_name(row):
    return f"{row['map']}_r{row['round']:02}"


def one_layout(run):
    files = list((Path(run) / "artifacts").glob("*_bombs.json"))
    if len(files) != 1:
        raise ValueError(f"Expected one bomb layout in {run}, got {len(files)}")
    return files[0]


def integrity(run, manifest_sha):
    return {"manifest_sha256": manifest_sha,
            "files": {p.relative_to(run).as_posix(): digest(p) for p in sorted(run.rglob("*"))
                      if p.is_file() and p.name not in ("integrity.json", "integrity.json.tmp")}}


def validate_receipt(run, manifest_sha):
    receipt = read(run / "integrity.json")
    if receipt["manifest_sha256"] != manifest_sha:
        raise ValueError(f"Manifest provenance mismatch: {run}")
    actual = integrity(run, manifest_sha)
    if actual != receipt:
        raise ValueError(f"Round file integrity mismatch: {run}")


def valid_result(row, request, phase):
    verify_controller(request)
    verify_controller(row, request["controller_selector"])
    if any(row.get(key) != request[key] for key in ("map", "round", "seed")):
        raise ValueError("Round identity or seed mismatch")
    if row.get("status") != "ok" or row.get("exit_code") != 0:
        raise ValueError("Cannot reuse unsuccessful result")
    if row.get("walltime_limited") or row.get("controller_crashed") or row.get("phase1") is None:
        raise ValueError("Cannot reuse incomplete or limited result")
    if phase == "red":
        if not row.get("submitted") or not row.get("valid_shape"):
            raise ValueError("Red submission missing or malformed")
        if row.get("expected_count") != request["drones"] * request["bombs"]:
            raise ValueError("Red inventory conditions mismatch")
    else:
        checks = {"layout_sha256": digest(request["layout"]),
                  "red_result_sha256": digest(request["red_result"]),
                  "blue_source_sha256": digest((Path(blue.__file__).parent.parent / "my_drone_eval_blue.py"))}
        if any(row.get(key) != value for key, value in checks.items()):
            raise ValueError("Blue source, source result, or layout mismatch")


class Batch:
    """Coordinator-thread-only bookkeeping; execute() is safely concurrent."""

    def __init__(self, output, phase, args, reuse=(), red_output=None):
        self.identity = controller_identity(args.controller)
        source = Path(__file__).parent.parent / (CONTROLLERS[args.controller][0] + ".py")
        if not source.is_file():
            raise ValueError(f"Controller implementation must exist before source freeze: {source}")
        self.output, self.phase, self.args = Path(output).resolve(), phase, args
        self.output.mkdir(parents=True, exist_ok=False)
        namespace = argparse.Namespace(**vars(args))
        namespace.output = str(self.output)
        namespace.phase = phase
        namespace.strategy = STRATEGY
        namespace.ray_backend = "cpu"
        namespace.no_recording = False
        namespace.visual = False
        manifest = red.environment_manifest(namespace)
        manifest["arguments"] = {k: str(v) if isinstance(v, Path) else v
                                 for k, v in vars(namespace).items()}
        manifest["score_scope"] = SCOPE if phase == "blue" else "red contribution /80; final paired score pending"
        manifest.update(self.identity)
        manifest["reused_from"] = [str(Path(p).resolve()) for p in reuse]
        if red_output is not None:
            verify_controller(read(Path(red_output) / "manifest.json"), args.controller)
            manifest["red_manifest_sha256"] = digest(Path(red_output) / "manifest.json")
        atomic_json(self.output / "manifest.json", manifest)
        self.manifest, self.manifest_sha = manifest, digest(self.output / "manifest.json")
        self.rows, self.origins = {}, {}
        for directory in reuse:
            directory = Path(directory).resolve()
            old = read(directory / "manifest.json")
            verify_controller(old, args.controller)
            verify_sources(old)
            if old["arguments"].get("strategy") != STRATEGY or old["arguments"].get("phase") != phase:
                raise ValueError(f"Wrong strategy or phase in reuse source {directory}")
            for key in ("seed", "drones", "bombs", "ray_backend", "no_recording", "controller"):
                if old["arguments"].get(key) != manifest["arguments"].get(key):
                    raise ValueError(f"Changed reuse condition {key}: {directory}")
            # Discover complete receipts even if the parent was interrupted before
            # its aggregate results.json was updated. Failed rounds remain visible
            # in the old directory and are not reused.
            for result_path in sorted(directory.glob("Map*_r*/result.json")):
                row = read(result_path)
                if row.get("status") != "ok" or not (result_path.parent / "integrity.json").is_file():
                    continue
                validate_receipt(result_path.parent, digest(directory / "manifest.json"))
                key = row["map"], row["round"]
                if key in self.origins and self.origins[key][1] != row:
                    raise ValueError(f"Conflicting successful reuse results: {key}")
                self.origins[key] = result_path.parent, row
        atomic_json(self.output / "results.json", [])

    def prepare(self, request):
        verify_controller(request, self.args.controller)
        run = self.output / round_name(request)
        request = dict(request, run_dir=str(run))
        key = request["map"], request["round"]
        if key in self.origins:
            source, row = self.origins[key]
            valid_result(row, request, self.phase)
            old_request = read(source / "request.json")
            verify_controller(old_request, self.args.controller)
            for field in ("drones", "bombs", "recording", "ray_backend", "strategy"):
                if field in request and old_request.get(field) != request[field]:
                    raise ValueError(f"Reused request mismatch: {field}")
            if self.phase == "red":
                layout = read(one_layout(source))
                if layout["map_name"] != request["map"] or len(layout["bombs"]) != row["placed_count"]:
                    raise ValueError("Reused layout inconsistent with red result")
            shutil.copytree(source, run)
            atomic_json(run / "integrity.json", integrity(run, self.manifest_sha))
            self.record(row)
            print(f"REUSED {self.phase} {round_name(row)}", flush=True)
            return request, row
        run.mkdir()
        return request, None

    def record(self, row):
        self.rows[row["map"], row["round"]] = row
        ordered = [self.rows[key] for key in sorted(self.rows)]
        atomic_json(self.output / "results.json", ordered)
        if self.phase == "red":
            atomic_json(self.output / "summary.json", red.summarize(ordered))

    def completed(self, request, row):
        run = self.output / round_name(row)
        # Preserve errors in results, but never issue reusable integrity receipts
        # for malformed or incomplete rounds.
        try:
            valid_result(row, request, self.phase)
            if self.phase == "red":
                layout = read(one_layout(run))
                if layout["map_name"] != row["map"] or len(layout["bombs"]) != row["placed_count"]:
                    raise ValueError("Generated layout inconsistent with red result")
        except ValueError as exc:
            row = dict(row, validation_error=str(exc))
            if row.get("status") == "ok":
                row["status"] = "validation_error"
            atomic_json(run / "result.json", row)
        else:
            atomic_json(run / "integrity.json", integrity(run, self.manifest_sha))
        self.record(row)
        score = (row.get("phase1") or {}).get("round_score")
        print(f"DONE {self.phase} {round_name(row)} status={row['status']} score={score} "
              f"blue={row.get('blue_score')} count={len(self.rows)}", flush=True)
        return row


def red_requests(args):
    identity = controller_identity(args.controller)
    for map_name in dict.fromkeys(args.maps):
        for number in range(getattr(args, "start_round", 1), getattr(args, "start_round", 1) + args.rounds):
            yield dict(map=map_name, round=number, seed=args.seed + red.MAPS.index(map_name) * 10000 + number - 1,
                       drones=args.drones, bombs=args.bombs, recording=True, ray_backend="cpu",
                       strategy=STRATEGY, visual=False, **identity)


def blue_request(row, red_output, drones=5):
    manifest = read(Path(red_output) / "manifest.json")
    identity = verify_controller(manifest)
    verify_controller(row, identity["controller_selector"])
    run = Path(red_output) / round_name(row)
    return dict(map=row["map"], round=row["round"], seed=row["seed"], drones=drones,
                layout=str(one_layout(run)), red_result=str(run / "result.json"), **identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--phase", choices=("red", "blue"), required=True)
    args = parser.parse_args()
    return worker(args.worker, args.phase)


if __name__ == "__main__":
    raise SystemExit(main())

