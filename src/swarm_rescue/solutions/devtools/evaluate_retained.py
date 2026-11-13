"""Compare/demo the three retained versions without changing strategy sources.

Historical freezes describe the pre-cleanup workspace. Start a NEW batch
after cleanup rather than reusing those old source manifests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType
import zipfile

from swarm_rescue.solutions.devtools import evaluate_verified_revision as evaluation

ROOT = evaluation.ROOT
MAPS = ("Map01", "Map02", "Map03", "Map04", "Map05", "Map07")
IDENTITIES = dict(evaluation.CONTROLLERS)


def configure():
    evaluation.red.MAPS = MAPS
    evaluation.CONTROLLERS = MappingProxyType({**evaluation.CONTROLLERS, **IDENTITIES})
    evaluation.command = command


def command(request_path, phase):
    script = str(Path(__file__).resolve())
    argv = [script, "--worker", str(request_path), "--phase", phase]
    bootstrap = (f"import sys,runpy; sys.path[:] = {sys.path!r}; "
                 f"sys.prefix={sys.prefix!r}; sys.exec_prefix={sys.exec_prefix!r}; "
                 f"sys.argv={argv!r}; runpy.run_path({script!r},run_name='__main__')")
    return [sys.executable, "-u", "-B", "-S", "-X", "utf8", "-X", "faulthandler", "-c", bootstrap]


def reopen_batch(output, phase, args):
    """Continue only the identical frozen source; never replace completed scores."""
    batch=object.__new__(evaluation.Batch)
    batch.output,batch.phase,batch.args=Path(output).resolve(),phase,args
    batch.identity=evaluation.controller_identity(args.controller)
    batch.manifest=evaluation.read(batch.output/"manifest.json")
    evaluation.verify_controller(batch.manifest,args.controller)
    evaluation.verify_sources(batch.manifest)
    batch.manifest_sha=evaluation.digest(batch.output/"manifest.json")
    batch.origins={}
    batch.rows={}
    for row in evaluation.read(batch.output/"results.json"):
        run=batch.output/evaluation.round_name(row)
        if evaluation.read(run/"result.json")!=row:
            raise ValueError(f"Saved result changed: {run}")
        batch.rows[row["map"],row["round"]]=row
    return batch


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    phases = ("red",) if args.red_only else ("red", "blue")
    if not args.resume:
        with zipfile.ZipFile(output/"source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted((ROOT/"src").rglob("*.py")):
                archive.write(path, path.relative_to(ROOT).as_posix())
    schedule = []
    for name in args.maps:
        for number in range(1, args.rounds + 1):
            order = args.controllers if number % 2 else list(reversed(args.controllers))
            for selector in order:
                for phase in phases:
                    schedule.append(dict(map=name, round=number, controller=selector, phase=phase))
    plan=dict(
        created_utc=datetime.now(timezone.utc).isoformat(), maps=args.maps, rounds=args.rounds,
        controllers=args.controllers, seed=args.seed, drones=args.drones, bombs=args.bombs,
        ray_backend="cpu", competition_mode=True, simultaneous_simulations=1,
        original_limits=True, score_scope=evaluation.SCOPE, schedule=schedule,
        source_snapshot_sha256=evaluation.digest(output/"source_snapshot.zip"))
    thread_settings = {k: os.environ[k] for k in (
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENCV_FOR_THREADS_NUM") if k in os.environ}
    if thread_settings:
        plan["thread_environment"] = thread_settings
    if args.resume:
        old=evaluation.read(output/"plan.json")
        if old.get("thread_environment", {}) != thread_settings:
            raise ValueError("CPU thread configuration changed")
        for key in plan:
            if key!="created_utc" and old[key]!=plan[key]:
                raise ValueError(f"Resume conditions differ: {key}")
    else:
        evaluation.atomic_json(output / "plan.json", plan)
    ledger_path=output/"execution_receipts.json"
    ledger=evaluation.read(ledger_path) if args.resume else {}
    batches, requests = {}, {}
    for selector in args.controllers:
        namespace = argparse.Namespace(**vars(args), controller=selector, start_round=1)
        red = (reopen_batch(output/selector/"red","red",namespace) if args.resume else
               evaluation.Batch(output / selector / "red", "red", namespace))
        batches[selector] = {"red": red}
        if not args.red_only:
            batches[selector]["blue"] = (reopen_batch(output/selector/"blue","blue",namespace)
                if args.resume else evaluation.Batch(
                    output / selector / "blue", "blue", namespace, red_output=red.output))
        requests[selector] = {(r["map"], r["round"]): r for r in evaluation.red_requests(namespace)}
    history = []
    for item in schedule:
        selector, phase = item["controller"], item["phase"]
        batch = batches[selector][phase]
        key = item["map"], item["round"]
        if phase == "red":
            request = requests[selector][key]
        else:
            red = batches[selector]["red"]
            if not red.rows[key].get("submitted") or red.rows[key].get("phase1") is None:
                history.append(dict(**item, status="skipped_red_failure"))
                continue
            request = evaluation.blue_request(red.rows[key], red.output, args.drones)
        receipt_key=f"{selector}/{phase}/{evaluation.round_name(request)}"
        run=batch.output/evaluation.round_name(request)
        if key in batch.rows:
            if receipt_key in ledger:
                if evaluation.integrity(run,batch.manifest_sha)!=ledger[receipt_key]:
                    raise ValueError(f"Completed artifacts changed: {run}")
            else:
                evaluation.validate_receipt(run,batch.manifest_sha)
            row=batch.rows[key]
            print(f"RETAIN {len(history)+1}/{len(schedule)} {receipt_key}",flush=True)
        else:
            if run.exists():
                # An interrupted worker has no committed aggregate row. Preserve
                # it in this batch's quarantine; do not delete or count it as done.
                quarantine=output/"interrupted"/selector/phase
                quarantine.mkdir(parents=True,exist_ok=True)
                if not run.resolve().is_relative_to(output):
                    raise ValueError("Interrupted path escaped output")
                run.rename(quarantine/(run.name+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
            request, _ = batch.prepare(request)
            evaluation.atomic_json(output / "progress.json", dict(
                active=item, completed=len(history), expected=len(schedule), history=history))
            print(f"START {len(history)+1}/{len(schedule)} {selector} {phase} {key}", flush=True)
            row = batch.completed(request, evaluation.execute(request, phase, args.timeout))
        ledger[receipt_key]=evaluation.integrity(run,batch.manifest_sha)
        evaluation.atomic_json(ledger_path,ledger)
        history.append(dict(**item, status=row["status"], placed=row.get("placed_count"),
            submitted=row.get("submitted"), walltime_limited=row.get("walltime_limited"),
            score=(row.get("phase1") or {}).get("round_score")))
        evaluation.atomic_json(output / "progress.json", dict(
            active=None, completed=len(history), expected=len(schedule), history=history))
    for pair in batches.values():
        for batch in pair.values():
            evaluation.verify_sources(batch.manifest)
    complete = len(history) == len(schedule) and all(r["status"] == "ok" for r in history)
    placement_complete=all(r.get("placed")==args.drones*args.bombs and r.get("submitted")
                           for r in history if r["phase"]=="red")
    evaluation.atomic_json(output / "completion.json", dict(
        complete=complete, completed=len(history), expected=len(schedule),
        sources_unchanged=True, history=history,
        placement_complete=placement_complete))
    print(f"COMPLETE strict_ok={complete} {len(history)}/{len(schedule)}", flush=True)
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--phase", choices=("red", "blue"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", help="Batch output: use a NEW directory after cleanup")
    parser.add_argument("--controllers", nargs="+", choices=IDENTITIES, default=["advanced2", "advanced4"])
    parser.add_argument("--maps", nargs="+", choices=MAPS, default=list(MAPS))
    parser.add_argument("--rounds", type=int, choices=(1, 3), default=3)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--drones", type=int, default=5)
    parser.add_argument("--bombs", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--red-only", action="store_true")
    parser.add_argument("--resume",action="store_true",help="Continue identical frozen sources and conditions")
    args = parser.parse_args()
    configure()
    if args.worker:
        if not args.phase:
            parser.error("--worker requires --phase")
        return evaluation.worker(args.worker, args.phase)
    if min(args.drones, args.bombs, args.timeout) < 1:
        parser.error("Inventory, team and timeout must be positive")
    if args.live:
        if len(args.maps) != 1 or len(args.controllers) != 1:
            parser.error("--live requires one map and one controller")
        out = ROOT / "output/retained_runs" / ("live_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        out.mkdir(parents=True)
        request = dict(map=args.maps[0], round=1,
            seed=args.seed + MAPS.index(args.maps[0])*10000,
            drones=args.drones, bombs=args.bombs, recording=True, ray_backend="cpu",
            visual=True, strategy=evaluation.STRATEGY, run_dir=str(out),
            **evaluation.controller_identity(args.controllers[0]))
        row = evaluation.execute(request, "red", args.timeout)
        print(f"Live result: {row['status']} {out}")
        return 0 if row["status"] == "ok" else 1
    if not args.output:
        parser.error("--output is required for batch evaluation")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

