"""Read the archived R21d receipts and reconcile task exits with recorded events."""

import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from tempfile import mkdtemp
from time import perf_counter


REPO = Path(__file__).resolve().parents[4]
OUTPUT = REPO / "output/sf13"
BATCH = OUTPUT / "r21d_20261118_095350_321"
REVIEW = OUTPUT / "r21d_review_20261118_095350_321"
PLAN_HASH = "fa1e3103d4f854dcd7abb5d8acd2fbe728c6bb48c34b910eec34bf5176d4a2e7"
# Each pair identifies the completed attempt's integrity.json and its offline event index.
RECEIPTS = {
    ("yh_map_fs_72", 1): ("c8536ba40dd6cb52f0298669f9f9837964d7315bc01c8bfbed0f01e8d541297e",
                          "20aedd22bfa35f4f55b40ae20daf2ef47676e6a2a9760ee3a02b26649bc9bef4"),
    ("yh_map_fs_72", 2): ("b35ff1d383c0892248b9c9b238a18f93811acaf537dc3ff75672d790cd8b0eba",
                          "a393c42527c70d40d41db3e40020f4b95575c5f6ef801ad4076c51649aa1985b"),
    ("yh_map_fs_37", 1): ("817493bc146dcdb9b38e8b398d687f5d3bf89fc52ab2180022f962f098951d8d",
                          "4e0d4e114f6e26008db5cfae0dc11d03630395a833ebfc1b69f7afbff2c0d8bf"),
    ("yh_map_fs_37", 2): ("e559fe4294ebaba5a26d83207f341d49657d93e2c7dc433a08d624c1f5e33cff",
                          "2d33efe8e773442b3a1b43d3e5a639aafb491d6c0a388b238ad15d8271cf8ccc"),
    ("yh_map_fs_24", 1): ("2ad67af280ae89800183d73f9eb23a7ded06227a41d7de06cababac9a982910a",
                          "82e1e2922fd3c8dbfb7f1fcf3cbb68851682188c587f0ee557dcc490bf7d962e"),
    ("yh_map_fs_24", 2): ("80ad23f3a0391fd1d08c953eabd80acdfde96e9c5eefcd71d33998df9105ada0",
                          "cd855c3d1999714ca74583387da2204162ad96c60b640e26e9355ac9118755c5"),
}
EXIT_REASONS = {
    "completed", "fresh_scan_obstructed", "mission_preempted", "no_progress",
    "task_timeout", "invalid_odometry", "peer_wait_budget", "inertial_risk",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def verify(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"{path}: expected {expected}, found {actual}")


def recorded_tasks(narrow):
    tasks = {}
    for owner_id, drone in narrow["drones"].items():
        owner = int(owner_id)
        active = None
        previous_tick = -1
        for event in drone["events"]:
            tick, reason = event["tick"], event["reason"]
            if tick < previous_tick:
                raise ValueError(f"Unordered events for {owner} at {tick}")
            previous_tick = tick
            if reason == "entered" and active is None and (owner, tick) not in tasks:
                active = dict(owner=owner, start=tick, replan_ticks=[])
            elif reason == "local_replan" and active is not None:
                active["replan_ticks"].append(tick)
            elif reason in EXIT_REASONS and active is not None:
                active.update(end=tick, reason=reason)
                tasks[owner, active["start"]] = active
                active = None
            else:
                raise ValueError(f"Unpaired event for {owner} at {tick}: {reason}")
        if active is not None:
            raise ValueError(f"Missing exit for {owner} at {active['start']}")
    return tasks


def reconcile_tasks(narrow, review):
    """Match raw task windows to the independent, offline list of physical crossings."""
    raw = recorded_tasks(narrow)
    counts = Counter()
    seen_events = set()
    for crossing in review["crosses"]:
        owner, tick = crossing["owner"], crossing["tick"]
        event_key = (owner, tick, crossing["gate"])
        if event_key in seen_events:
            raise ValueError(f"Duplicate crossing {event_key}")
        seen_events.add(event_key)
        matches = [key for key, task in raw.items()
                   if key[0] == owner and task["start"] <= tick <= task["end"]]
        claimed = crossing["task"]
        if matches != ([] if claimed is None else [(owner, claimed)]):
            raise ValueError(f"Crossing {event_key} disagrees with raw task window")
        counts.update(matches)

    tasks = []
    for entry in review["tasks"]:
        key = (entry["owner"], entry["start"])
        task = raw.pop(key, None)
        if task is None or (task["end"], task["reason"]) != (entry["end"], entry["reason"]):
            raise ValueError(f"Task {key}: missing, repeated, or inconsistent exit")
        samples, cells = entry["samples"], entry["new_cells"]
        if samples < 0 or cells < 0 or (not samples and cells):
            raise ValueError(f"Task {key}: inconsistent coverage samples")
        if counts[key] != entry["cross"]:
            raise ValueError(f"Task {key}: inconsistent crossing count")
        tasks.append(dict(**task, crossings=counts[key], samples=samples,
                          sampled_new_cells=cells if samples else None))
    if raw:
        raise ValueError(f"Missing tasks in offline index: {list(raw)}")
    return tasks


def audit_round(item):
    name, number = item["name"], item["round"]
    receipt_hash, event_hash = RECEIPTS[name, number]
    root = BATCH / "runs" / f"{name}_r{number:02d}" / item["controller"] / "red"
    done = read(root / "done.json")
    if done != dict(attempt="attempt_001", receipt_sha256=receipt_hash):
        raise ValueError(f"{root}: completion receipt changed")
    attempt = root / done["attempt"]
    verify(attempt / "integrity.json", receipt_hash)
    receipt = read(attempt / "integrity.json")
    if receipt["plan_sha256"] != PLAN_HASH:
        raise ValueError(f"{root}: unexpected plan")
    for filename, expected in receipt["files"].items():
        verify(attempt / filename, expected)
    event_file = REVIEW / f"{name}_r{number}_events.json"
    verify(event_file, event_hash)
    index = read(event_file)
    tasks = reconcile_tasks(read(attempt / "narrow_diagnostics.json"), index)
    result = read(attempt / "result.json")
    if (result["round"], result["map"], result["seed"]) != (number, item["map"], item["seed"]):
        raise ValueError(f"{root}: result identity differs from the schedule")
    return dict(map=name, round=number, seed=item["seed"], receipt_sha256=receipt_hash,
                event_index_sha256=event_hash, verified_files=len(receipt["files"]),
                engine_status=result["status"], exit_code=result["exit_code"],
                callback_errors=result["engine_callback_errors"], tasks=tasks,
                exit_reasons=dict(Counter(t["reason"] for t in tasks)),
                crossings_all=len(index["crosses"]),
                crossings_in_tasks=sum(t["crossings"] for t in tasks))


def audit(report):
    verify(BATCH / "plan.json", PLAN_HASH)
    plan = read(BATCH / "plan.json")
    verify(BATCH / "source_bytes.json", plan["source_bytes_sha256"])
    frozen = read(BATCH / "source_bytes.json")
    if frozen.keys() != plan["source_sha256"].keys():
        raise ValueError("Source archive index differs from the frozen plan")
    for name, expected in plan["source_sha256"].items():
        data = base64.b64decode(frozen[name], validate=True)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Unexpected archived bytes for {name}")
        live = REPO / name
        actual = digest(live) if live.is_file() else None
        if actual != expected:
            report["live_differences"].append(dict(path=name, expected=expected, actual=actual))
    report["frozen_files"] = len(frozen)
    for item in plan["schedule"]:
        try:
            report["rounds"].append(audit_round(item))
        except (OSError, ValueError, KeyError, TypeError) as error:
            report["errors"].append(f"{item['name']} r{item['round']}: {error}")


def main():
    if len(sys.argv) != 1:
        sys.exit("Run without arguments to audit the fixed historical archive.")
    start = perf_counter()
    output = Path(mkdtemp(prefix="r21d_receipts_offline_", dir=OUTPUT))
    report = dict(batch=BATCH.name, review=REVIEW.name, plan_sha256=PLAN_HASH,
                  tool_sha256=digest(Path(__file__)), new_flights=0, rounds=[],
                  live_differences=[], errors=[])
    try:
        audit(report)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["errors"].append(str(error))
    report["audit_ok"] = not (report["errors"] or report["live_differences"])
    report["audit_seconds"] = perf_counter() - start
    with (output / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"{(output / 'audit.json').relative_to(REPO)} | offline audit_ok={report['audit_ok']}, "
          f"archived rounds={len(report['rounds'])}, new flights=0")
    return 0 if report["audit_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
