"""Serial payload-free Map09 comparison of an extracted submission and frozen RL11."""

import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import mkdtemp
from zipfile import ZipFile

from .optimization.evaluation.resources import run_limited
from . import red_exploration22_batch as baseline


REPO = baseline.REPO
SF13 = REPO / 'output/sf13'
OLD = SF13 / 'rl22_official_20261119_run'
INTERRUPTED = OLD / 'runs/Map09_r01/rl11_lines/red/attempt_001'
EXTRACTED = REPO / 'team742_preliminary'
WORKER = 'swarm_rescue.solutions.devtools.map09_extracted_mapping_worker'


def read(path):
    return json.loads(path.read_bytes())


def hashes(folder):
    return {p.relative_to(folder).as_posix(): baseline.digest(p)
            for p in sorted(folder.rglob('*')) if p.is_file()}


def preflight():
    p = read(OLD / 'plan.json')
    baseline.validate_plan(OLD, p)
    with ZipFile(baseline.ARCHIVE) as archive:
        members = {n: hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()}
    if hashes(EXTRACTED) != members:
        raise ValueError('Extracted files differ from the original ZIP')
    selected = [row for row in p['schedule'] if row['map'] == 'Map09'
                and row['controller'] in ('rl11_lines', 'archived_rl17r')]
    if len(selected) != 4 or p['configuration']['smoke']:
        raise ValueError('Unexpected source schedule')
    for row in selected:
        phase = OLD / 'runs' / f"Map09_r{row['round']:02}" / row['controller'] / 'red'
        if (phase / 'done.json').exists():
            raise ValueError(f'New official result present; review before running: {phase}')
    deps = {n: p['source_sha256']['src/swarm_rescue/solutions/' + n]
            for n in p['archive_audit']['red_dependency_sha256']}
    return selected, members, deps


def main():
    selected, members, deps = preflight()
    worker_file = Path(__file__).with_name('map09_extracted_mapping_worker.py')
    worker_sha = baseline.digest(worker_file)
    previous = hashes(INTERRUPTED)
    out = Path(mkdtemp(prefix='map09_extracted_unarmed_', dir=SF13))
    print(out, flush=True)
    plan = dict(map='Map09', schedule=selected, mode='payload-free mapping only',
                frozen_plan_sha256=baseline.digest(OLD / 'plan.json'),
                interrupted_original=str(INTERRUPTED), interrupted_files=previous,
                worker_sha256=worker_sha, driver_sha256=baseline.digest(Path(__file__)),
                zip_sha256=baseline.digest(baseline.ARCHIVE), extracted_members=members)
    baseline.write_new(out / 'plan.json', plan)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8',
               PYGLET_SHADOW_WINDOW='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', OPENCV_FOR_THREADS_NUM='1', PYTHONHASHSEED='0')
    verdicts = []
    for item in selected:
        version = 'RL11' if item['controller'] == 'rl11_lines' else 'RL17-R'
        run = out / f"r{item['round']}_{version}"
        run.mkdir()
        if baseline.digest(worker_file) != worker_sha or hashes(EXTRACTED) != members:
            raise ValueError('Worker or extracted submission changed')
        request = dict(version=version, seed=item['seed'], round=item['round'], map='Map09',
                       extracted_dir=str(EXTRACTED), expected_members=members, expected_rl11=deps)
        baseline.write_new(run / 'request.json', request)
        print('START', run.name, 'seed', item['seed'], flush=True)
        with (run / 'run.log').open('xb') as log:
            resources = run_limited([sys.executable, '-B', '-u', '-m', WORKER, str(run)],
                                    env, log, 240, memory_bytes=4_000_000_000)
        baseline.write_new(run / 'resources.json', resources)
        result = read(run / 'result.json') if (run / 'result.json').is_file() else {}
        issues = []
        if resources.get('status') != 'exited' or resources.get('exit_code') != 0:
            issues.append('worker:' + str(resources.get('status')))
        if result.get('status') != 'ok':
            issues.append(str(result.get('error', 'missing or unsuccessful result')))
        if not result.get('submitted'):
            issues.append('no submitted map')
        if result.get('payload') != 0 or result.get('placed_count') != 0:
            issues.append('zero payload result unverified')
        if not (run / 'flight.npz').is_file():
            issues.append('missing flight arrays')
        log_text = (run / 'run.log').read_text(encoding='utf-8', errors='replace')
        for marker in ('Traceback (most recent call last)', 'Exception ignored from cffi callback',
                       'Fatal Python error'):
            if marker in log_text:
                issues.append(marker)
        baseline.write_new(run / 'integrity.json', hashes(run))
        verdicts.append(dict(run=run.name, seed=item['seed'], issues=issues))
        print('END', run.name, 'steps', result.get('steps'), 'issues', issues, flush=True)
        if issues and not (run / 'flight.npz').exists() and not result.get('steps'):
            break
    baseline.validate_plan(OLD, read(OLD / 'plan.json'))
    if hashes(INTERRUPTED) != previous or baseline.digest(worker_file) != worker_sha or hashes(EXTRACTED) != members:
        raise ValueError('Source or interrupted run bytes changed')
    baseline.write_new(out / 'verdicts.json', dict(expected=4, rows=verdicts,
                                                   interrupted_bytes_verified=True))
    return 0 if len(verdicts) == 4 and all(not r['issues'] for r in verdicts) else 2


if __name__ == '__main__':
    raise SystemExit(main())
