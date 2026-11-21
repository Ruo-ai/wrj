"""Fixed three-map trial of a final-raster change with passive map diagnostics."""

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from . import red_exploration22_batch as prior
from .optimization.evaluation import runner
from .optimization.evaluation.report import flatten
from .optimization.evaluation.scenarios import official_rows


MODULE = 'swarm_rescue.solutions.devtools.map23_trial'
SELECTOR = 'map23'
IDENTITY = ('my_drone_red_map23', 'MyDroneRedMap23')
HISTORY = runner.REPO / 'output/sf13/rl22_official_20261119_run'


def register():
    runner.ev.CONTROLLERS = {**runner.ev.CONTROLLERS, SELECTOR: IDENTITY}


class WallReturns:
    """Passive first physical WALL return per image pixel, inaccessible to the mapper."""

    def __init__(self):
        self.wall_ids = None
        self.first = None
        self.scans = 0

    def observe(self, drone):
        from swarm_rescue.simulation.utils.definitions import CollisionTypes

        sensor = drone.lidar()
        if self.wall_ids is None:
            self.wall_ids = np.array(sorted(
                e.uid for e in sensor.playground.elements
                if getattr(e, '_collision_type', None) == CollisionTypes.WALL))
            width, height = map(int, drone.size_area)
            self.first = np.zeros((height, width), np.uint16)
            if not len(self.wall_ids):
                raise ValueError('No wall entities for sensor audit')
        if drone.lidar_values() is None or sensor._hitpoints is None:
            return
        self.scans += 1
        hp = sensor._hitpoints
        points = hp[np.isin(hp[:, 8], self.wall_ids), 2:4]
        h, w = self.first.shape
        x = np.rint(points[:, 0] + w / 2).astype(int)
        y = np.rint(h / 2 - points[:, 1]).astype(int)
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x, y = x[valid], y[valid]
        unseen = self.first[y, x] == 0
        self.first[y[unseen], x[unseen]] = int(drone._steps)


def worker(request_path):
    if sys.stdin.readline().strip() != 'GO':
        raise RuntimeError('Missing resource wrapper')
    if os.name != 'nt':
        from .optimization.evaluation.resources import linux_limit
        linux_limit(int(os.environ['G3_MEMORY_BYTES']))
    from swarm_rescue.simulation.gui_map import gui_sr
    from .optimization.evaluation.worker import run
    from .red_lines11_diagnostics import recording as legal_inputs
    from .red_rx10_diagnostics import recording as source_maps

    req = runner.read(request_path)
    attempt = Path(req['run_dir'])
    plan = next(p / 'plan.json' for p in attempt.parents if (p / 'plan.json').is_file())
    if prior.digest(plan) != req['plan_sha256']:
        raise ValueError('Request/plan mismatch')
    runner.validate_sources(runner.read(plan)['source_sha256'])
    register()
    original = gui_sr.run_control
    identities = {}
    walls = WallReturns()

    def observe(drone):
        cmd = original(drone)
        ident = str(drone.identifier)
        if ident not in identities:
            cls = type(drone)
            mapper = getattr(drone, '_submission_mapper', None)
            identities[ident] = dict(controller=cls.__module__ + '.' + cls.__name__,
                mapper=None if mapper is None else type(mapper).__module__ + '.' + type(mapper).__name__)
        if req['phase'] == 'red':
            walls.observe(drone)
        return cmd

    with ExitStack() as stack:
        stack.enter_context(patch.object(gui_sr, 'run_control', observe))
        if req['phase'] == 'red':
            stack.enter_context(legal_inputs(attempt / 'line_diagnostics.json'))
            stack.enter_context(source_maps(attempt / 'exploration_diagnostics.json'))
        code = run(request_path)
    if walls.first is not None:
        np.savez_compressed(attempt / 'wall_returns.npz', first_tick=walls.first, scan_count=walls.scans)
    modules = {identity['controller'].rsplit('.', 1)[0] for identity in identities.values()}
    modules.update(n for n in sys.modules if n.startswith('swarm_rescue.solutions.my_drone_red')
                   or n == 'swarm_rescue.solutions.semifinal_localization')
    files = [Path(sys.modules[n].__file__).resolve() for n in modules]
    prior.write_new(attempt / 'loaded_identity.json', dict(observed_instances=identities,
        loaded_sources={f.relative_to(runner.REPO).as_posix(): prior.digest(f) for f in files}))
    return code


def prepare(out, smoke):
    archive = prior.ARCHIVE
    if prior.digest(archive) != prior.EXPECTED_ARCHIVE_SHA256:
        raise ValueError('Reference ZIP changed')
    with ZipFile(archive) as z:
        for name in z.namelist():
            if name.endswith('.py') and name not in ('__init__.py', 'my_drone_eval.py'):
                if z.read(name) != (runner.REPO / 'src/swarm_rescue/solutions' / name).read_bytes():
                    raise ValueError('Live parent differs from archived member ' + name)
    rows = official_rows(prior.MAPS[:1] if smoke else prior.MAPS)
    old = runner.read(HISTORY / 'plan.json')
    schedule = [dict(item, controller=SELECTOR) for item in old['schedule']
                if item['controller'] == 'archived_rl17r' and item['round'] == 1
                and item['name'] in {r['name'] for r in rows}]
    if [i['map_id'] for i in schedule] != [r['id'] for r in rows]:
        raise ValueError('Historical map identity mismatch')
    if any(i['seed'] != 2026111901 or i['blue_variant'] != 'adapted' for i in schedule):
        raise ValueError('Historical round settings changed')
    config = dict(old['configuration'], smoke=smoke, diagnostic_steps=16 if smoke else None,
                  controllers=[SELECTOR], controller_identities={SELECTOR: runner.ev.controller_identity(SELECTOR)})
    sources = runner.source_receipt(rows)
    sources[archive.name] = prior.digest(archive)
    out.mkdir(parents=True, exist_ok=False)
    source = runner.REPO / 'src/swarm_rescue/solutions/my_drone_red_map23.py'
    (out / source.name).write_bytes(source.read_bytes())
    plan = dict(configuration=config, schedule=schedule, map_rows=rows, source_sha256=sources,
                head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                frozen_candidate_sha256=prior.digest(source), historical_plan_sha256=prior.digest(HISTORY / 'plan.json'))
    prior.write_new(out / 'plan.json', plan)
    prior.write_new(out / 'frozen.json', dict(plan_sha256=prior.digest(out / 'plan.json'),
        candidate_sha256=prior.digest(out / source.name), smoke=smoke))
    return plan


def trial(out, smoke):
    register()
    plan = prepare(out, smoke)
    prior.use_ascii_temp()
    os.environ['PYGLET_SHADOW_WINDOW'] = 'false'
    results = []
    with patch.object(prior, 'MODULE', MODULE), patch.object(runner, 'worker_command', prior.command):
        for item in plan['schedule']:
            print(item['map'], 'starting', flush=True)
            runner.validate_sources(plan['source_sha256'])
            red, attempt = runner.execute_phase(out, plan, item, 'red')
            if red['status'] != 'ok':
                raise RuntimeError(f"Red stage failed: {attempt} ({red['status']})")
            issues = prior.loaded_issues(red, attempt, 'red', smoke)
            for name in ('line_diagnostics.json', 'exploration_diagnostics.json'):
                issues.extend(runner.read(attempt / name)['errors'])
            if issues:
                raise ValueError(issues)
            if smoke:
                np.load(attempt / 'mapper_legal_inputs.npz').close()
                with np.load(attempt / 'wall_returns.npz') as obs:
                    if not obs['first_tick'].any():
                        raise ValueError('Missing physical wall returns')
                print('16-step recorder smoke passed', flush=True)
                continue
            blue, blue_attempt = None, None
            if runner.phase_valid(red, 'red'):
                blue, blue_attempt = runner.execute_phase(out, plan, item, 'blue', attempt)
                if (blue_attempt / 'loaded_identity.json').is_file():
                    issues.extend(prior.loaded_issues(blue, blue_attempt, 'blue', False))
            row = flatten(dict(item, red=red, blue=blue, red_run=str(attempt), blue_run=str(blue_attempt)))
            results.append(dict(map=item['name'], seed=item['seed'], red_valid=runner.phase_valid(red, 'red'),
                map_score=row['map_score'] if runner.phase_valid(red, 'red') else None,
                full_total=row['total_score'] if row['valid_score'] and not issues else None,
                failure=row['failure_reason'], identity_issues=issues,
                red_attempt=str(attempt), blue_attempt=str(blue_attempt) if blue_attempt else None))
            runner.write(out / 'results.json', results)
            print(results[-1], flush=True)
    runner.validate_sources(plan['source_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path)
    group.add_argument('--worker', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker(args.worker))
    trial(args.output.resolve(), args.smoke)
