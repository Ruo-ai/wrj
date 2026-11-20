"""One official Map09 mapping-only trial in a fresh, resource-limited process."""

import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys
import traceback
from unittest.mock import patch


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_files(folder, members):
    for name, expected in members.items():
        if Path(name).name != name or sha256(folder / name) != expected:
            raise ValueError(f'Unexpected solution member: {name}')


def load_solution(request):
    import swarm_rescue.solutions as solutions

    workspace = Path(__file__).resolve().parents[1]
    extracted = Path(request['extracted_dir']).resolve(strict=True)
    if request['version'] == 'RL17-R':
        folder, members = extracted, request['expected_members']
        verify_files(folder, members)
        if any(name.startswith('swarm_rescue.solutions.my_drone')
               or name == 'swarm_rescue.solutions.semifinal_localization' for name in sys.modules):
            raise RuntimeError('A solution was imported before the extracted directory was selected')
        solutions.__path__ = [str(folder)]
        entry = importlib.import_module('swarm_rescue.solutions.my_drone_eval')
        controller = entry.drone_class_for_mode('place')
        if controller.__name__ != 'MyDroneRedRepair17':
            raise ValueError('Unexpected extracted controller')
    elif request['version'] == 'RL11':
        folder, members = workspace, request['expected_rl11']
        verify_files(folder, members)
        controller = importlib.import_module('swarm_rescue.solutions.my_drone_red_lines11').MyDroneRedLines11
    else:
        raise ValueError('Unsupported comparison version')
    return controller, folder, members


def loaded_identity(folder, members):
    recorded = {}
    for name, module in sys.modules.copy().items():
        if not name.startswith('swarm_rescue.solutions.'):
            continue
        tail = name.removeprefix('swarm_rescue.solutions.')
        if tail == 'devtools' or tail.startswith('devtools.'):
            continue
        path = Path(module.__file__).resolve(strict=True)
        if path.parent != folder or path.name not in members:
            raise ValueError(f'Unexpected solution origin: {name} {path}')
        actual = sha256(path)
        if actual != members[path.name]:
            raise ValueError(f'Loaded source changed: {path}')
        recorded[name] = dict(path=str(path), sha256=actual)
    return recorded


def fly(request, destination, result):
    import cv2
    import numpy as np
    from swarm_rescue.maps.map_09 import Map09
    from swarm_rescue.simulation.elements.sensor_disablers import NoComZone, NoGpsZone, ZoneType
    from swarm_rescue.simulation.gui_map import gui_sr
    from swarm_rescue.simulation.reporting.place_score_manager import PlaceScoreManager, walls_only_grid
    from swarm_rescue.solutions.devtools.evaluate_red import configure_ray_backend
    from swarm_rescue.solutions.devtools.payload_free_mapping_worker import check_empty

    if request['map'] != 'Map09':
        raise ValueError('This trial requires native Map09')
    controller, folder, members = load_solution(request)
    result['loaded_members'] = loaded_identity(folder, members)
    cv2.setNumThreads(1)
    configure_ray_backend('cpu')
    seed = request['seed']
    random.seed(seed)
    np.random.seed(seed)
    zones = (ZoneType.NO_COM_ZONE, ZoneType.NO_GPS_ZONE)
    world = Map09(drone_type=controller, zones_config=zones, number_drones=5)
    gui = None
    try:
        pg = world.playground
        pg.rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
        check_empty(world)
        if (len(world.drones) != 5 or any(type(d) is not controller for d in world.drones)
                or any(not any(isinstance(e, t) for e in pg.elements) for t in (NoGpsZone, NoComZone))):
            raise ValueError('Unexpected Map09 instance or roster')
        result.update(controller=controller.__module__ + '.' + controller.__name__,
                      instances=[type(d).__name__ for d in world.drones],
                      zones=[t.name for t in zones], initial_poses=world._drones_pos,
                      limit_steps=world.max_timestep_limit, limit_seconds=world.max_walltime_limit)
        for d in world.drones:
            d.set_competition_mode(True)
        # Preserve native scheduling, communication, early submission, and sensor guards.
        # Only the three propulsion commands are connected to the empty scene.
        guarded_control = gui_sr.run_control
        def movement_only(drone):
            commands = guarded_control(drone)
            movement = {}
            for key, value in commands.items():
                if key in ('forward', 'lateral', 'rotation'):
                    if not math.isfinite(value) or abs(value) > 1:
                        raise ValueError(f'Invalid movement value for {key}')
                    movement[key] = value
                elif value != 0:
                    raise ValueError(f'Nonzero non-movement output: {key}')
            return movement

        truth = walls_only_grid(pg)
        if truth is None or not truth.any():
            raise ValueError('Empty scoring truth')
        gui = gui_sr.GuiSR(the_map=world, headless=True, competition_mode=True)
        world.explored_map.reset()
        positions = []
        with patch.object(gui_sr, 'run_control', movement_only):
            while not gui._terminate:
                gui.on_update(1 / 30)
                check_empty(world)
                samples = np.full((5, 2), np.nan, dtype=np.float32)
                for d in world.drones:
                    if not d.removed:
                        samples[int(d.identifier)] = d.true_position()
                positions.append(samples)
                result.update(steps=gui.elapsed_timestep, wall_seconds=gui.elapsed_walltime)
                if len(positions) % 600 == 0:
                    print(f'frame={len(positions)} elapsed={gui.elapsed_walltime:.1f}s', flush=True)

        submitter = next((d for d in world.drones if d._exploration_map_submission is not None), None)
        pred = np.zeros_like(truth) if submitter is None else np.asarray(submitter._exploration_map_submission)
        if pred.shape != truth.shape:
            raise ValueError('Invalid submission dimensions')
        pred = (pred != 0).astype(np.uint8)
        score = PlaceScoreManager().compute_map_score(truth, pred, elapsed_walltime=gui.elapsed_walltime,
                                                       max_walltime_limit=world.max_walltime_limit)
        if score.error:
            raise ValueError(score.error)
        breakdown = score.breakdown
        coverage = float(100 * world.explored_map.score())
        visible = world.explored_map.get_pretty_map_explo_zones() == 128
        with (destination / 'flight.npz').open('xb') as stream:
            np.savez_compressed(stream, truth=truth, pred=pred, visible=visible,
                                positions=np.asarray(positions))
        verify_files(folder, members)
        result.update(loaded_members=loaded_identity(folder, members), status='ok',
                      submitted=submitter is not None, submitter=None if submitter is None else int(submitter.identifier),
                      stop_reason=('wall_limit' if gui.is_max_walltime_limit_reached else
                                   'submitted' if submitter is not None else 'step_or_native_termination'),
                      walltime_limited=bool(gui.is_max_walltime_limit_reached), frames=len(positions),
                      zero_payload_checks=len(positions) + 1, map_score=score.score,
                      coverage_percent=coverage, coverage_definition='Native 200px wall-constrained trajectory coverage',
                      wall_hits=breakdown['wall_hits'], wall_misses=breakdown['wall_pixels']-breakdown['wall_hits'],
                      charged_pixels=breakdown['charged_pixels'], map_breakdown=breakdown,
                      final_health=[int(d.drone_health) for d in world.drones])
    finally:
        if gui is None or not gui._terminate:
            world.playground.window.close()


def main():
    if len(sys.argv) != 2 or sys.stdin.readline().strip() != 'GO':
        raise SystemExit('Expected run directory and resource wrapper GO')
    directory = Path(sys.argv[1]).resolve(strict=True)
    req = json.loads((directory / 'request.json').read_text(encoding='utf-8-sig'))
    if any((directory / name).exists() for name in ('result.json', 'flight.npz')):
        raise FileExistsError('Existing attempt outputs')
    result = {key: req[key] for key in ('version', 'map', 'round', 'seed')}
    result.update(status='error', steps=0, wall_seconds=0, stop_reason='exception', payload=0, placed_count=0)
    try:
        fly(req, directory, result)
    except Exception as exc:
        traceback.print_exc()
        result['error'] = repr(exc)
    with (directory / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
