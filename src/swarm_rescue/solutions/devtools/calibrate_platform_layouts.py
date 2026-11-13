"""Replay verified Map07 layouts with the frozen local blue, without synthesizing red scores."""
import json
import os
from pathlib import Path
import sys
from zipfile import ZipFile

from swarm_rescue.solutions.devtools import evaluate_provided_blue as blue
from swarm_rescue.solutions.devtools import evaluate_red as red


def main(output, variant='baseline'):
    if variant not in ('baseline', 'sweep'):
        raise ValueError(variant)
    red.configure_ray_backend('cpu')
    from swarm_rescue.maps.map_07 import Map07
    from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
    from swarm_rescue.simulation.reporting.place_score_manager import PlaceScoreManager, walls_and_boundary_grid
    output.mkdir(parents=True, exist_ok=False)
    world = Map07(drone_type=DroneMotionless, number_drones=5)
    walls, boundary = walls_and_boundary_grid(world._playground)
    world._playground.window.close()
    red.worker_command = blue.command
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               PYTHONDONTWRITEBYTECODE='1', SWARM_BLUE_DEBUG='1')
    rows = []
    for archive_path in sorted((red.ROOT/'平台评测结果').glob('*.zip')):
        with ZipFile(archive_path) as archive:
            result_path, = [n for n in archive.namelist() if n.endswith('/result.json')]
            for detail in json.loads(archive.read(result_path))['detail']:
                member, = [n for n in archive.namelist() if n.endswith('/'+detail['炸弹文件'])]
                raw = archive.read(member)
                layout = json.loads(raw)
                count = detail['放置数']
                points = [(b['x'], b['y']) for b in layout['bombs']]
                if layout['map_name'] != 'Map07' or len(points) != count:
                    raise ValueError(member)
                wall = 100 * PlaceScoreManager().compute_wall_score(points, walls,
                    boundary_mask=boundary, expected_n_bombs=count).score
                number = detail['轮次']
                run = output/archive_path.stem/f'r{number:02}'
                run.mkdir(parents=True)
                path = run/'layout.json'
                path.write_bytes(raw)
                row = dict(archive=archive_path.name, round=number, local_wall=wall,
                           platform_wall=detail['炸墙分'], platform_difficulty=detail['炸弹分(校准)'],
                           layout_sha256=blue.digest(path), status='geometry_unconfirmed')
                if abs(wall-detail['炸墙分']) < 0.006:
                    result = red.execute_round(dict(platform_layout_only=True, map='Map07', blue_variant=variant,
                        round=number, seed=20261103, count=count, drones=5, run_dir=str(run),
                        layout=str(path)), 600, env, None)
                    row.update({k: result.get(k) for k in ('status','walltime_limited','disposed_count','blue_score')})
                rows.append(row)
                red.write_json(output/'calibration.json', rows)
                print(row, flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(), *sys.argv[2:])
