"""Verify a saved receipt and compare three final rasters on its delivered source versions."""

import argparse
from collections import defaultdict
import importlib
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from . import red_exploration22_batch as prior
from .optimization.evaluation import runner
from .red_exploration22_archive import load_archived_controller, archive_runtime_identity
from ..my_drone_red_lines11 import WallLineMapper
from ..my_drone_red_map23 import submission_pixels
from swarm_rescue.tools.map_exploration_scoring import MapExplorationScorer


CHECKPOINTS = (1350, 2700, 4050, 5400)


def score_panels(attempt, shape):
    h, w = shape
    (path,) = (attempt / 'artifacts').glob('*_exploration_diff.png')
    with Image.open(path) as img:
        colors = np.asarray(img.convert('RGB'))
    if colors.shape[:2] != (h + 128, w * 3 + 4):
        raise ValueError('Unexpected official panel geometry')
    grids = []
    for left in (0, w + 2):
        panel = colors[24:24 + h, left:left + w]
        if not np.all((panel == 40).all(2) | (panel == 255).all(2)):
            raise ValueError('Non-binary official score panel')
        grids.append((panel[:, :, 0] == 40).astype(np.uint8))
    return grids


def visits(attempt, shape, end, out):
    (path,) = (attempt / 'artifacts/recordings').glob('*.npz')
    with np.load(path, allow_pickle=False) as z:
        states = z['drone_states']
    # GUI step 1 warms up; the first captured frame is GUI step 2.
    if len(states) + 1 != end:
        raise ValueError('Unexpected trajectory/GUI time alignment')
    h, w = shape
    cell = 40
    nx, ny = (w + cell - 1) // cell, (h + cell - 1) // cell
    x, y = states['x'].astype(float) + w / 2, h / 2 - states['y'].astype(float)
    live = states['active']
    if not np.all((x[live] >= 0) & (x[live] < w) & (y[live] >= 0) & (y[live] < h)):
        raise ValueError('Active centre outside the map')
    steps = np.arange(2, end + 1)
    cell_id = (y // cell).astype(int) * nx + (x // cell).astype(int)
    first = np.full(nx * ny, end + 1, int)
    np.minimum.at(first, cell_id[live], np.broadcast_to(steps[:, None], live.shape)[live])
    counts = np.bincount(first, minlength=end + 2).cumsum()[steps]
    np.savez_compressed(out / 'visits.npz', steps=steps, counts=counts,
                        first_step=np.where(first <= end, first, 0).reshape(ny, nx))
    return dict(cell_pixels=cell, total_cells=nx * ny, final=int(counts[-1]), end_step=end,
                checkpoints={str(s): int(counts[s - 2]) if s <= end else None for s in CHECKPOINTS},
                trajectory_sha256=prior.digest(path))


def semantic_objects(raw, names):
    kinds = {v: SimpleNamespace(name=k) for k, v in names.items()}
    groups = defaultdict(list)
    for ident, tick, code, distance, angle in raw:
        groups[int(ident), int(tick)].append(SimpleNamespace(
            entity_type=kinds[int(code)], distance=float(distance), angle=float(angle)))
    return groups


def replay_sources(cls, size, sequences, tapes, objects, captured=None, witness_cells=None):
    sources, witnesses = {}, []
    for ident, tick in sequences.items():
        mapper = cls(size)
        rows, angles = tapes['drone_' + ident], tapes['angles_' + ident]
        if tick < 1 or tick > len(rows) or not np.array_equal(rows[:, 0], np.arange(1, len(rows) + 1)):
            raise ValueError(f'Missing or misaligned samples for source {ident}, version {tick}')
        observe = witness_cells is not None
        for row in rows[:tick]:
            step = int(row[0])
            mapper.update(row[5:7].copy() if row[1] else None, float(row[7]) if row[2] else None,
                row[8:11].copy() if row[3] else None, row[11:].copy() if row[4] else None,
                angles, objects.get((int(ident), step), ()))
            if observe and step <= 400:
                # Check every local tick until the first final erroneous cell is visible.
                marks = mapper.mask() & witness_cells & captured[ident]
                if marks.any():
                    y, x = np.argwhere(marks)[0]
                    witnesses.append(dict(source=int(ident), version=tick, first_local_tick=step,
                                          cell_xy=[int(x), int(y)]))
                    observe = False
        sources[ident] = mapper.mask()
        if captured is not None and not np.array_equal(sources[ident], captured[ident]):
            raise ValueError(f'Archived replay differs from received source {ident} v{tick}')
    return mapper.to_pixels(np.logical_or.reduce(list(sources.values()))), witnesses


def compare(phase, out, historical=False):
    cv2.setNumThreads(1)
    plan_file = next(p / 'plan.json' for p in phase.parents if (p / 'plan.json').is_file())
    result, attempt = runner.load_done(phase, prior.digest(plan_file))
    if not result or not runner.phase_valid(result, 'red'):
        raise ValueError('A valid red receipt is required')
    out.mkdir(parents=True, exist_ok=False)
    shape = tuple(result['expected_shape'])
    size = shape[::-1]
    truth, submitted = score_panels(attempt, shape)
    (grid_file,) = (attempt / 'artifacts').glob('*_exploration.json')
    if not np.array_equal(np.asarray(runner.read(grid_file)['grid']), submitted):
        raise ValueError('Submitted JSON differs from the official score panel')
    meta = runner.read(attempt / 'line_diagnostics.json')
    delivery = runner.read(attempt / 'exploration_diagnostics.json')
    if not meta['mapper_inputs_complete'] or meta['errors'] or delivery['errors']:
        raise ValueError('Incomplete observations or source masks')
    submission, = delivery['submissions']
    sources = submission['sources']
    with np.load(attempt / 'exploration_stages.npz') as z:
        captured = {s: z[f"submitter_{submission['drone']}_source_{s}"] for s in sources}
    tape_path = attempt / 'mapper_legal_inputs.npz'
    with np.load(tape_path) as z:
        tapes = {k: z[k] for k in z.files}
    objects = semantic_objects(tapes['semantic'], meta['kinds'])
    if prior.digest(prior.ARCHIVE) != prior.EXPECTED_ARCHIVE_SHA256:
        raise ValueError('Unexpected reference archive')
    controller = load_archived_controller(prior.ARCHIVE)
    arch = importlib.import_module(controller.__module__).RepairMapper17
    official = MapExplorationScorer()
    surface = official.split_regions(truth)[0].astype(bool)
    distance = official.distance_field(truth)
    pixels = arch(size).to_pixels(np.logical_or.reduce(list(captured.values())))
    candidate = submission_pixels(pixels)
    if not np.array_equal(pixels if historical else candidate, submitted):
        raise ValueError('Rendering the received masks differs from the actual submission')
    # Trace an isolated, charged spur from the historical final grid into one actual source.
    witness_cells = None
    if historical:
        only_one = np.sum(list(captured.values()), axis=0) == 1
        repairs = (pixels != 0) & (candidate == 0) & (distance > official.config.dead_zone)
        ry, rx = np.nonzero(repairs)
        witness_cells = np.zeros_like(only_one)
        witness_cells[ry // 3, rx // 3] = True
        witness_cells &= only_one
    archived, witnesses = replay_sources(arch, size, sources, tapes, objects, captured, witness_cells)
    if not np.array_equal(archived, pixels):
        raise ValueError('Archive legal-tape replay differs from the delivered map')
    if witnesses:
        witness = min(witnesses, key=lambda r: (r['first_local_tick'], r['source']))
        x, y = witness['cell_xy']
        sy, sx = np.nonzero(repairs[3*y:3*y+3, 3*x:3*x+3])
        xx, yy = int(3*x+sx[0]), int(3*y+sy[0])
        witness.update(pixel_xy=[xx, yy], distance_to_wall=float(distance[yy, xx]),
                       baseline_pixel=int(pixels[yy, xx]), candidate_pixel=int(candidate[yy, xx]))
    else:
        witness = None
    rl11, _ = replay_sources(WallLineMapper, size, sources, tapes, objects)
    maps = {'RL11': rl11, 'RL17-R': archived, 'RL23': candidate}
    metrics, deltas = {}, {}
    for name, grid in maps.items():
        z = official.score(truth, grid, result['elapsed_walltime'])
        metrics[name] = dict(map_score=100*z.s_map, hits=z.wall.hits, missed=z.wall.misses,
            charged=z.charged_pixels, penalty=100*z.penalty, credit=100*z.credit)
        if name != 'RL23':
            new = (candidate > 0) & (grid == 0)
            lost = (candidate == 0) & (grid > 0)
            deltas[name] = dict(new_hits=int((new & surface).sum()), lost_hits=int((lost & surface).sum()),
                penalty=100*official.score(truth, candidate, 0).penalty-metrics[name]['penalty'])
    live_version = 'RL17-R' if historical else 'RL23'
    live_score = result['phase1']['score_map']
    if abs(live_score - metrics[live_version]['map_score']) > 1e-7:
        raise ValueError('Score differs from live receipt')
    recorded_visits = visits(attempt, shape, result['elapsed_steps'], out)
    sensor = None
    if (attempt / 'wall_returns.npz').is_file():
        with np.load(attempt / 'wall_returns.npz') as z:
            seen = surface & (z['first_tick'] > 0)
            sensor = dict(scans=int(z['scan_count']), direct_surface=int(seen.sum()),
                          also_drawn=int((seen & (candidate > 0)).sum()),
                          observed_missed=int((seen & (candidate == 0)).sum()))
    np.savez_compressed(out / 'maps.npz', truth=truth, **maps)
    report = dict(map=result['map'], seed=result['seed'], attempt=str(attempt),
        plan_sha256=prior.digest(plan_file), receipt_sha256=runner.read(phase / 'done.json')['receipt_sha256'],
        live_version=live_version, live_score=live_score, submission=submission,
        gui_end_step=result['elapsed_steps'], source_masks_exact=True, final_pixels_exact=True,
        input_sha256=prior.digest(tape_path), stages_sha256=prior.digest(attempt / 'exploration_stages.npz'),
        archive=archive_runtime_identity(controller), witness=witness, metrics=metrics, deltas=deltas,
        scored_surface=int(surface.sum()), visits=recorded_visits, lidar=sensor)
    prior.write_new(out / 'comparison.json', report)
    print(result['map'], metrics, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--historical', action='store_true')
    args = p.parse_args()
    compare(args.phase, args.output, args.historical)
