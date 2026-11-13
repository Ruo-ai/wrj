"""Read-only comparison of two frozen red final maps; never sensor replay.

The native wall truth is constructed only in this offline tool. Changes in final
pixels cannot identify which scan, erasure, agent or postprocess caused them.
"""
import argparse
from pathlib import Path
import numpy as np
from swarm_rescue.solutions.devtools.optimization.evaluation.runner import read, write, receipt, file_hash
from swarm_rescue.solutions.devtools.optimization.evaluation.replay import validate_replay_sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempts', nargs=2, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from swarm_rescue.solutions.devtools.evaluate_red import configure_ray_backend
    from swarm_rescue.solutions.devtools.optimization.evaluation.scenarios import resolve_map
    from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
    from swarm_rescue.simulation.elements.sensor_disablers import ZoneType
    from swarm_rescue.simulation.reporting.place_score_manager import walls_and_boundary_grid
    from swarm_rescue.tools.map_exploration_scoring import scorer_for_walltime, load_prediction_grid
    requests = []; results = []; predictions = []
    for path in args.attempts:
        path = path.resolve()
        batch = next(q for q in path.parents if (q/'plan.json').is_file())
        plan = read(batch/'plan.json'); validate_replay_sources(plan['source_sha256'])
        if read(path/'integrity.json') != receipt(path, file_hash(batch/'plan.json')):
            raise ValueError('Attempt integrity mismatch')
        requests.append(read(path/'request.json')); results.append(read(path/'result.json'))
        predictions.append(load_prediction_grid(next((path/'artifacts').glob('*_exploration.json'))) > 0)
    for key in ('map', 'seed', 'round', 'zones', 'map_file'):
        if requests[0][key] != requests[1][key]: raise ValueError('Pair differs: '+key)
    configure_ray_backend('cpu')
    q = requests[0]
    world = resolve_map(q)(DroneMotionless, tuple(ZoneType[z] for z in q['zones']), 5)
    try: truth = walls_and_boundary_grid(world.playground)[0]
    finally: world.playground.window.close()
    scorer = scorer_for_walltime(180)
    surface = scorer.split_regions(truth)[0] > 0
    far = scorer.distance_field(truth) > 10
    records = []
    for pred, result, request in zip(predictions, results, requests):
        score = scorer.score(truth, pred, result['elapsed_walltime'])
        if abs(score.s_map*100-result['phase1']['score_map']) > 1e-5:
            raise ValueError('Official score recomputation mismatch')
        missed = surface & ~pred; distance = scorer.distance_field(pred)
        records.append(dict(controller=request['controller_selector'], map_score=100*score.s_map,
            credit=100*score.credit, penalty=100*score.penalty,
            hits=score.wall.hits, misses=int(missed.sum()), charged_pixels=score.charged_pixels,
            miss_distance_bins={f'{lo}-{hi}':int((missed & (distance > lo) & (distance <= hi)).sum())
                                for lo, hi in ((0,3),(3,10),(10,30),(30,60),(60,100000))}))
    a, b = predictions; removed = a & ~b; added = b & ~a
    delta = dict(removed_predictions=int(removed.sum()), added_predictions=int(added.sum()),
        removed_true_surface=int((removed & surface).sum()), added_true_surface=int((added & surface).sum()),
        removed_charged_pixels=int((removed & far).sum()), added_charged_pixels=int((added & far).sum()))
    value = dict(scope='final-map pixel comparison, not full sensor replay or causal erasure attribution',
        map=q['map'], round=q['round'], seed=q['seed'], attempts=[str(v.resolve()) for v in args.attempts],
        records=records, delta=delta)
    write(args.output/'audit.json', value)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    image = np.full((*truth.shape, 3), .97)
    image[truth > 0] = (.5, .5, .5)
    image[removed & ~surface] = (.65, .75, .9)
    image[added & ~surface] = (.75, .25, .7)
    image[removed & surface] = (1., .45, 0.)
    image[added & surface] = (0., .65, .2)
    fig, ax = plt.subplots(figsize=(12, 8)); ax.imshow(image); ax.axis('off')
    ax.set_title(f'{q["map"]}: final A3 to A4 pixel delta (offline)\n'
                 f'Orange: lost true hits {delta["removed_true_surface"]}; green: gained true hits {delta["added_true_surface"]}\n'
                 'Light blue: removed other predictions; purple: added other predictions')
    fig.tight_layout(); fig.savefig(args.output/'delta.png', dpi=120); plt.close(fig)
    import json
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
