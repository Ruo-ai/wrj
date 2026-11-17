"""RL18-A: deterministic, read-only replay and recovery attribution.

Only this offline tool sees truth. No simulator, flight, radio or policy runs.
Map names/ROIs below select diagnostics, never online behavior. Output is new.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from swarm_rescue.solutions.my_drone_red_lines11 import WallLineMapper
from swarm_rescue.solutions.my_drone_red_repair17 import RepairMapper17
from swarm_rescue.solutions.my_drone_red_mapping import anchor_frames, free_points, wall_points
from swarm_rescue.solutions.semifinal_localization import transform, wrap
from swarm_rescue.solutions.devtools.red_p2l17_replay import ReplayProcess, summary
from swarm_rescue.tools.map_exploration_scoring import MapExplorationScorer
from swarm_rescue.solutions.devtools.red_stopturn_match_report import COMPARISON_HEADER_HEIGHT


EVENTS = {'Map08': [(3, 1755), (4, 3276)], 'Map09': [(0, 1377), (3, 2793)]}
# Approximate screenshot extents, diagnostic only; not identical to disabler bounds.
ROI = {'Map08': (950, 730, 1655, 1117), 'Map09': (6, 6, 1694, 325)}
CLASSES = {'rl11_lines': WallLineMapper, 'rl17_repair': RepairMapper17}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def legal_arguments(row, angles, objects):
    return (row[5:7].copy() if row[1] else None,
            float(row[7]) if row[2] else None,
            row[8:11].copy() if row[3] else None,
            row[11:].copy() if row[4] else None, angles, objects)


def semantic_objects(inputs, meta):
    names = {v: k for k, v in meta['kinds'].items()}
    result = defaultdict(list)
    for owner, tick, kind, distance, angle in inputs['semantic']:
        result[int(owner), int(tick)].append(SimpleNamespace(
            entity_type=SimpleNamespace(name=names[int(kind)]), distance=distance, angle=angle))
    return result


class Metrics:
    """Cache truth-only fields. Reproduce official formula; do not change scorer."""
    def __init__(self, truth, roi):
        self.truth = truth.astype(bool)
        self.scorer = MapExplorationScorer()
        self.surface = self.scorer.split_regions(truth)[0].astype(bool)
        self.distance = self.scorer.distance_field(truth)
        self.far = self.distance > self.scorer.config.dead_zone
        self.charges = self.scorer.penalty_curve(self.distance)
        l, t, r, b = roi
        self.roi = np.s_[t:b+1, l:r+1]

    def count(self, pred):
        p = pred.astype(bool)
        hits = int((p & self.surface).sum())
        credit = hits / int(self.surface.sum())
        penalty = self.scorer.config.penalty_weight * float(self.charges[p].sum()) / int(self.surface.sum())
        return dict(hits=hits, charged=int((p & self.far).sum()),
                    credit=credit*100, penalty=penalty*100,
                    map_score=float(np.clip(credit-penalty, 0, 1)*100),
                    roi_hits=int((p & self.surface)[self.roi].sum()),
                    roi_charged=int((p & self.far)[self.roi].sum()))

    def delta(self, before, after):
        a, b = before.astype(bool), after.astype(bool)
        return dict(added_hits=int((b & ~a & self.surface).sum()),
                    removed_hits=int((a & ~b & self.surface).sum()),
                    added_charged=int((b & ~a & self.far).sum()),
                    removed_charged=int((a & ~b & self.far).sum()))

    def rgb(self, pred):
        p = pred.astype(bool)
        rgb = np.full((*p.shape, 3), 247, np.uint8)
        rgb[self.truth] = (170, 175, 180)
        rgb[self.surface & ~p] = (65, 120, 210)
        rgb[p & ~self.surface] = (225, 190, 65)
        rgb[p & self.far] = (200, 55, 65)
        rgb[p & self.surface] = (35, 150, 90)
        return rgb


def evidence_grid(mapper, frames, initial=None, fixed_weight=None):
    grid = np.zeros_like(mapper.hits) if initial is None else initial.copy()
    for _, points, pos, heading, weight, free in frames:
        weight = weight if fixed_weight is None else fixed_weight
        mapper._erase(grid, transform(free, pos, heading), weight)
        mapper._insert(grid, transform(points, pos, heading), weight)
    return grid


def pack_frames(frames):
    arrays = {'meta': np.array([[f[0], *f[2], f[3], f[4]] for f in frames]).reshape(-1, 5)}
    for label, index in [('points', 1), ('free', 5)]:
        sizes = [len(f[index]) for f in frames]
        arrays[label+'_offsets'] = np.r_[0, np.cumsum(sizes)]
        arrays[label] = np.concatenate([f[index] for f in frames]) if frames else np.empty((0, 2))
    return arrays


def recovery_layers(mapper, args):
    gps, compass, odom, _, _, _ = args
    assert mapper.pending and mapper.missing and gps is not None and compass is not None
    pos, heading = mapper.position.copy(), mapper.heading
    if odom is not None and np.isfinite(odom).all():
        pos += odom[0]*np.array([np.cos(heading+odom[1]), np.sin(heading+odom[1])])
        heading = wrap(heading+odom[2])
    corrected = anchor_frames(mapper.pending, mapper.start, pos, heading, gps, compass, mapper.step+1)
    pending = evidence_grid(mapper, mapper.pending)
    promoted = evidence_grid(mapper, mapper.pending, mapper.hits, .8)
    anchored = evidence_grid(mapper, corrected, mapper.hits, .8)
    return dict(hits_before=mapper.hits.copy(), pending_before=pending,
                mask_before=(mapper.hits >= 2.4) | (pending >= 2.4),
                promoted_unanchored=promoted, anchored_before_current=anchored,
                predicted_pose=np.r_[pos, heading], measured_pose=np.r_[gps, compass]), corrected


def radio_states(radio, holder, ticks):
    events = [e for e in radio if e['holder'] == holder and e['event'] in ('local_publish', 'received')]
    assert all(events[i]['step'] <= events[i+1]['step'] for i in range(len(events)-1))
    result, versions, index = {}, {}, 0
    for tick in sorted(ticks):
        while index < len(events) and events[index]['step'] <= tick:
            e = events[index]
            assert e['version'] <= e['step']
            versions[int(e['source'])] = int(e['version'])
            index += 1
        result[tick] = dict(versions)
    return result


def audit(run, out, process):
    out.mkdir(parents=True, exist_ok=False)
    started, cpu = time.perf_counter(), time.process_time()
    request, result = read(run/'request.json'), read(run/'result.json')
    name, controller = request['map'], request['controller_selector']
    meta, diag, detail = [read(run/f) for f in ('line_diagnostics.json', 'exploration_diagnostics.json', 'p2l17_diagnostics.json')]
    assert meta['mapper_inputs_complete'] and not meta['errors'] and not diag['errors'] and not detail['errors']
    inputs = load_npz(run/'mapper_legal_inputs.npz')
    stages, recorded = load_npz(run/'exploration_stages.npz'), load_npz(run/'p2l17_diagnostics.npz')
    objects = semantic_objects(inputs, meta)
    h, w = result['expected_shape']
    png = np.array(Image.open(next((run/'artifacts').glob('*_exploration_diff.png'))).convert('RGB'))
    truth = np.all(png[COMPARISON_HEADER_HEIGHT:COMPARISON_HEADER_HEIGHT+h, :w] == 40, axis=2).astype(np.uint8)
    metric = Metrics(truth, ROI[name]); np.save(out/'truth.npy', truth)
    submission = diag['submissions'][0]
    target_ticks = {submission['step']}
    targets = defaultdict(set)
    for owner, recovery in EVENTS[name]:
        data = inputs[f'drone_{owner}']; good = data[:, 1].astype(bool) & data[:, 2].astype(bool)
        assert good[recovery-1] and not good[recovery-2]
        last_good = int(np.flatnonzero(good[:recovery-1])[-1]+1)
        targets[owner].update(range(((last_good+3)//4)*4, recovery, 40))
        targets[owner].update(range(recovery-32, recovery+33, 4))
        targets[owner].update((recovery-1, recovery))
        target_ticks.update((recovery-32, recovery-1, recovery, recovery+1, recovery+8, recovery+32))
        receipts = [e['step'] for e in detail['radio'] if e['event'] in ('local_publish', 'received') and
                    e['holder']==submission['drone'] and e['source']==owner and e['version']>=recovery]
        if receipts: target_ticks.add(min(receipts))
    states = radio_states(detail['radio'], submission['drone'], target_ticks)
    needed = defaultdict(set)
    for state in states.values():
        for owner, version in state.items(): needed[owner].add(version)
    for owner, info in diag['drones'].items():
        for event in info['history']: needed[int(owner)].add(event['published_sequence'])
        needed[int(owner)].add(submission['sources'][owner])
    published = {(e['source'], e['version']): e['sha256'] for e in detail['radio'] if e['event']=='local_publish'}
    bad_radio = [e for e in detail['radio'] if e['event']=='received' and
                 (published.get((e['source'], e['version'])) != e['sha256'] or e['version']>e['step'])]
    assert not bad_radio
    snapshots, traces, events, exact, arrays = {}, [], [], {}, {}
    latencies = []; recovery_set = set(EVENTS[name]); mapper_seconds = 0.
    for ident in sorted(diag['drones'], key=int):
        owner = int(ident); mapper = CLASSES[controller]((w, h))
        data, angles = inputs['drone_'+ident], inputs['angles_'+ident]
        assert np.array_equal(data[:, 0], np.arange(1, len(data)+1))
        poses = []; local_trace = []
        for row in data:
            tick = int(row[0]); args = legal_arguments(row, angles, objects[owner, tick])
            recovery = (owner, tick) in recovery_set
            if recovery:
                layers, corrected = recovery_layers(mapper, args)
                cache = {**layers, **{'pending_'+k: v for k, v in pack_frames(mapper.pending).items()},
                         **{'corrected_'+k: v for k, v in pack_frames(corrected).items()}}
                event = dict(owner=owner, step=tick, start=mapper.start[0], pending_before=len(mapper.pending))
            t0 = time.perf_counter(); mapper.update(*args); elapsed = time.perf_counter()-t0
            latencies.append(elapsed); mapper_seconds += elapsed
            poses.append([*mapper.position, mapper.heading])
            if recovery:
                expected = layers['anchored_before_current'].copy()
                if tick%4 == 0:
                    mapper._erase(expected, transform(free_points(args[3], args[4]), mapper.position, mapper.heading), 1.)
                    mapper._insert(expected, transform(wall_points(args[3], args[4], args[5]), mapper.position, mapper.heading), 1.)
                assert np.array_equal(expected, mapper.hits), 'reprojection state differs from original code'
                assert not mapper.pending
                cache['hits_after'] = mapper.hits.copy(); cache['mask_after'] = mapper.mask()
                np.savez_compressed(out/f'recovery_d{owner}_t{tick}.npz', **cache)
                maps = dict(before=mapper.to_pixels(layers['mask_before']),
                            unanchored_commit=mapper.to_pixels(layers['promoted_unanchored']>=2.4),
                            anchored_commit=mapper.to_pixels(layers['anchored_before_current']>=2.4),
                            after=mapper.to_pixels(mapper.mask()))
                event.update(pending_after=0, state_exact=True,
                             stages={k: metric.count(v) for k, v in maps.items()},
                             deltas={a+'_to_'+b: metric.delta(maps[a], maps[b]) for a, b in
                                     [('before','unanchored_commit'), ('unanchored_commit','anchored_commit'), ('anchored_commit','after'), ('before','after')]},
                             caveat='Promotion is a counterfactual .8 commit without pose correction; deltas are ordered, not independent causal shares.')
                events.append(event)
                for key, pred in maps.items(): Image.fromarray(metric.rgb(pred)).save(out/f'd{owner}_t{tick}_{key}.png')
            if tick in targets[owner]:
                provisional = evidence_grid(mapper, mapper.pending)
                mask = (mapper.hits>=2.4) | (provisional>=2.4)
                local_trace.append(dict(step=tick, missing=mapper.missing, pending=len(mapper.pending),
                    hits_cells=int((mapper.hits>=2.4).sum()), pending_cells=int((provisional>=2.4).sum()),
                    **metric.count(mapper.to_pixels(mask))))
                arrays[f'local_d{owner}_t{tick}'] = mask
            if tick in needed[owner]: snapshots[owner, tick] = mapper.mask().copy()
        poses = np.array(poses); arrays['poses_'+ident] = poses
        assert np.array_equal(recorded['pose_'+ident][:, 0], data[:, 0])
        pose_delta = float(np.max(np.abs(poses-recorded['pose_'+ident][:, 4:7])))
        histories = {str(e['step']): bool(np.array_equal(snapshots[owner, e['published_sequence']], stages[f"published_{owner}_{e['step']}"])) for e in diag['drones'][ident]['history']}
        exact[ident] = dict(pose_max_abs=pose_delta, pose_exact=pose_delta < 1e-9,
            final_local=bool(np.array_equal(mapper.mask(), stages['local_final_'+ident])), published=histories,
            received=bool(np.array_equal(snapshots[owner, submission['sources'][ident]], stages[f"submitter_{submission['drone']}_source_{ident}"])))
        traces.append(dict(owner=owner, rows=local_trace))
    actual = np.asarray(read(next((run/'artifacts').glob('*_exploration.json')))['grid'], np.uint8)
    merged_rows = []
    for tick, state in states.items():
        masks = [snapshots[o, v] for o, v in state.items()]
        merged = np.logical_or.reduce(masks); pixels = mapper.to_pixels(merged)
        arrays[f'merged_t{tick}'] = merged
        merged_rows.append(dict(step=tick, sources=state, **metric.count(pixels)))
        if tick == submission['step']:
            final = pixels; count = np.stack(masks).sum(axis=0).astype(np.uint8)
            arrays['source_count'] = count
            source_rows = []
            for owner, version in state.items():
                other = np.logical_or.reduce([snapshots[o, v] for o, v in state.items() if o != owner])
                alone = mapper.to_pixels(snapshots[owner, version]); without = mapper.to_pixels(other)
                source_rows.append(dict(owner=owner, version=version, standalone=metric.count(alone),
                    marginal_after_postprocessing=metric.delta(without, final)))
                arrays[f'source_{owner}_v{version}'] = snapshots[owner, version]
    final_exact = bool(np.array_equal(final, actual))
    passed = final_exact and all(x['pose_exact'] and x['final_local'] and x['received'] and all(x['published'].values()) for x in exact.values())
    formula = metric.scorer.score(truth, final, result['elapsed_walltime'])
    assert abs(formula.s_map*100-metric.count(final)['map_score']) < 1e-6
    score_exact = abs(formula.s_map*100-result['phase1']['score_map']) < 1e-6
    passed = passed and score_exact
    np.save(out/'pixels.npy', final); np.savez_compressed(out/'maps_poses.npz', **arrays)
    Image.fromarray(metric.rgb(final)).save(out/'mapping.png')
    report = dict(schema='rl18-recovery-audit-v1', source=str(run.resolve()), request=request,
        source_hashes={f: digest(run/f) for f in ('mapper_legal_inputs.npz','p2l17_diagnostics.npz','p2l17_diagnostics.json','exploration_stages.npz','result.json')},
        source_code={str(p): digest(p) for p in [Path(__file__), *[Path(__file__).parents[1]/n for n in ('my_drone_red_lines11.py','my_drone_red_repair17.py','my_drone_red_p2l17.py','my_drone_red_mapping.py','semifinal_localization.py')]]},
        exact=exact, final_exact=final_exact, score_exact=score_exact, valid=passed,
        final=metric.count(final), source_contributions=source_rows, recovery=events,
        local_growth=traces, holder_actual_versions=merged_rows, roi=list(ROI[name]),
        resources=dict(wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
                       update_seconds=summary(latencies),mapper_seconds=mapper_seconds,
                       peak_rss_bytes=process.memory(),affinity=process.affinity),
        limitations=['Intermediate regenerated masks are supported by every-step pose and all saved mask checks; original packet contains nav+mapping bytes but full packet bytes were not saved.',
                     'Holder states follow accepted version log, no future/final-local oracle merge.',
                     'Reliable free-space evidence, uncertainty and per-ray provenance were not transmitted. Stage D cross-agent free contradiction is unavailable; unknown is not free.',
                     'Local scoring and counterfactual promotion are diagnostics, not additional official matches.'])
    write(out/'results.json', report)
    print(json.dumps(dict(map=name, controller=controller, valid=passed, final=report['final'], resources=report['resources'])), flush=True)
    if not passed: raise RuntimeError('Baseline did not reproduce; stop before shadow work')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    process=ReplayProcess(); cv2.setNumThreads(1); reports=[]; input_checks={}
    for name in EVENTS:
        runs={c: args.batch/f'runs/{name}_r01/{c}/red/attempt_001' for c in CLASSES}
        a,b=[load_npz(r/'mapper_legal_inputs.npz') for r in runs.values()]
        same=set(a)==set(b) and all(np.array_equal(a[k],b[k],equal_nan=True) for k in a)
        input_checks[name]=same
        if not same: raise RuntimeError('Pair legal input mismatch')
        del a,b
        for controller,run in runs.items(): reports.append(audit(run,args.output/f'{name}_{controller}',process))
    write(args.output/'summary.json', dict(valid=all(r['valid'] for r in reports),same_inputs=input_checks,
          rows=[dict(map=r['request']['map'],controller=r['request']['controller_selector'],**r['final']) for r in reports]))


if __name__=='__main__': main()
