"""Attribute archived placements from their physical positions, and measure track windows."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def trace_audit(replay, xy, masks, weight):
    bombs, drones = replay['bomb_states'], replay['drone_states']
    track = np.stack((drones['x'], drones['y']), axis=-1)
    if not drones['active'].all() or not bombs['active'][-1].all():
        raise ValueError('Cannot assume constant drone slots or persistent bombs')
    recorded = np.column_stack((bombs['x'][-1], bombs['y'][-1]))
    costs = np.linalg.norm(xy[:, None] - recorded, axis=2)
    labels, slots = linear_sum_assignment(costs)
    if len(labels) != len(xy) or costs[labels, slots].max() > 2:
        raise ValueError('Exported and recorded positions disagree')
    times = np.argmax(bombs['active'][:, slots], axis=0)
    owners = []
    records = []
    covered = np.zeros_like(masks[0])
    for i in np.argsort(times, kind='stable'):
        t = times[i]
        born = np.array([bombs['x'][t, slots[i]], bombs['y'][t, slots[i]]])
        d = np.linalg.norm(track[t] - born, axis=1)
        near, runner = np.argsort(d)[:2]
        if d[near] > 12 or d[runner]-d[near] < 12:
            raise ValueError(f'Ambiguous owner at frame {t+1}: {d}')
        owners.append((i, near))
        records.append(dict(bomb=int(i+1), frame=int(t+1), owner=int(near), gap=float(d[runner]-d[near]),
                            distance=float(d[near]), added=float(weight[masks[i] & ~covered].sum()),
                            gross=float(weight[masks[i]].sum())))
        covered |= masks[i]
    order = sorted(owners)
    collective = []
    by_drone = []
    last = int(times.max())
    pairs = []
    for i, owner in order:
        for j, other in order[:i]:
            overlap = float(weight[masks[i] & masks[j]].sum())
            if overlap:
                pairs.append(dict(pair=[int(j+1),int(i+1)], same=bool(owner==other), overlap=overlap))
    for owner in range(track.shape[1]):
        ids = [i for i,o in order if o == owner]
        union = np.any(masks[ids], axis=0)
        collective.append(union)
        path = track[:,owner]
        # Sliding windows summed by union, so overlapping windows never double count.
        flagged = np.zeros(last+1, dtype=bool)
        for t in range(0, last+2-80, 10):
            window = path[t:t+80]
            travel = np.linalg.norm(np.diff(window,axis=0),axis=1).sum()
            if travel >= 80 and np.linalg.norm(window[-1]-window[0]) <= 20 and np.ptp(window,axis=0).max() <= 70:
                flagged[t:t+80] = True
        ends = np.linalg.norm(path[last:]-path[0],axis=1)
        by_drone.append(dict(owner=owner, bombs=[int(i+1) for i in ids], union=float(weight[union].sum()),
                            loop_frames=int(flagged.sum()), tail_path=float(np.linalg.norm(
                                np.diff(path[last:],axis=0),axis=1).sum()),
                            tail_near_home=int((ends<60).sum()), tail_end_distance=float(ends[-1])))
    gross=sum(r['gross'] for r in records)
    team=float(weight[covered].sum())
    footprint=sum(d['union'] for d in by_drone)
    assert sum(r['added'] for r in records)==team
    return dict(frames=len(track), tail=len(track)-1-last, gross=gross, team=team,
                same_drone_duplicate=gross-footprint, cross_drone_duplicate=footprint-team,
                by_drone=by_drone, bombs=records, overlap_pairs=pairs)
