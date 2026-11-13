"""Expand archived L1 decisions offline; require exact agreement with the live target."""
import base64
import gzip
import json
from pathlib import Path
import sys
import zlib

import numpy as np
from swarm_rescue.solutions import my_drone_red_advanced2 as a
from swarm_rescue.solutions.devtools.report_advanced5 import events


def unpack(blob):
    shape, encoded = blob
    bits = np.frombuffer(zlib.decompress(base64.b64decode(encoded)), np.uint8)
    return np.unpackbits(bits, count=int(np.prod(shape))).reshape(shape).astype(bool)


def replay(e):
    g = a._M09_ExplorationGrid(e['size'])
    g.free = unpack(e['free'])
    g.obstacle = 2 * unpack(e['blocked'])
    walls = unpack(e['walls'])
    args = (walls, e['cell'], e['size'])
    f = a._M16_CoverageField(*args, e['placed'], e['claims'], e['radius'])
    raw = a._M16_CoverageField(*args, [], [], e['radius'])
    pos = np.asarray(e['position'])
    center = None if e['center'] is None else np.asarray(e['center'])
    actual = a._M17_choose_placement(g, pos, f, center)
    target = actual.target.tolist() if actual is not None else None
    if target != e['selected']:
        raise ValueError(f'Live/replay mismatch at {e["drone"]}, {e["step"]}')
    distances, _ = a._M16_reachable(g, pos, 50)
    transport = a._M17_transport_costs(g, center)
    base = a._M16_choose_placement(g, pos, f)
    ranked = []
    for c, steps in distances.items():
        p = g.point(c)
        gain = f.sampled_value(p)
        near = any(np.linalg.norm(p-b)<65 for b in f.placed)
        ranked.append((gain/(1+.015*steps), c, steps, gain, near))
    eligible = sorted((r for r in ranked if not r[-1] and r[3]>0), reverse=True)
    ranks = {r[1]:i for i,r in enumerate(eligible,1)}
    coarse = []
    for _, c, steps, gain, _ in ranked:
        if base and c in transport and gain >= .9*base.gain:
            d = a._M17_difficulty(g.point(c), transport[c], center, e['size'])
            coarse.append((gain/(1+.015*steps)*(1+.35*d), c))
    refined = set()
    for _, c in sorted(coarse, reverse=True):
        if all(np.linalg.norm(g.point(c)-g.point(other)) >= .5*f.radius for other in refined):
            refined.add(c)
            if len(refined)==12:
                break
    rows = []
    for score,c,steps,gain,near in ranked:
        p = g.point(c)
        rank = ranks.get(c)
        base_refined = rank is not None and rank<=6
        cell_reason = ('refined' if base_refined else 'near_placed' if near else
                       'no_gain' if rank is None else 'outside_top6')
        disk = f.disk(p)
        rows.append(dict(position=p.tolist(), rank=rank, coarse_score=score,
                         skeleton_coverage=raw.value(p), observed_area_cells=int(walls[disk].sum()),
                         peer_discount=float(f.unclaimed[disk].sum()-f.value(p)),
                         marginal_gain=gain, route_distance=steps*g.CELL,
                         base_reason=cell_reason, difficulty_refined=c in refined))
        if not base_refined and c not in refined:
            continue
        for dx in (-8,0,8):
            for dy in (-8,0,8):
                q=p+[dx,dy]
                valid = abs(q[0])<g.width/2 and abs(q[1])<g.height/2 and g.line_clear(p,q)
                safe = all(np.linalg.norm(q-b)>=65 for b in f.placed)
                v=f.value(q)
                value=v/(1+.015*(steps+np.linalg.norm(q-p)/g.CELL))
                diff=a._M17_difficulty(q,transport.get(c),center,e['size'])
                floor_ok = base is not None and v>=.9*base.gain and value>=.9*base.gain/(1+.015*(len(base.route)-1))
                bonus=.35*diff if c in refined and floor_ok else 0.
                selected=target==q.tolist()
                reason=('blocked_or_bounds' if not valid else 'near_placed' if not safe else
                        'selected' if selected else 'difficulty_floor' if c in refined and not floor_ok
                        else 'not_best')
                rows.append(dict(position=q.tolist(),parent=c,rank=rank,precise_score=value*(1+bonus),
                                 marginal_gain=v,difficulty_bonus=bonus,selected=selected,reject_reason=reason))
    return rows


def main(batch):
    for run in sorted((batch/'advanced6_l1/red').glob('Map*_r*')):
        log = events(run)
        decisions = [e for e in log if e['event']=='a6_selection']
        targets = {(e['step'], e['drone']):e['target'] for e in log if e['event']=='bomb_target'}
        keys = {(e['step'],e['drone']) for e in decisions}
        if not decisions or not targets.keys() <= keys or len(keys) != len(decisions):
            raise ValueError(f'Incomplete selection log {run}')
        n = 0
        with gzip.open(batch/(run.name+'_candidates.jsonl.gz'),'wt',encoding='utf-8') as out:
            for e in decisions:
                # The target may be cancelled later within the same navigation call.
                e = dict(e, selected=targets.get((e['step'],e['drone'])))
                rows = replay(e)
                n += len(rows)
                out.write(json.dumps(dict(run=run.name,step=e['step'],drone=e['drone'],
                          bomb_id=e['bomb_id'],selected=e['selected'],candidates=rows))+'\n')
        print(run.name, len(decisions), n, flush=True)


if __name__=='__main__':
    main(Path(sys.argv[1]))
