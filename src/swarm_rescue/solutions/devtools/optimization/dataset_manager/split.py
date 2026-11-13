"""Deterministic grouping and whole-group assignment; never split similar siblings."""
from collections import defaultdict
import math

from ..map_generator.validate import digest


class UnionFind:
    def __init__(self,ids):
        self.parent = {i:i for i in ids}

    def find(self,i):
        while self.parent[i]!=i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self,a,b):
        a,b = self.find(a),self.find(b)
        if a==b:
            return False
        self.parent[max(a,b)] = min(a,b)
        return True


def similarity(a,b):
    base = int(a['signatures'][0],16)
    best = 0.
    for variant in b['signatures']:
        other = int(variant,16)
        union = (base|other).bit_count()
        best = max(best,(base&other).bit_count()/union if union else 1.)
    return best


def group_maps(rows,seed_block=10,near_threshold=.80):
    if type(seed_block) is not int or seed_block<1:
        raise ValueError('seed-block must be a positive integer')
    if not .5<=near_threshold<=1:
        raise ValueError('near-threshold must be in 0.5..1')
    rows = sorted(rows,key=lambda r:r['map_path'])
    ids = [r['id'] for r in rows]
    if len(set(ids))!=len(ids):
        raise ValueError('Duplicate index IDs')
    uf,edges = UnionFind(ids),[]
    anchors,exact = {},defaultdict(list)
    for row in rows:
        exact[row['structure_sha256']].append(row['id'])
        config = [row['family'],row['difficulty'],row['size'],row['layout_algorithm'],
                  row['generator_version'],row['wall_thickness'],row['passage_width'],
                  row['calibration_sha256'],row['seed']//seed_block]
        keys = [('structure',row['structure_sha256']),('same_seed',str(row['seed'])),
                ('parameter_seed_block',digest(config))]
        for reason,key in keys:
            token = (reason,key)
            if token in anchors:
                other = anchors[token]
                if uf.union(other,row['id']):
                    edges.append(dict(a=other,b=row['id'],reason=reason))
            else:
                anchors[token] = row['id']
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if uf.find(a['id'])==uf.find(b['id']):
                continue
            score = similarity(a,b)
            if score>=near_threshold:
                uf.union(a['id'],b['id'])
                edges.append(dict(a=a['id'],b=b['id'],reason='near_geometry',iou=round(score,6)))
    duplicates = [dict(kept=members[0],excluded=members[1:],structure_sha256=key)
                  for key,members in sorted(exact.items()) if len(members)>1]
    excluded = {i for d in duplicates for i in d['excluded']}
    members = defaultdict(list)
    for row in rows:
        members[uf.find(row['id'])].append(row['id'])
    groups = [dict(id=digest(sorted(group)),members=sorted(group),
                   active=sorted(i for i in group if i not in excluded)) for group in members.values()]
    return sorted(groups,key=lambda g:g['id']),duplicates,edges


def assign(rows,groups,ratios=(70,15,15),seed=2026,holdout_families=()):
    if len(ratios)!=3 or any(not math.isfinite(r) or r<0 for r in ratios) or sum(ratios)<=0:
        raise ValueError('Ratios must be three finite nonnegative values with positive sum')
    names = ('train','validation','test')
    proportions = {s:r/sum(ratios) for s,r in zip(names,ratios)}
    known = {r['family'] for r in rows}
    holdout = set(holdout_families)
    if holdout-known:
        raise ValueError('Holdout families absent from maps: '+', '.join(sorted(holdout-known)))
    by_id = {r['id']:r for r in rows}
    bins = {s:[] for s in (*names,'extrapolation')}
    assignment = {}
    ordinary = []
    for group in groups:
        if any(by_id[i]['family'] in holdout for i in group['members']):
            bins['extrapolation'].extend(group['active'])
            assignment[group['id']] = 'extrapolation'
        else:
            ordinary.append(group)
    available = [s for s in names if proportions[s]>0]
    if len(ordinary)<len(available):
        raise ValueError(f'Only {len(ordinary)} independent groups for {len(available)} nonempty splits; '
                         'add diverse maps, do not split a protected group to force ratios')
    total = sum(len(g['active']) for g in ordinary)
    target = {s:total*proportions[s] for s in names}
    ordered = sorted(ordinary,key=lambda g:(-len(g['active']),digest([seed,g['id']])))
    for n,group in enumerate(ordered):
        empty = [s for s in available if not bins[s]]
        candidates = empty if len(ordered)-n==len(empty) else available
        size = len(group['active'])
        def cost(split):
            current = len(bins[split])
            delta = ((current+size-target[split])**2-(current-target[split])**2)/max(1,target[split])
            return delta,digest([seed,group['id'],split])
        chosen = min(candidates,key=cost)
        bins[chosen].extend(group['active'])
        assignment[group['id']] = chosen
    for values in bins.values():
        values.sort()
    # Hard is an evaluation view of held-out maps, never an extra training pool.
    bins['hard'] = sorted(i for i in bins['test']+bins['extrapolation']
                          if by_id[i]['difficulty']=='hard')
    return bins,assignment,target


def make_split(rows,ratios=(70,15,15),seed=2026,seed_block=10,near_threshold=.80,holdout_families=()):
    groups,duplicates,edges = group_maps(rows,seed_block,near_threshold)
    bins,assignment,target = assign(rows,groups,ratios,seed,holdout_families)
    warnings = []
    if not bins['extrapolation']:
        warnings.append('No structure-extrapolation holdout: select unseen families after generating diverse structures.')
    if len({r['family'] for r in rows})==1:
        warnings.append('Only one structure family is present; ordinary test measures within-family generalization.')
    warnings.append('Hard labels are generator presets, not measured adversarial difficulty.')
    if duplicates:
        warnings.append('Repeated/reflected wall structures are represented once; aliases are recorded in duplicates.json.')
    return dict(bins=bins,groups=groups,duplicates=duplicates,edges=edges,assignment=assignment,
                target=target,warnings=warnings)
