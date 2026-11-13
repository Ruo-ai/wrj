"""Connected room graphs with controlled detours for semifinal training."""
import random
from collections import deque
from .structures import VERSION
from .validate import validate,digest
from .errors import LayoutExhaustedError


def build_challenge(width,height,seed,thickness=None,passage=None):
    from .semifinal import RULES,reference_manifest
    refs=reference_manifest()
    tier=(seed-1000)%3
    for attempt in range(48):
        rng=random.Random(f'semifinal-challenge-v3:{seed}:{attempt}')
        cols,rows=((5,3),(6,4),(7,4))[tier]
        t=thickness or 6
        gap=passage or rng.choice((64,72,80) if tier<2 else (56,64,72))
        xs=[6]+[round(6+(width-12)*i/cols)+rng.randint(-18,18) for i in range(1,cols)]+[width-6]
        ys=[300]+[round(300+(height-306)*i/rows)+rng.randint(-12,12) for i in range(1,rows)]+[height-6]
        def neighbors(n):
            x,y=n%cols,n//cols
            return [yy*cols+xx for xx,yy in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)) if 0<=xx<cols and 0<=yy<rows]
        root=cols-1
        visited={root}; stack=[root]; doors=set()
        while stack:
            n=stack[-1]; candidates=[v for v in neighbors(n) if v not in visited]
            if not candidates: stack.pop(); continue
            nxt=rng.choice(candidates); doors.add(tuple(sorted((n,nxt))))
            visited.add(nxt); stack.append(nxt)
        extras=[(n,v) for n in range(cols*rows) for v in neighbors(n) if n<v and (n,v) not in doors]
        rng.shuffle(extras)
        doors.update(extras[:(4,2,1)[tier]])
        walls=[]
        def wall(l,y,r,b):
            if l<r and y<b: walls.append(['wall',l,y,r,b])
        def horizontal(y,l,r,opened):
            if opened:
                mid=(l+r)//2+rng.randint(-12,12)
                wall(l,y,mid-gap//2,y+t);wall(mid+gap//2,y,r,y+t)
            else: wall(l,y,r,y+t)
        def vertical(x,top,bottom,opened):
            if opened:
                mid=(top+bottom)//2+rng.randint(-12,12)
                wall(x,top,x+t,mid-gap//2);wall(x,mid+gap//2,x+t,bottom)
            else: wall(x,top,x+t,bottom)
        for row in range(rows):
            for col in range(cols):
                n=row*cols+col
                horizontal(ys[row],xs[col],xs[col+1],col==root if row==0 else (n-cols,n) in doors)
                if col>0: vertical(xs[col],ys[row]+t,ys[row+1],(n-1,n) in doors)
        # Rooms are connected by a spanning tree plus a few genuine loops.
        adjacency={n:[] for n in range(cols*rows)}
        for a,b in doors: adjacency[a].append(b);adjacency[b].append(a)
        distances={root:0}; queue=deque([root])
        while queue:
            n=queue.popleft()
            for nxt in adjacency[n]:
                if nxt not in distances: distances[nxt]=distances[n]+1;queue.append(nxt)
        if max(distances.values())<(5,9,12)[tier]: continue
        zones=[dict(type='NO_COM_ZONE',rect=[30,320,int(width*.65),height-30]),
               dict(type='NO_GPS_ZONE',rect=[int(width*.30),350,width-30,height-30])]
        spec=dict(version=VERSION,size=[width,height],walls=sorted(walls),gates=[],passage_width=gap,
            spawn=[width-250,170],center=[width-80,170],center_size=[96,180],
            start_area=[width-330,90,width-170,250],rules=RULES.copy(),zones=zones)
        # Whole-map reflection varies start orientation without breaking graph edges.
        if rng.choice((True,False)):
            for wallrow in spec['walls']:
                l,r=wallrow[1],wallrow[3];wallrow[1],wallrow[3]=width-r,width-l
            for z in zones:
                l,top,r,b=z['rect'];z['rect']=[width-r,top,width-l,b]
            spec['spawn'][0]=width-spec['spawn'][0];spec['center'][0]=width-spec['center'][0]
            l,top,r,b=spec['start_area'];spec['start_area']=[width-r,top,width-l,b]
        try: mask,stats=validate(spec)
        except ValueError as error:
            last_error=str(error)
            continue
        stats.update(layout_algorithm='semifinal-challenge-v3',layout_family=('connected_rooms','deep_rooms','stress_rooms')[tier],
            challenge_tier=tier+1,room_count=cols*rows,topology_dead_ends=sum(len(v)==1 for v in adjacency.values()),
            graph_max_distance=max(distances.values()),graph_extra_edges=len(doors)-(cols*rows-1),
            attempt=attempt,wall_thickness=t,passage_width=gap,calibration_sources=refs,
            calibration_sha256=digest(refs),rules_profile=RULES.copy(),zones=zones,zone_mode='both',wall_fraction=float(mask.mean()))
        return spec,stats,mask
    raise LayoutExhaustedError('Challenge layout failed: '+locals().get('last_error','topology'))
