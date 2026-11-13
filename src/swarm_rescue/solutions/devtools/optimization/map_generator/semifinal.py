"""Offline mixed-layout generation under explicit semifinal scenario conditions."""
import hashlib
import json
from pathlib import Path
import random

from .errors import LayoutExhaustedError
from .structures import VERSION
from .validate import validate, digest
from .wall_spacing import surface_gap

REPO = Path(__file__).resolve().parents[6]
REFERENCES = tuple(f'map_{n:02d}.json' for n in range(1,10))+('map_07t4.json',)
RULES = dict(stage='semifinal',version=1,lidar_range=230,semantic_range=130,
    communication_range=180,initial_health=25,time_best_ratio=.3,
    red_drones=5,blue_drones=10,bombs_per_drone=2,timestep_limit=5400,walltime_limit=180)


def reference_manifest():
    # Explicit inputs: adding a random file cannot silently change the population.
    return {name:hashlib.sha256((REPO/'map_data'/name).read_bytes()).hexdigest()
            for name in REFERENCES}


def build_semifinal(width,height,difficulty,seed,thickness=None,passage=None):
    if min(width,height)<900:
        raise ValueError('Semifinal maps require both sides >= 900 px')
    if difficulty=='hard':
        from .randomized_semifinal import build_randomized
        return build_randomized(width,height,seed,thickness,passage)
    refs = reference_manifest()
    family = ('hall_rooms','corridor_wings','mixed_blocks')[(seed-1000)%3]
    for attempt in range(48):
        rng = random.Random(f'semifinal-layout-v2:{seed}:{attempt}')
        w,h = width,height
        t = thickness or 6
        gap = passage or rng.choice([76,96,116])
        walls = []
        spawn = [w-250,170]
        center = [w-80,170]
        reserved = [(spawn[0]-90,spawn[1]-90,spawn[0]+90,spawn[1]+90),
                    (center[0]-65,center[1]-120,center[0]+65,center[1]+120)]

        def overlap(a,b):
            return a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3]

        def rect(l,y,r,b,kind='wall'):
            box = tuple(map(int,(l,y,r,b)))
            if not (6<=box[0]<box[2]<=w-6 and 6<=box[1]<box[3]<=h-6):
                return False
            if any(overlap(box,z) for z in reserved) or any(overlap(box,s[1:]) for s in walls):
                return False
            if any(0<surface_gap(box,s[1:])<40 for s in walls):
                return False
            walls.append([kind,*box]);return True

        def horizontal(y,l,r):
            return rect(l,y,r,y+t)

        def vertical(x,top,bottom):
            return rect(x,top,x+t,bottom)

        def room(x,y,rw,rh):
            # A three-sided room with a real-width opening on the fourth side.
            horizontal(y,x,x+rw)
            vertical(x,y+t,y+rh)
            vertical(x+rw-t,y+t,y+rh)
            horizontal(y+rh,x,x+(rw-gap)//2)
            horizontal(y+rh,x+(rw+gap)//2,x+rw)

        if family=='hall_rooms':
            # Large open area separated from a locally dense room wing.
            split = int(w*rng.uniform(.53,.61))
            vertical(split,310,h//2-gap//2)
            vertical(split,h//2+gap//2,h-6)
            horizontal(285,160,w-6)
            horizontal(int(h*.62),80,split-90)
            for y in (int(h*.38),int(h*.70)):
                room(split+95,y,w-split-180,int(h*.21))
            blocks = [(rng.randint(160,260),h//3,150,210),
                      (180,h-280,155,200),(410,h-280,155,200)]
        elif family=='corridor_wings':
            # A long, open-ended corridor; rooms do not fill the whole canvas.
            top,bottom = int(h*.36),int(h*.55)
            for y in (top,bottom):
                for l,r in ((6,w//3-gap//2),(w//3+gap//2,2*w//3-gap//2),(2*w//3+gap//2,w-6)):
                    horizontal(y,l,r)
            for x in (100,430,760):
                room(x,55,240,top-150)
            for x in (90,470,850):
                room(x,bottom+90,270,h-bottom-180)
            blocks = [(w-290,h-250,45,45),(w-440,h-170,40,40)]
        else:
            # Column hall with local U-shaped partitions, not a BSP room tree.
            for y in (300,int(h*.72)):
                horizontal(y,100,int(w*.47))
            room(120,350,int(w*.32),int(h*.27))
            vertical(int(w*.49),310,h-100)
            blocks = [(int(w*.60)+c*110,370+r*160,rng.choice([28,36,44]),36)
                      for r in range(3) for c in range(4)]
            blocks += [(190,h-230,180,110)]
        for x,y,bw,bh in blocks:
            expanded = (x-42,y-42,x+bw+42,y+bh+42)
            if not any(overlap(expanded,s[1:]) for s in walls):
                rect(x,y,x+bw,y+bh,'box')
        # Jittered free wall stubs add alternate corridors without cloning references.
        for _ in range(8):
            x,y = rng.randrange(80,w-400),rng.randrange(80,h-100)
            if rng.random()<.5:
                box=(x,y,min(w-350,x+rng.randrange(70,180)),y+t)
            else:
                box=(x,y,x+t,min(h-46,y+rng.randrange(70,180)))
            # Keep random details away from door jambs and structural walls.
            if all(surface_gap(box,s[1:])>=max(76,gap) for s in walls):
                rect(*box)
        mode = ('both','communication','localization','none')[((seed-1000)//3)%4]
        zones = []
        if mode in ('both','communication'):
            zones.append(dict(type='NO_COM_ZONE',rect=[50,h//3,int(w*.49),h-30]))
        if mode in ('both','localization'):
            zones.append(dict(type='NO_GPS_ZONE',rect=[int(w*.36),h//2,w-50,h-30]))
        # Reflect complete geometry/markers/zones together; preserve wall joints.
        flip_x,flip_y = rng.choice([False,True]),rng.choice([False,True])
        def transform(box):
            l,top,r,b = box
            return [w-r if flip_x else l,h-b if flip_y else top,
                    w-l if flip_x else r,h-top if flip_y else b]
        walls = [[kind,*transform([l,y,r,b])] for kind,l,y,r,b in walls]
        for zone in zones:
            zone['rect'] = transform(zone['rect'])
        spawn = [w-spawn[0] if flip_x else spawn[0],h-spawn[1] if flip_y else spawn[1]]
        center = [w-center[0] if flip_x else center[0],h-center[1] if flip_y else center[1]]
        spec = dict(version=VERSION,size=[w,h],walls=sorted(walls),gates=[],passage_width=gap,
            spawn=spawn,center=center,center_size=[96,180],
            start_area=[spawn[0]-80,spawn[1]-80,spawn[0]+80,spawn[1]+80],
            rules=RULES.copy(),zones=zones)
        try:
            mask,stats = validate(spec)
        except ValueError:
            continue
        stats.update(layout_algorithm='semifinal-modules-v2',layout_family=family,
            room_count={'hall_rooms':2,'corridor_wings':6,'mixed_blocks':1}[family],topology_dead_ends=None,
            attempt=attempt,wall_thickness=t,passage_width=gap,
            calibration_sources=refs,calibration_sha256=digest(refs),
            rules_profile=RULES.copy(),zones=zones,zone_mode=mode,
            wall_fraction=float(mask.mean()))
        return spec,stats,mask
    raise LayoutExhaustedError(f'Semifinal {family}: no valid layout in 48 attempts')
