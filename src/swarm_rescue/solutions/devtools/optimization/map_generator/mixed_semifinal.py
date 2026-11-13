"""Random mixed arenas: unequal wings, bent corridors, rooms and obstacle halls."""
import random
from .structures import VERSION
from .validate import validate,digest
from .wall_spacing import surface_gap
from .errors import LayoutExhaustedError

FAMILIES=('room_wings','offset_corridors','obstacle_courtyard','mixed_partitions')

def build_mixed(width,height,seed,thickness=None,passage=None):
    from .semifinal import RULES,reference_manifest
    family=FAMILIES[(seed-1000)%5]
    refs=reference_manifest()
    for attempt in range(64):
        rng=random.Random(f'mixed-v4:{seed}:{attempt}')
        w,h=width,height;t=thickness or 6;gap=passage or rng.choice((64,76,88))
        walls=[];rooms=0
        spawn=[w-250,170];center=[w-80,170]
        reserved=[(w-350,65,w-145,270),(w-140,45,w-20,285)]
        def rect(l,y,r,b,kind='wall',clearance=40):
            a=tuple(map(int,(l,y,r,b)))
            if not (6<=a[0]<a[2]<=w-6 and 6<=a[1]<a[3]<=h-6): return False
            if any(surface_gap(a,z)<clearance for z in reserved): return False
            for row in walls:
                z=row[1:];d=surface_gap(a,z)
                if 0<d<clearance:return False
                if a[0]<z[2] and z[0]<a[2] and a[1]<z[3] and z[1]<a[3]:return False
            walls.append([kind,*a]);return True
        def room(x,y,rw,rh,side):
            nonlocal rooms
            before=len(walls)
            pieces=[(x,y,x+rw,y+t),(x,y+rh-t,x+rw,y+rh),
                    (x,y+t,x+t,y+rh-t),(x+rw-t,y+t,x+rw,y+rh-t)]
            l,top,r,b=pieces.pop(side)
            if side<2:
                mid=(l+r)//2;pieces += [(l,top,mid-gap//2,b),(mid+gap//2,top,r,b)]
            else:
                mid=(top+b)//2;pieces += [(l,top,r,mid-gap//2),(l,mid+gap//2,r,b)]
            if all(rect(*p) for p in pieces): rooms+=1
            else: del walls[before:]
        if family=='room_wings':
            # Unequal room clusters and an open center; room entrances vary.
            for xbase in (60,int(w*.51)):
                y=60 if xbase<100 else 330
                while y<h-180:
                    rh=rng.randint(150,210);rw=rng.randint(260,420)
                    room(xbase+rng.randint(0,35),y,rw,min(rh,h-55-y),rng.randrange(4))
                    y+=rh+rng.randint(64,105)
            for y in (int(h*.32),int(h*.65)):
                rect(int(w*.34),y,int(w*.48),y+t)
        elif family=='offset_corridors':
            # Alternating long partitions force detours; openings are staggered.
            for i,y in enumerate(range(320,h-110,rng.randint(170,210))):
                mid=rng.randint(250,w-300)
                rect(6,y,mid-gap//2,y+t);rect(mid+gap//2,y,w-6,y+t)
                for x in rng.sample(list(range(180,w-250,230)),3):
                    length=rng.randint(70,105)
                    rect(x,y+t,x+t,y+t+length)
            for x in (90,440,790):room(x,65,rng.randint(220,280),155,rng.randrange(4))
        elif family=='obstacle_courtyard':
            # Large boxes produce occlusions; local room wings preserve thin walls.
            for x in (65,int(w*.71)):
                for y in (350,650):room(x,y,rng.randint(250,330),rng.randint(180,240),rng.randrange(4))
            for _ in range(100):
                x=rng.randint(int(w*.28),int(w*.66));y=rng.randint(100,h-170)
                rw,rh=rng.randint(55,140),rng.randint(55,150)
                rect(x,y,x+rw,y+rh,'box',clearance=64)
        else:
            # Pack variable-sized local rooms, never a canvas-wide regular lattice.
            for _ in range(100):
                rw,rh=rng.randint(180,340),rng.randint(145,240)
                x=rng.randint(50,w-rw-50);y=rng.randint(50,h-rh-50)
                room(x,y,rw,rh,rng.randrange(4))
                if rooms>=rng.randint(9,12):break
        # Small pillars and separated L partitions vary open-space routes.
        for _ in range(30):
            x,y=rng.randint(65,w-200),rng.randint(65,h-160)
            if rng.random()<.4:rect(x,y,x+32,y+32,'box',clearance=76)
            else:rect(x,y,x+rng.randint(80,150),y+t,clearance=88)
        if len(walls)<24:continue
        zones=[dict(type='NO_COM_ZONE',rect=[35,rng.randint(300,430),int(w*rng.uniform(.48,.70)),h-35]),
               dict(type='NO_GPS_ZONE',rect=[int(w*rng.uniform(.25,.45)),rng.randint(310,470),w-35,h-35])]
        spec=dict(version=VERSION,size=[w,h],walls=walls,gates=[],passage_width=gap,spawn=spawn,center=center,
            center_size=[96,180],start_area=[w-330,90,w-170,250],rules=RULES.copy(),zones=zones)
        fx,fy=rng.choice((False,True)),rng.choice((False,True))
        def transform(a):
            l,top,r,b=a
            return [w-r if fx else l,h-b if fy else top,w-l if fx else r,h-top if fy else b]
        spec['walls']=[[k,*transform(a)] for k,*a in walls]
        spec['start_area']=transform(spec['start_area'])
        for z in zones:z['rect']=transform(z['rect'])
        for key in ('spawn','center'):
            x,y=spec[key];spec[key]=[w-x if fx else x,h-y if fy else y]
        try:mask,stats=validate(spec)
        except ValueError as error:last_error=str(error);continue
        stats.update(layout_algorithm='semifinal-mixed-v4',layout_family=family,room_count=rooms,
            topology_dead_ends=None,attempt=attempt,wall_thickness=t,passage_width=gap,
            calibration_sources=refs,calibration_sha256=digest(refs),rules_profile=RULES.copy(),
            zones=zones,zone_mode='both',wall_fraction=float(mask.mean()))
        return spec,stats,mask
    raise LayoutExhaustedError('Mixed layout exhausted: '+locals().get('last_error','density'))
