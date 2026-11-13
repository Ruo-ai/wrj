"""Variable partition trees, local obstacles, sampled starts and measured detours."""
import random
from collections import deque
import cv2
import numpy as np
from .structures import VERSION
from .validate import validate,wall_mask,digest
from .wall_spacing import check_spacing,surface_gap
from .errors import LayoutExhaustedError


def route_metrics(mask,spawn):
    # Conservative 8px cells: every pixel must clear the physical drone radius.
    clearance=cv2.distanceTransform(1-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    step=8;h,w=mask.shape;hh,ww=h//step,w//step
    free=(clearance[:hh*step,:ww*step].reshape(hh,step,ww,step).min(axis=(1,3))>18)
    sx,sy=spawn[0]//step,spawn[1]//step
    dist=np.full(free.shape,-1,np.int32)
    if not free[sy,sx]:raise ValueError('Start outside route grid')
    dist[sy,sx]=0;queue=deque([(sy,sx)])
    while queue:
        y,x=queue.popleft()
        for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0<=yy<hh and 0<=xx<ww and free[yy,xx] and dist[yy,xx]<0:
                dist[yy,xx]=dist[y,x]+1;queue.append((yy,xx))
    yy,xx=np.where(dist>=0);ds=dist[yy,xx]*step
    direct=np.hypot(xx-sx,yy-sy)*step;far=direct>200
    return dict(route_reachable_fraction=float((dist>=0).sum()/free.sum()),
        route_p90_px=float(np.percentile(ds,90)),
        detour_p90=float(np.percentile(ds[far]/direct[far],90)))


def build_randomized(w,h,seed,thickness=None,passage=None):
    from .semifinal import RULES,reference_manifest
    refs=reference_manifest();last='no candidate'
    for attempt in range(96):
        rng=random.Random(f'partition-mix-v5:{seed}:{attempt}')
        t=thickness or 6;gap=passage or rng.choice((56,64,72,80,88))
        leaves=[(6,6,w-6,h-6)];walls=[]
        target=rng.randint(19,30);omitted=0
        for _ in range(110):
            if len(leaves)>=target:break
            candidates=[a for a in leaves if max(a[2]-a[0],a[3]-a[1])>390]
            if not candidates:break
            a=rng.choices(candidates,weights=[((v[2]-v[0])*(v[3]-v[1]))**rng.uniform(.5,1.3) for v in candidates])[0]
            l,top,r,b=a;vertical=(r-l)/(b-top)>rng.uniform(.65,1.6)
            low,high,lo,hi=(l,r,top,b) if vertical else (top,b,l,r)
            if high-low<390 or hi-lo<gap+125:continue
            mid=rng.randint(low+180,high-180)
            door=rng.randint(lo+60,hi-gap-60)
            # A minority of removed separators form irregular joined halls.
            previous=len(walls)
            remove=rng.random()<.14
            if remove:
                omitted+=1
            elif vertical:
                walls.extend([['wall',mid,top,mid+t,door],['wall',mid,door+gap,mid+t,b]])
            else:
                walls.extend([['wall',l,mid,door,mid+t],['wall',door+gap,mid,r,mid+t]])
            try:check_spacing(dict(size=[w,h],walls=walls))
            except ValueError:
                del walls[previous:]
                continue
            leaves.remove(a)
            offset=0 if remove else t
            leaves.extend([(l,top,mid,b),(mid+offset,top,r,b)] if vertical else [(l,top,r,mid),(l,mid+offset,r,b)])
        # Each leaf independently chooses empty, pillar, box or local partition.
        modules={'box':0,'partition':0}
        for l,top,r,b in leaves:
            kind=rng.choices(('empty','box','partition'),(.3,.35,.35))[0]
            if kind=='empty' or min(r-l,b-top)<180:continue
            x=rng.randint(l+60,r-100);y=rng.randint(top+60,b-100)
            bw=rng.randint(28,max(28,min(160,r-x-55)))
            bh=rng.randint(28,max(28,min(190,b-y-55))) if kind=='box' else t
            candidate=[kind if kind=='box' else 'wall',x,y,x+bw,y+bh]
            if all(surface_gap(candidate[1:],v[1:])>=48 for v in walls):
                walls.append(candidate);modules[kind]+=1
        spec=dict(version=VERSION,size=[w,h],walls=walls,gates=[],passage_width=gap)
        try:
            mask=wall_mask(spec);check_spacing(spec)
            clearance=cv2.distanceTransform(1-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            # A 95px disk contains the full ten-drone formation and hulls.
            ys,xs=np.where(clearance[::12,::12]>95)
            points=[(int(x*12),int(y*12)) for x,y in zip(xs,ys)]
            if not points:continue
            spawn=rng.choice(points)
            nearby=[p for p in points if 200<np.hypot(p[0]-spawn[0],p[1]-spawn[1])<500]
            if not nearby:continue
            center=rng.choice(nearby)
            zones=[]
            for zone_type in ('NO_COM_ZONE','NO_GPS_ZONE'):
                for _ in range(40):
                    zw=rng.randint(w//3,int(w*.72));zh=rng.randint(h//3,int(h*.72))
                    l=rng.randint(15,w-zw-15);top=rng.randint(15,h-zh-15)
                    if not (l-85<=spawn[0]<=l+zw+85 and top-85<=spawn[1]<=top+zh+85):
                        zones.append(dict(type=zone_type,rect=[l,top,l+zw,top+zh]));break
            if len(zones)!=2:continue
            spec.update(spawn=list(spawn),center=list(center),center_size=[96,120],
                start_area=[spawn[0]-80,spawn[1]-80,spawn[0]+80,spawn[1]+80],rules=RULES.copy(),zones=zones)
            mask,stats=validate(spec)
            occupied=mask.copy();x,y=center;occupied[y-60:y+60,x-48:x+48]=1
            metrics=route_metrics(occupied,spawn)
            if metrics['route_reachable_fraction']<.98 or metrics['route_p90_px']<.9*np.hypot(w,h) or metrics['detour_p90']<1.7:
                last='insufficient route difficulty';continue
        except ValueError as error:last=str(error);continue
        stats.update(layout_algorithm='semifinal-partition-mix-v5',layout_family='random_partition_mix',
            room_count=None,partition_leaf_count=len(leaves),removed_separators=omitted,local_modules=modules,
            topology_dead_ends=None,attempt=attempt,wall_thickness=t,passage_width=gap,
            calibration_sources=refs,calibration_sha256=digest(refs),rules_profile=RULES.copy(),
            zones=zones,zone_mode='both',wall_fraction=float(mask.mean()),**metrics)
        return spec,stats,mask
    raise LayoutExhaustedError('Randomized arena exhausted: '+last)
