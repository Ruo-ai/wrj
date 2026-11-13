"""New BSP geometry calibrated with aggregate official map features."""
import random

from .official_profile import read_profiles
from .structures import BORDER, VERSION
from .validate import validate
from .errors import LayoutExhaustedError


def intersects(a,b):
    return a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3]


def build_official(width,height,difficulty,seed,thickness=None,passage=None):
    profiles = sorted(read_profiles(),key=lambda p:p['internal_line_count'])
    choices = {'easy':profiles[:3],'medium':profiles[1:5],'hard':profiles[3:]}[difficulty]
    reason = ''
    for attempt in range(64):
        rng = random.Random(f'official-bsp:3:{seed}:{attempt}')
        reference = rng.choice(choices)
        t = thickness if thickness is not None else rng.choice(reference['line_thicknesses'])
        gap = passage if passage is not None else {'easy':120,'medium':100,'hard':76}[difficulty]
        # Marker pairs remain paired; dimensions and location are jittered, not copied.
        shift = [rng.uniform(-.025,.025),rng.uniform(-.025,.025)]
        markers = {}
        for name in ('start_area','disposal_center'):
            model = reference['markers'][name]
            minimum = 128 if name=='start_area' else 48
            size = [max(minimum,2*round(s*v*rng.uniform(.9,1.1)/2))
                    for s,v in zip(model['size'],(width,height))]
            xy = [max(d//2+8,min(v-d//2-8,round((q+delta)*v)))
                  for d,v,q,delta in zip(size,(width,height),model['center'],shift)]
            markers[name] = [xy[0]-size[0]//2,xy[1]-size[1]//2,
                             xy[0]+size[0]//2,xy[1]+size[1]//2]
        start,center = markers['start_area'],markers['disposal_center']
        if intersects([start[0]-20,start[1]-20,start[2]+20,start[3]+20],center):
            continue
        reserved = [[max(BORDER,r[0]-20),max(BORDER,r[1]-20),
                     min(width-BORDER,r[2]+20),min(height-BORDER,r[3]+20)] for r in (start,center)]
        scale_area = width*height/(reference['size'][0]*reference['size'][1])
        target_lines = max(2,round(reference['internal_line_count']*scale_area*rng.uniform(.85,1.15)))
        leaves = [[BORDER,BORDER,width-BORDER,height-BORDER]]
        walls,gates = [],[]
        # Each split produces two thin wall segments and one doorway. New split
        # coordinates and topology are independent of the reference map geometry.
        for _ in range(160):
            if len(walls)>=target_lines:
                break
            possible = [r for r in leaves if max(r[2]-r[0],r[3]-r[1])>=350]
            if not possible:
                break
            room = rng.choices(possible,weights=[(r[2]-r[0])*(r[3]-r[1]) for r in possible])[0]
            l,top,r,b = room
            axis = 'v' if (r-l)/(b-top)>rng.uniform(.8,1.2) else 'h'
            low,high,lo,hi = (l,r,top,b) if axis=='v' else (top,b,l,r)
            if high-low<350 or hi-lo<gap+70:
                continue
            mid = rng.randint(low+150,high-150)
            door = rng.randint(lo+30,hi-gap-30)
            a = mid-t//2
            segments = ([['wall',a,lo,a+t,door],['wall',a,door+gap,a+t,hi]] if axis=='v'
                        else [['wall',lo,a,door,a+t],['wall',door+gap,a,hi,a+t]])
            if any(intersects(s[1:],zone) for s in segments for zone in reserved):
                continue
            walls.extend(segments)
            gates.append([axis,mid,door,door+gap])
            leaves.remove(room)
            leaves.extend([[l,top,a,b],[a+t,top,r,b]] if axis=='v'
                          else [[l,top,r,a],[l,a+t,r,b]])
        # Extra openings admit alternate routes; a pure BSP doorway tree would
        # overrepresent forced backtracking compared with the open official maps.
        additional_openings = 0
        eligible = [s for s in walls if max(s[3]-s[1],s[4]-s[2])>gap+120]
        rng.shuffle(eligible)
        for segment in eligible[:len(walls)//6]:
            _,l,top,r,b = segment
            axis = 'v' if b-top>r-l else 'h'
            lo,hi,mid = (top,b,l+t//2) if axis=='v' else (l,r,top+t//2)
            door = rng.randint(lo+50,hi-gap-50)
            walls.remove(segment)
            walls.extend([['wall',l,top,r,door],['wall',l,door+gap,r,b]] if axis=='v'
                         else [['wall',l,top,door,b],['wall',door+gap,top,r,b]])
            gates.append([axis,mid,door,door+gap])
            additional_openings += 1
        # Entity sizes are sampled from observed size statistics with jitter.
        for bw,bh in reference['box_sizes']:
            bw = max(24,round(bw*width/reference['size'][0]*rng.uniform(.9,1.1)))
            bh = max(24,round(bh*height/reference['size'][1]*rng.uniform(.9,1.1)))
            for _ in range(60):
                l,top,r,b = rng.choice(leaves)
                if r-l<bw+80 or b-top<bh+80:
                    continue
                x,y = rng.randint(l+40,r-bw-40),rng.randint(top+40,b-bh-40)
                box = [x,y,x+bw,y+bh]
                expanded = [x-38,y-38,x+bw+38,y+bh+38]
                if any(intersects(expanded,s[1:]) for s in walls) or any(intersects(box,z) for z in reserved):
                    continue
                walls.append(['box',*box])
                break
        spec = dict(version=VERSION,size=[width,height],walls=sorted(walls),gates=gates,
                    passage_width=gap,spawn=[(start[0]+start[2])//2,(start[1]+start[3])//2],
                    center=[(center[0]+center[2])//2,(center[1]+center[3])//2],
                    center_size=[center[2]-center[0],center[3]-center[1]],start_area=start)
        try:
            mask,stats = validate(spec)
            count = sum(s[0]=='wall' for s in walls)
            if count<max(2,.8*target_lines):
                raise ValueError('Insufficient partition density after reserving base areas')
            if sum(s[0]=='box' for s in walls)!=reference['box_count']:
                raise ValueError('Entity count could not match sampled profile')
            fraction = mask.mean()
            # Reference uses editor raster; a generous tolerance accounts for
            # edge rasterization and newly sampled geometry, not a scoring claim.
            if not .6*reference['wall_fraction_proxy']<=fraction<=1.6*reference['wall_fraction_proxy']:
                raise ValueError('Wall area fraction outside sampled profile tolerance')
        except ValueError as error:
            reason = str(error)
            continue
        stats.update(attempt=attempt,room_count=len(leaves),topology_dead_ends=None,
                     wall_thickness=t,passage_width=gap,layout_algorithm='official-bsp-v3',
                     additional_openings=additional_openings,
                     calibration_source=reference['source'],calibration_sha256=reference['source_sha256'],
                     target_internal_lines=target_lines,actual_internal_lines=count,
                     reference_lines_per_mpx=round(reference['internal_line_count']*1e6/(reference['size'][0]*reference['size'][1]),4),
                     actual_lines_per_mpx=round(count*1e6/(width*height),4),
                     wall_fraction=round(float(fraction),6),target_wall_fraction_proxy=reference['wall_fraction_proxy'],
                     box_count=reference['box_count'])
        return spec,stats,mask
    raise LayoutExhaustedError(f'Official-style generation exhausted 64 attempts: {reason}')
