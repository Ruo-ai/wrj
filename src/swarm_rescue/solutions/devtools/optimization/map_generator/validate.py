"""Static geometry checks shared by export and native loading."""
import hashlib
import json

import cv2
import numpy as np

from .structures import BORDER, VERSION, spawn_points


CLEARANCE = 18  # Native hull radius 16.401 plus sqrt(2) raster margin, rounded up.


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('ascii')).hexdigest()


def geometry_hash(spec):
    data = [spec['size'],spec['spawn'],spec['center'],sorted(spec['walls'])]
    if 'center_size' in spec:
        data.append(spec['center_size'])
    if 'rules' in spec:
        data.extend([spec['rules'],spec.get('zones',[])])
    return digest(data)


def structure_hash(spec):
    """Do not count changes of spawn, wall order, or axis reflections as new structures."""
    w,h = spec['size']
    variants = []
    for flip_x,flip_y in ((False,False),(True,False),(False,True),(True,True)):
        rects = []
        for _,l,t,r,b in spec['walls']:
            if flip_x:
                l,r = w-r,w-l
            if flip_y:
                t,b = h-b,h-t
            rects.append((l,t,r,b))
        variants.append(sorted(rects))
    return digest([w,h,min(variants)])


def point(value):
    return isinstance(value, list) and len(value)==2 and all(type(v) is int for v in value)


def wall_mask(spec):
    if spec.get('version') != VERSION or not point(spec.get('size')):
        raise ValueError('Unsupported map version or size')
    w,h = spec['size']
    if not all(600 <= v <= 1800 and v%2==0 for v in (w,h)):
        raise ValueError('Canvas must have even sides of 600..1800 px')
    mask = np.zeros((h,w), np.uint8)
    mask[:BORDER] = mask[-BORDER:] = 1
    mask[:,:BORDER] = mask[:,-BORDER:] = 1
    for kind,l,t,r,b in spec['walls']:
        if kind not in ('box','wall') or any(type(k) is not int for k in (l,t,r,b)):
            raise ValueError('Unsupported wall record')
        if not (BORDER <= l < r <= w-BORDER and BORDER <= t < b <= h-BORDER):
            raise ValueError('Wall is degenerate or outside inner border')
        if kind=='wall' and r-l==b-t:
            raise ValueError('A square obstacle must use box geometry')
        region = mask[t:b,l:r]
        if region.any():
            raise ValueError('Overlapping interiors of walls')
        region[:] = 1
    return mask


def validate(spec):
    walls = wall_mask(spec)
    h,w = walls.shape
    if cv2.connectedComponents(1-walls, connectivity=4)[0]!=2:
        raise ValueError('Unexpected enclosed region in raw geometry')
    if not point(spec.get('spawn')) or not point(spec.get('center')):
        raise ValueError('Expected integer spawn and center points')
    gap = spec['passage_width']
    if type(gap) is not int or not 38<=gap<=140:
        raise ValueError('Invalid design passage width')
    for axis,mid,lo,hi in spec['gates']:
        if axis not in ('h','v') or not all(type(v) is int for v in (mid,lo,hi)):
            raise ValueError('Invalid passage')
        view = walls if axis=='h' else walls.T
        if not (0<=mid<view.shape[0] and 0<lo<hi<view.shape[1] and hi-lo==gap
                and hi-lo>2*CLEARANCE and view[mid,lo-1] and view[mid,hi]
                and not view[mid,lo:hi].any()):
            raise ValueError('Passage does not match its declared clear width')
    x,y = spec['center']
    size = spec.get('center_size',[48,48])
    if not point(size) or any(v<24 or v%2 for v in size):
        raise ValueError('Expected even disposal center dimensions of at least 24')
    hx,hy = [v//2 for v in size]
    if not (hx<x<w-hx and hy<y<h-hy) or walls[y-hy:y+hy,x-hx:x+hx].any():
        raise ValueError('Disposal center intersects wall or border')
    occupied = walls.copy()
    occupied[y-hy:y+hy,x-hx:x+hx] = 1
    clearance = cv2.distanceTransform(1-occupied, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    free = (clearance>CLEARANCE).astype(np.uint8)
    components, _ = cv2.connectedComponents(free, connectivity=4)
    if components != 2:
        raise ValueError('Unexpected disconnected navigable regions')
    for count in ((5,10) if 'rules' in spec else (5,)):
        for sx,sy in spawn_points(spec['spawn'],count):
            if not (0<=sx<w and 0<=sy<h and free[sy,sx]):
                raise ValueError(f'{count}-drone formation lacks clearance')
    if 'rules' in spec:
        from .semifinal import RULES
        from .wall_spacing import check_spacing
        check_spacing(spec)
        if spec['rules']!=RULES:
            raise ValueError('Unrecognized semifinal rule profile')
        for zone in spec.get('zones',[]):
            if zone.get('type') not in ('NO_COM_ZONE','NO_GPS_ZONE'):
                raise ValueError('Unsupported zone')
            l,t,r,b = zone['rect']
            if not all(type(v) is int for v in (l,t,r,b)) or not (0<=l<r<=w and 0<=t<b<=h):
                raise ValueError('Invalid zone bounds')
            sx,sy = spec['spawn']
            if l-80<=sx<=r+80 and t-80<=sy<=b+80:
                raise ValueError('Initial formation must be outside disabling zones')
    # Require an accessible approach to a face of the disposal center.
    ox,oy = hx+CLEARANCE+2,hy+CLEARANCE+2
    if not any(0<=px<w and 0<=py<h and free[py,px]
               for px,py in ((x-ox,y),(x+ox,y),(x,y-oy),(x,y+oy))):
        raise ValueError('No approach to disposal center')
    for axis,mid,lo,hi in spec['gates']:
        view = free if axis=='h' else free.T
        if not view[mid,lo:hi].any():
            raise ValueError('Center obstructs required passage')
    return walls, dict(wall_count=len(spec['walls']), wall_area_px=int(walls.sum()),
                       connected_components=components-1, clearance_px=CLEARANCE,
                       door_count=len(spec['gates']),
                       narrow_door_count=sum(hi-lo<=60 for _,_,lo,hi in spec['gates']))
