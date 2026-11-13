"""Generate axis-aligned geometry in canvas pixels; rectangles are half-open."""
import random


FAMILIES = ('semifinal', 'official', 'open', 'rooms', 'branches', 'maze', 'bottleneck', 'narrow', 'u_wall',
            'l_wall', 'ring', 'long_wall', 'thick_wall', 'blocks', 'junction', 'occluded', 'mixed')
VERSION = 2
DIFFICULTY = {'easy': (12, 110), 'medium': (16, 76), 'hard': (20, 46)}
BORDER = 6


def spawn_points(center, count=5):
    """Same 40px square-grid convention as the official five-drone maps, y-down."""
    x, y = center
    import math
    side = math.ceil(math.sqrt(count))
    return [[x + 40*(i%side) - (side-1)*20,
             y - (i//side*40 - (side-1)*20)] for i in range(count)]


def geometry(family, w, h, difficulty, seed, thickness=None, passage=None):
    rng = random.Random(seed)
    t, gap = DIFFICULTY[difficulty]
    t = t if thickness is None else thickness
    gap = gap if passage is None else passage
    walls, gates = [], []
    rooms = dead_ends = 0

    def rect(a,b,c,d,kind='wall'):
        walls.append([kind,a,b,c,d])

    def divide(axis, mid, lo, hi, opening=None, depth=t):
        left = mid-depth//2
        spans = [(lo,hi)]
        if opening is not None:
            start = opening-gap//2
            if not lo < start < start+gap < hi:
                raise ValueError('door does not fit divider')
            spans = [(lo,start),(start+gap,hi)]
            gates.append([axis,mid,start,start+gap])
        for a,b in spans:
            if axis=='v':
                rect(left,a,left+depth,b)
            else:
                rect(a,left,b,left+depth)

    cx, cy = w//2+rng.randint(-w//12,w//12), h//2+rng.randint(-h//12,h//12)
    if family in ('rooms','branches','maze','mixed'):
        cols = 3 if family!='maze' else 4
        rows = 2 if family in ('rooms','mixed') else 3
        xs = [BORDER]+[i*w//cols+rng.randint(-w//(14*cols),w//(14*cols)) for i in range(1,cols)]+[w-BORDER]
        ys = [BORDER]+[i*h//rows+rng.randint(-h//(14*rows),h//(14*rows)) for i in range(1,rows)]+[h-BORDER]
        cells = [(r,c) for r in range(rows) for c in range(cols)]
        edges = [(u,v) for u in cells for v in ((u[0]+1,u[1]),(u[0],u[1]+1)) if v in cells]
        doors = set()
        if family=='branches':
            trunk = rng.randrange(rows)
            doors = {(a,b) for a,b in edges if a[1]==b[1] or a[0]==trunk}
        else:
            # Randomized DFS spanning tree. Rooms/mixed admit extra cycles.
            stack = [rng.choice(cells)]
            visited = set(stack)
            while stack:
                u = stack[-1]
                choices = [v for v in cells if v not in visited and abs(v[0]-u[0])+abs(v[1]-u[1])==1]
                if choices:
                    v = rng.choice(choices)
                    doors.add(tuple(sorted((u,v))))
                    visited.add(v)
                    stack.append(v)
                else:
                    stack.pop()
            if family!='maze':
                doors.update(edge for edge in edges if edge not in doors and rng.random()<.45)
        rooms = len(cells)
        dead_ends = sum(sum(u in edge for edge in doors)==1 for u in cells)
        for a,b in edges:
            r,c = a
            if a[0]==b[0]:
                axis,mid,lo,hi = 'v',xs[c+1],ys[r],ys[r+1]
            else:
                axis,mid = 'h',ys[r+1]
                lo = xs[c]+(t-t//2 if c else 0)
                hi = xs[c+1]-(t//2 if c+1<cols else 0)
            center = (lo+hi)//2+rng.randint(-(hi-lo)//12,(hi-lo)//12)
            divide(axis,mid,lo,hi,center if (a,b) in doors else None)
        if family=='mixed':
            r,c = rng.choice(cells)
            x,y = (xs[c]+xs[c+1])//2,(ys[r]+ys[r+1])//2
            rect(x-30,y-22,x+30,y+22,'box')
    elif family in ('bottleneck','narrow'):
        rooms = 2
        divide('v',cx,BORDER,h-BORDER,cy,depth=t if family=='bottleneck' else w//6)
    elif family in ('u_wall','l_wall'):
        dx,dy = w//5,h//5
        rect(cx-dx,cy-dy,cx-dx+t,cy+dy)
        rect(cx-dx+t,cy+dy-t,cx+dx,cy+dy)
        if family=='u_wall':
            rect(cx+dx-t,cy-dy,cx+dx,cy+dy-t)
            dead_ends = 1
    elif family=='ring':
        rect(cx-w//4-t,cy-h//4-t,cx+w//4+t,cy+h//4+t,'box')
    elif family=='long_wall':
        divide('h',cy,w//9,w-w//9)
    elif family=='thick_wall':
        depth = rng.randint(3*t,5*t)
        rect(cx-depth, h//6, cx+depth, h-h//6,'box')
    elif family=='junction':
        for l,r in ((BORDER,cx-gap//2),(cx-gap//2+gap,w-BORDER)):
            for top,bot in ((BORDER,cy-gap//2),(cy-gap//2+gap,h-BORDER)):
                # Leave open chambers at the ends of the cross.
                rect(l+(w//8 if r<cx else 0),top+(h//6 if bot<cy else 0),
                    r-(w//8 if l>cx else 0),bot-(h//6 if top>cy else 0),'box')
        gates.extend([['h',cy-h//5,cx-gap//2,cx-gap//2+gap],
                      ['v',cx+w//5,cy-gap//2,cy-gap//2+gap]])
    elif family=='occluded':
        for i in range(3):
            y = (i+1)*h//4+rng.randint(-h//24,h//24)
            l,r = (BORDER,cx+w//8) if i%2==0 else (cx-w//8,w-BORDER)
            divide('h',y,l,r)
    elif family in ('open','blocks'):
        tiles = [(i,j) for i in range(3) for j in range(2)]
        for i,j in rng.sample(tiles,1 if family=='open' else 5):
            x = (2*i+1)*w//6+rng.randint(-w//24,w//24)
            y = (2*j+1)*h//4+rng.randint(-h//18,h//18)
            dx = rng.randint(12,23) if family=='open' else rng.randint(w//20,w//10)
            dy = rng.randint(12,23) if family=='open' else rng.randint(h//16,h//8)
            rect(x-dx-t//2,y-dy-t//2,x+dx+t//2,y+dy+t//2,'box')
    else:
        raise ValueError(f'Unknown structure {family}')
    return walls,gates,rooms,dead_ends
