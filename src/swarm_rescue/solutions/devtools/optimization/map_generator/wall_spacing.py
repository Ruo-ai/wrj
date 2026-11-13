"""Surface spacing: touching joints are allowed, accidental slits are not."""
import math

def surface_gap(a,b):
    return math.hypot(max(a[0]-b[2],b[0]-a[2],0),max(a[1]-b[3],b[1]-a[3],0))

def check_spacing(spec,minimum=40):
    walls=[row[1:] for row in spec['walls']]
    w,h=spec['size']
    borders=[(0,0,6,h),(w-6,0,w,h),(0,0,w,6),(0,h-6,w,h)]
    for i,a in enumerate(walls):
        for b in walls[i+1:]+borders:
            gap=surface_gap(a,b)
            if 0<gap<minimum:
                # A third wall may fill this interval at a T/cross joint.
                if max(a[0],b[0])<=min(a[2],b[2]):
                    top,bottom=min(a[3],b[3]),max(a[1],b[1])
                    if any(c[0]<=max(a[0],b[0]) and c[2]>=min(a[2],b[2]) and c[1]<=top and c[3]>=bottom for c in walls):
                        continue
                if max(a[1],b[1])<=min(a[3],b[3]):
                    left,right=min(a[2],b[2]),max(a[0],b[0])
                    if any(c[1]<=max(a[1],b[1]) and c[3]>=min(a[3],b[3]) and c[0]<=left and c[2]>=right for c in walls):
                        continue
                raise ValueError(f'Accidental wall slit: {gap:.2f} px < {minimum} px: {a}, {b}')

def audit_official():
    import json
    from .semifinal import REPO
    result=[]
    for path in sorted((REPO/'map_data').glob('map_*.json')):
        if path.stem.endswith('_yh'): continue
        doc=json.loads(path.read_text(encoding='utf-8'))
        rects,thickness=[],[]
        diagonal=0
        for e in doc['elements']:
            if e['type']=='rect_wall': rects.append([*e['top_left'],*e['bottom_right']])
            elif e['type']=='line_wall':
                (x,y),(xx,yy)=e['points']; t=e['thickness']; thickness.append(t)
                if x!=xx and y!=yy:
                    diagonal+=1; continue
                rects.append([min(x,xx)-t/2,min(y,yy)-t/2,max(x,xx)+t/2,max(y,yy)+t/2])
        gaps=[surface_gap(a,b) for i,a in enumerate(rects) for b in rects[i+1:]]
        positive=[g for g in gaps if g>0]
        result.append(dict(map=path.name,line_thickness=sorted(set(thickness)),
            minimum_positive_aabb_gap=min(positive,default=None),
            pairs_under_40=sum(g<40 for g in positive),diagonal_lines_skipped=diagonal))
    return result

if __name__=='__main__':
    import json
    print(json.dumps(audit_official(),ensure_ascii=False,indent=2))


