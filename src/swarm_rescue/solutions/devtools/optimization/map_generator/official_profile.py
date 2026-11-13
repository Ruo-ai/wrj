"""Read official editor data for offline distribution calibration, never for control."""
import argparse
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[6]


def summarize(document):
    w,h = document['canvas']['width'],document['canvas']['height']
    mask = np.zeros((h,w),np.uint8)
    boxes = np.zeros_like(mask)
    lines,rects,border_count,thicknesses = [],[],0,[]
    markers = {}
    for element in document['elements']:
        kind = element['type']
        if kind=='line_wall':
            (x0,y0),(x1,y1) = element['points']
            t = element['thickness']
            thicknesses.append(t)
            # This is an editor raster proxy, not the official scoring mask.
            cv2.line(mask,(int(x0),int(y0)),(int(x1),int(y1)),1,t)
            border = ((abs(y1-y0)<1 and min(y0,h-y0)<=6 and abs(x1-x0)>.8*w) or
                      (abs(x1-x0)<1 and min(x0,w-x0)<=6 and abs(y1-y0)>.8*h))
            border_count += int(border)
            if not border:
                lines.append(element)
        elif kind=='rect_wall':
            l,t = element['top_left']
            r,b = element['bottom_right']
            boxes[t:b,l:r] = 1
            mask[t:b,l:r] = 1
            rects.append([r-l,b-t])
        elif kind in ('start_area','disposal_center'):
            l,t = element['top_left']
            r,b = element['bottom_right']
            markers[kind] = dict(center=[(l+r)/2/w,(t+b)/2/h],size=[(r-l)/w,(b-t)/h])
    # Closing 75px gaps yields a reproducible compartment proxy, not semantic rooms.
    occupied = mask.copy()
    occupied[:6] = occupied[-6:] = 1
    occupied[:,:6] = occupied[:,-6:] = 1
    closed = cv2.morphologyEx(occupied,cv2.MORPH_CLOSE,np.ones((75,75),np.uint8))
    _,_,stats,_ = cv2.connectedComponentsWithStats(1-closed,connectivity=4)
    return dict(size=[w,h],internal_line_count=len(lines),border_line_count=border_count,
                box_count=len(rects),line_thicknesses=sorted(set(thicknesses)),
                wall_fraction_proxy=round(float(mask.mean()),6),
                box_fraction=round(float(boxes.mean()),6),box_sizes=rects,
                compartment_proxy_75=sum(int(v)>=900 for v in stats[1:,cv2.CC_STAT_AREA]),
                markers=markers)


@lru_cache(maxsize=1)
def read_profiles():
    profiles = []
    for path in [REPO/'map_data'/f'map_{n:02d}.json' for n in (1,2,3,4,5,7)]:
        if not re.fullmatch(r'map_[0-9]+\.json',path.name):
            continue  # Generated *_yh maps must never enter official calibration.
        raw = path.read_bytes()
        row = summarize(json.loads(raw.decode('utf-8')))
        row.update(source=path.name,source_sha256=hashlib.sha256(raw).hexdigest())
        profiles.append(row)
    if len(profiles)<2:
        raise ValueError('Official map_data references are missing')
    return profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    result = json.dumps(read_profiles(),ensure_ascii=False,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as stream:
            stream.write(result)
    else:
        print(result)


if __name__=='__main__':
    main()
