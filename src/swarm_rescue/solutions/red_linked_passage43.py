"""Fresh on-wire receipt checks and current-scan fine navigation only."""
import heapq
import math
import numpy as np
from swarm_rescue.solutions.my_drone_red_rn7 import LocalPassage, rotate


def valid_receipt(value, number, clock):
    if not (isinstance(value, tuple) and len(value)==2 and type(value[0]) is int
            and 0 <= value[0] <= clock and isinstance(value[1], tuple)
            and len(value[1]) <= number):
        return False
    seen=set()
    for item in value[1]:
        if not (isinstance(item,tuple) and len(item)==3 and type(item[0]) is int
                and 0<=item[0]<number and item[0] not in seen and type(item[1]) is int
                and 0<=item[1]<=value[0] and isinstance(item[2],str) and len(item[2])==64
                and all(c in '0123456789abcdef' for c in item[2])):
            return False
        seen.add(item[0])
    return True


def receipt_delivered(receipt, own, snapshot, digest, clock, sealed=False):
    """At most one 90-step live pipeline; finals require exact payload digest."""
    if receipt is None or snapshot is None or clock-receipt[0]>120:
        return False
    row=next((r for r in receipt[1] if r[0]==own),None)
    if row is None or row[1]>snapshot[1]:return False
    if row[1]==snapshot[1]:return row[2]==digest
    return not sealed and 0<=snapshot[1]-row[1]<=90 and clock-row[1]<=120


def fine_path(fine, target, limit=190.):
    """A* through observed free space, including swept segment validation."""
    target=np.asarray(target,float).copy()
    if not np.isfinite(target).all() or np.linalg.norm(target)>limit:return []
    start=fine.cell(np.zeros(2));end=fine.cell(target);clear=fine.clearance
    if min(clear[start],clear[end])<fine.PLAN_RADIUS:return []
    if fine.line_clear(np.zeros(2),target):return [np.zeros(2),target.copy()]
    costs={start:0.};parents={};heap=[(0.,0.,start)];count=0
    while heap and count<6000:
        _,cost,cell=heapq.heappop(heap)
        if cost!=costs[cell]:continue
        count+=1
        if cell==end:
            cells=[cell]
            while cells[-1]!=start:cells.append(parents[cells[-1]])
            points=[np.zeros(2)]+[fine.point(c) for c in reversed(cells[:-1])]
            path=[points[0]];i=0
            while i<len(points)-1:
                j=min(i+20,len(points)-1)
                while j>i+1 and not fine.line_clear(points[i],points[j]):j-=1
                if not fine.line_clear(points[i],points[j]):return []
                path.append(points[j]);i=j
            return path
        r,c=cell
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
            nxt=(r+dr,c+dc)
            if not (0<=nxt[0]<fine.shape[0] and 0<=nxt[1]<fine.shape[1]):continue
            if clear[nxt]<fine.PLAN_RADIUS:continue
            if dr and dc and min(clear[r+dr,c],clear[r,c+dc])<fine.PLAN_RADIUS:continue
            new=cost+fine.CELL*(math.sqrt(2) if dr and dc else 1.)
            if new>limit*1.6 or new>=costs.get(nxt,float('inf')):continue
            costs[nxt]=new;parents[nxt]=cell
            estimate=fine.CELL*math.hypot(nxt[0]-end[0],nxt[1]-end[1])
            heapq.heappush(heap,(new+estimate,new,nxt))
    return []


def observed_approach(distances, angles, objects, position, heading, goal):
    fine=LocalPassage(distances,angles,objects)
    target=rotate(np.asarray(goal)-position,-heading)
    distance=float(np.linalg.norm(target))
    if not 30<distance<=190:return fine,[]
    # Preserve the original unknown-region goal; a partial leg exposes the bend.
    path=fine_path(fine,target)
    if path:return fine,path
    # The turn itself may still be occluded. Approach along an observed safe
    # corridor rather than shrinking a straight ray that points at a wall.
    cells=np.argwhere(fine.clearance>=fine.PLAN_RADIUS+1.)
    points=np.column_stack(((cells[:,1]+.5)*fine.CELL-fine.width/2,
                            fine.height/2-(cells[:,0]+.5)*fine.CELL))
    lengths=np.linalg.norm(points,axis=1);remaining=np.linalg.norm(points-target,axis=1)
    eligible=(lengths>=30)&(lengths<=170)&(remaining<=distance-30)
    choices=np.flatnonzero(eligible);chosen=[]
    for ix in choices[np.argsort((remaining+.12*lengths)[choices])]:
        point=points[ix]
        if any(np.linalg.norm(point-p)<12 for p in chosen):continue
        chosen.append(point)
        path=fine_path(fine,point)
        if path:return fine,path
        if len(chosen)>=6:break
    return fine,[]
