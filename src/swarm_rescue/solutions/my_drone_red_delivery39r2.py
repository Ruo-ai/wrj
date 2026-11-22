"""Keep return breadcrumbs in Continuous38's corrected coordinate frame.

    Read the estimator's existing recovery result; never change its pose,
    pending scans, mapper, final renderer or navigation occupancy values.
"""
import numpy as np
from swarm_rescue.solutions.red_delivery39 import WalkGraph39
from swarm_rescue.solutions.my_drone_red_delivery39r1 import MyDroneRedDelivery39R1


class CorrectedWalkGraph39(WalkGraph39):
    def __init__(self, mapper):
        super().__init__()
        self.mapper = mapper
        self.birth_steps = []
        self.pose_history = {}
        self.recoveries = []

    def apply_recovery(self, corrected, step):
        keys = sorted(t for t in corrected if t in self.pose_history)
        if not keys:
            return
        shifts = np.array([corrected[t][0]-self.pose_history[t] for t in keys])
        moved = 0; max_move = 0.
        for i,t in enumerate(self.birth_steps):
            if keys[0]-3 <= t < step:
                delta = np.array([np.interp(t,keys,shifts[:,axis]) for axis in range(2)])
                self.points[i] += delta
                moved += 1; max_move = max(max_move,float(np.linalg.norm(delta)))
        for t in keys:self.pose_history[t]=np.asarray(corrected[t][0],float).copy()
        self.bins = {}
        for i,p in enumerate(self.points):self.bins.setdefault(tuple(np.floor(p/50).astype(int)),[]).append(i)
        for i,neighbors in enumerate(self.edges):
            for j in neighbors:neighbors[j]=float(np.linalg.norm(self.points[i]-self.points[j]))
        self.stats['coordinate_recoveries'] += 1
        self.stats['corrected_walk_nodes'] += moved
        self.recoveries.append(dict(step=step,frames=len(keys),moved_nodes=moved,max_shift=max_move))

    def observe(self, position, heading, distances, angles, step):
        recovery = self.mapper.recovery38
        if recovery is not None:self.apply_recovery(recovery,step)
        before = len(self.points)
        super().observe(position,heading,distances,angles,step)
        if len(self.points)>before:self.birth_steps.append(int(step))
        if position is not None:self.pose_history[int(step)]=np.asarray(position,float).copy()


class MyDroneRedDelivery39R2(MyDroneRedDelivery39R1):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._walk39 = CorrectedWalkGraph39(self._submission_mapper)
