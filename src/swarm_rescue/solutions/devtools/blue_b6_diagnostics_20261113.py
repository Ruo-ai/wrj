"""Opt-in B6 extension of existing lifecycle diagnostics, in isolated workers.

No engine/scoring/control output is changed. Truth is confined to the inherited
offline observer; new probe fields use only legal sensors and controller state.
"""
import base64
from collections import Counter
import time
import zlib
import numpy as np
from swarm_rescue.solutions.devtools.blue_lifecycle import instrument as original_instrument
from swarm_rescue.solutions.devtools.blue_lifecycle import LifecycleObserver as OriginalObserver


def pack(array):
    a=np.ascontiguousarray(array)
    return dict(shape=list(a.shape),dtype=str(a.dtype),zlib_b64=base64.b64encode(zlib.compress(a.tobytes(),1)).decode('ascii'))


def unpack(record):
    return np.frombuffer(zlib.decompress(base64.b64decode(record['zlib_b64'])),dtype=record['dtype']).reshape(record['shape'])


def instrument(base):
    parent=original_instrument(base)
    class Probe(parent):
        def __init__(self,**kwargs):
            super().__init__(**kwargs);self._b6_observations=[];self._b6_collisions=Counter()
            self._b6_diagnostic_seconds=0.;self._b6_last_carry=False

        def collide_wall(self):
            self._b6_collisions['wall_pre_solve_callbacks']+=1
            return super().collide_wall()

        def collide_drone(self):
            self._b6_collisions['drone_pre_solve_callbacks']+=1
            return super().collide_drone()

        def control(self):
            command=super().control()
            start=time.perf_counter();frame=self._probe_frame;carrying=bool(self.grasped_bombs())
            frame.update(heading=float(getattr(self,'_heading',0.)),
                b6_pose=getattr(self,'_b6_pose_info',None),collisions=dict(self._b6_collisions),
                b6_portal_reason=getattr(getattr(self,'_b6_portal',None),'reason',None))
            tick=int(self.elapsed_timestep)
            # Bounded full snapshots, plus possession transitions. No per-step
            # stdout and no every-step connected-component computation.
            if (tick%120==0 or carrying!=self._b6_last_carry) and getattr(self,'_position',None) is not None:
                from scipy.ndimage import label
                grid=self._grid;allowed,_=grid.layers();allowed[grid.cell(self._position)]=True
                labels,_=label(allowed);component=labels[grid.cell(self._position)]
                target=self._target;goal_cell=grid.cell(target) if target is not None else None
                objects=[dict(kind=getattr(s.entity_type,'name','?'),distance=float(s.distance),angle=float(s.angle),
                              grasped=bool(getattr(s,'grasped',False))) for s in self.semantic_values() or ()]
                raw=self.lidar_values();angles=self.lidar_rays_angles();odo=self.odometer_values()
                gps=self.measured_gps_position();compass=self.measured_compass_angle()
                self._b6_observations.append(dict(step=tick,position=self._position.tolist(),heading=float(self._heading),
                    gps=None if gps is None else np.asarray(gps).tolist(),compass=None if compass is None else float(compass),
                    odometry=None if odo is None else np.asarray(odo).tolist(),lidar=None if raw is None else np.asarray(raw).tolist(),
                    angles=np.asarray(angles).tolist(),semantic=objects,carrying=carrying,health=int(self.drone_health),
                    target=None if target is None else np.asarray(target).tolist(),route=[np.asarray(p).tolist() for p in self._route],
                    source=frame['source'],command=dict(command),wall_callbacks=dict(self._b6_collisions),
                    own_free=pack(grid.free),own_evidence=pack(grid.evidence),merged_allowed=pack(allowed),
                    component_size=int((labels==component).sum()) if component else 0,
                    target_in_component=bool(goal_cell is not None and component and labels[goal_cell]==component),
                    cell=grid.CELL,size=[grid.width,grid.height],peer_reports={str(k):dict(age=tick-v[0],position=v[1].tolist(),mode=v[2]) for k,v in self._peers.items()}))
                self._b6_observations=self._b6_observations[-64:]
            self._b6_last_carry=carrying
            self._b6_diagnostic_seconds+=time.perf_counter()-start
            return command
    Probe.__name__=base.__name__;Probe.__module__=base.__module__
    return Probe


class Observer(OriginalObserver):
    def __init__(self):
        super().__init__();self._b6_drones={}

    def begin(self,gui):
        # Keep a passive reference to destroyed drones so their last raw
        # snapshots are not lost when the engine removes them from its list.
        self._b6_drones.update({str(d.identifier):d for d in gui._drones})
        return super().begin(gui)

    def export(self,gui):
        data=super().export(gui)
        data['b6_diagnostics']=dict(version='blue-only-20261113',
            semantics='collision callback counts are not distinct contact episodes; snapshot component uses current merged allowed grid',
            drones={str(d.identifier):dict(snapshots=getattr(d,'_b6_observations',[]),
                collision_callbacks=dict(getattr(d,'_b6_collisions',{})),
                extra_seconds=getattr(d,'_b6_diagnostic_seconds',0.)) for d in self._b6_drones.values()})
        return data
