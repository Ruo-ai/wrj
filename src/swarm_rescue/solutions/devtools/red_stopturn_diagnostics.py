"""Optional red-only passive diagnostics. Truth never enters a controller call."""
from collections import Counter
from contextlib import contextmanager
import json
import math
from pathlib import Path
import numpy as np


def angle_delta(a,b):
    return (a-b+math.pi)%(2*math.pi)-math.pi


@contextmanager
def recording(destination):
    from swarm_rescue.simulation.gui_map import gui_sr
    from swarm_rescue.simulation.utils.definitions import CollisionTypes
    original_control,original_update=gui_sr.run_control,gui_sr.GuiSR.on_update
    rows={}; handlers=[]; seen=set(); contacts=Counter()
    def observed_update(gui,delta):
        pg=gui._playground
        if id(pg) not in seen:
            seen.add(id(pg))
            for kind,label in ((CollisionTypes.WALL,'wall'),(CollisionTypes.DRONE,'drone')):
                handler=pg.space.add_collision_handler(CollisionTypes.DRONE,kind)
                previous=handler.begin; handlers.append((handler,previous))
                def begin(arbiter,space,data,old=previous,name=label):
                    contacts[name]+=1
                    return old(arbiter,space,data) if old else True
                handler.begin=begin
        return original_update(gui,delta)
    def observed_control(drone):
        # BEFORE run_control: simulator access remains outside truth guard.
        position=np.asarray(drone.true_position(),float); heading=float(drone.true_angle())
        ident=str(drone.identifier)
        row=rows.setdefault(ident,dict(ticks=0,outage_ticks=0,states=Counter(),samples=[],
            distance=0.,outage_distance=0.,outage_abs_rotation=0.,outage_errors={},previous=None))
        motion=getattr(drone,'_segment_motion',None)
        if motion is not None: motion.record_events=True
        command=original_control(drone)
        gps,compass=drone.measured_gps_position(),drone.measured_compass_angle()
        missing=gps is None or compass is None
        row['ticks']+=1; row['outage_ticks']+=int(missing)
        if row['previous'] is not None:
            old_pos,old_angle=row['previous']
            distance=float(np.linalg.norm(position-old_pos)); rotation=abs(angle_delta(heading,old_angle))
            row['distance']+=distance
            if missing:
                row['outage_distance']+=distance;row['outage_abs_rotation']+=rotation
        row['previous']=(position,heading)
        state=motion.state if motion is not None else 'BASELINE'
        row['states'][state]+=1
        sample=dict(tick=row['ticks'],truth=position.tolist(),true_heading=heading,state=state,
            gps_available=gps is not None,compass_available=compass is not None,
            odometry=drone.odometer_values().tolist(),command=command,health=float(drone.drone_health))
        mapper=getattr(drone,'_submission_mapper',None)
        estimates=[('navigation',getattr(drone,'_position',None),getattr(drone,'_heading',None)),
                   ('mapper',getattr(mapper,'position',None),getattr(mapper,'heading',None))]
        if motion is not None:
            estimates.append(('motion',motion.pose.position,motion.pose.heading))
            row['motion_stats']=dict(motion.stats);row['pose_stats']=dict(motion.pose.stats)
            row['events']=list(motion.events)
        for name,estimate,angle in estimates:
            if estimate is None: continue
            error=float(np.linalg.norm(np.asarray(estimate)-position))
            sample[name]=dict(position=list(map(float,estimate)),position_error=error,
                              heading_error=abs(angle_delta(float(angle),heading)) if angle is not None else None)
            if missing:
                aggregate=row['outage_errors'].setdefault(name,dict(count=0,sum=0.,maximum=0.))
                aggregate['count']+=1;aggregate['sum']+=error;aggregate['maximum']=max(aggregate['maximum'],error)
        # Bounded sampling, NOT complete observation replay. Aggregates use every tick.
        if row['ticks']%12==0 and len(row['samples'])<512:
            row['samples'].append(sample)
        return command
    gui_sr.run_control=observed_control;gui_sr.GuiSR.on_update=observed_update
    try: yield
    finally:
        gui_sr.run_control=original_control;gui_sr.GuiSR.on_update=original_update
        for handler,previous in handlers: handler.begin=previous
        for row in rows.values():
            row.pop('previous',None)
            for value in row['outage_errors'].values(): value['mean']=value['sum']/value['count']
        with Path(destination).open('x',encoding='utf-8') as stream:
            json.dump(dict(schema='s6-passive-v1',aligned_before_physics=True,
                complete_sensor_replay=False,collision_contacts=dict(contacts),drones=rows),stream,indent=2,allow_nan=False)
