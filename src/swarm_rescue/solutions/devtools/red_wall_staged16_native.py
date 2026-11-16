"""Native-physics A fixture; fixed fixture goal is not a competition strategy."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np

MODULE='swarm_rescue.solutions.devtools.red_wall_staged16_native'


def run_case(noise=True,scene='single',steps=300):
    from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
    from swarm_rescue.simulation.drone.ground_truth_guard import run_control
    from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
    from swarm_rescue.simulation.utils.misc_data import MiscData
    from swarm_rescue.simulation.elements.normal_wall import NormalWall
    from swarm_rescue.simulation.elements.sensor_disablers import NoGpsZone
    from swarm_rescue.simulation.utils.definitions import CollisionTypes
    from swarm_rescue.solutions.my_drone_red_advanced2 import _M01_PoseFilter,_M11_safe_velocity
    from swarm_rescue.solutions.my_drone_red_wall16 import SingleWallMotion,plain
    from swarm_rescue.solutions.semifinal_localization import wrap
    random.seed(742);np.random.seed(742);size=(800,500)
    class Probe(DroneAbstract):
        def __init__(self,**kw):
            super().__init__(**kw);self.nav=_M01_PoseFilter();self.motion=SingleWallMotion(size);self.previous=None
        def define_message_for_all(self):return None
        def control(self):
            gps,c,odo=self.measured_gps_position(),self.measured_compass_angle(),self.odometer_values()
            d,a,obj=self.lidar_values(),self.lidar_rays_angles(),self.semantic_values()
            self.nav.update(gps,c,odo);self.motion.pose.update(gps,c,odo,d,a,obj)
            delta=np.array([220.,120.])-self.nav.position
            desired=wrap(math.atan2(delta[1],delta[0])-self.nav.heading)
            force,self.previous=_M11_safe_velocity(d,a,desired,self.motion.pose.velocity,self.previous,float(np.linalg.norm(delta)))
            command=dict(forward=float(force[0]),lateral=float(force[1]),rotation=0.,grasper=0)
            self.motion.task_angle=desired
            new=self.motion.step(d,a,desired,float(np.linalg.norm(delta)),obj)
            if new is not None:command.update(new)
            return command
    pg=ClosedPlayground(size);started=time.perf_counter()
    walls=[((-330,-70),(280,90))]
    if scene=='short':walls=[((-120,-15),(-90,-7))]
    if scene=='corner':walls.append(((70,35),(70,165)))
    for start,end in walls:
        wall=NormalWall(start,end,wall_thickness=6);pg.add(wall,wall.wall_coordinates)
    pg.add(NoGpsZone(size=(380,440)),((30,0),0.))
    drone=Probe(identifier=0,misc_data=MiscData(size_area=size,number_drones=1));pg.add(drone,((-290,35),0.))
    drone.set_competition_mode(True)
    if not noise:
        for sensor in drone.sensors:
            if hasattr(sensor,'_noise'):sensor._noise=False
    contacts=[0];handler=pg.space.add_collision_handler(CollisionTypes.DRONE,CollisionTypes.WALL);old=handler.begin
    def begin(a,s,d):
        contacts[0]+=1;return old(a,s,d) if old else True
    handler.begin=begin;pg.step();trace=[]
    try:
        for tick in range(steps):
            drone.elapsed_timestep=tick;p=np.asarray(drone.true_position());heading=float(drone.true_angle())
            command=run_control(drone);m=drone.motion
            trace.append(dict(step=tick,position=p.tolist(),heading=heading,phase=m.align_state,command=command,
                pose=m.pose.position.tolist(),relative_error=m.last_relative_error,absolute=m.pose.absolute,
                measured_velocity=m.pose.velocity.tolist(),angular_increment=m.pose.angular_increment,health=float(drone.drone_health)))
            pg.step({drone:command})
        return plain(dict(scene=scene,noise=noise,seed=742,steps=len(trace),seconds=time.perf_counter()-started,
            contacts=contacts[0],health=float(drone.drone_health),stats=dict(m.wall_stats),events=list(m.wall_events),trace=trace))
    finally:pg.close_window()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--case-json');args=p.parse_args()
    if args.case_json:
        if sys.stdin.readline().strip()!='GO':raise RuntimeError('No handshake')
        row=run_case(**json.loads(args.case_json))
        with args.output.open('x',encoding='utf-8') as f:json.dump(row,f,indent=2)
        return
    from swarm_rescue.solutions.devtools.optimization.evaluation.resources import run_limited
    args.output.mkdir(parents=True,exist_ok=False);rows=[]
    for name,params in [('single_ideal',dict(noise=False)),('single_noisy',dict()),('corner_noisy',dict(scene='corner')),('short_noisy',dict(scene='short'))]:
        path=args.output/(name+'.json');argv=[MODULE,'--output',str(path),'--case-json',json.dumps(params)]
        boot=(f'import runpy,sys;sys.path[:]={sys.path!r};sys.prefix={sys.prefix!r};sys.exec_prefix={sys.exec_prefix!r};sys.argv={argv!r};runpy.run_module({MODULE!r},run_name="__main__")')
        command=[sys.executable,'-B','-u','-S','-X','utf8','-c',boot]
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with path.with_suffix('.log').open('x',encoding='utf-8') as log:limits=run_limited(command,env,log,timeout=45)
        errors=path.with_suffix('.log').read_text(encoding='utf-8',errors='replace')
        row=json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(error=errors)
        row.update(case=name,valid=limits['exit_code']==0 and not any(s in errors for s in ('Traceback','Exception ignored from cffi callback')),resources=limits)
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('trace','events')}),flush=True)
    with (args.output/'results.json').open('x',encoding='utf-8') as f:json.dump(rows,f,indent=2)


if __name__=='__main__':main()
