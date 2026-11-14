"""RN7 native component checks; fixture geometry/truth is offline only."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np

MODULE='swarm_rescue.solutions.devtools.red_rn7_native'


def run_case(width=50,noise=True,steps=360,deadend=False):
    from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
    from swarm_rescue.simulation.drone.ground_truth_guard import run_control
    from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
    from swarm_rescue.simulation.utils.misc_data import MiscData
    from swarm_rescue.simulation.elements.normal_wall import NormalWall
    from swarm_rescue.simulation.utils.definitions import CollisionTypes
    from swarm_rescue.solutions.my_drone_red_advanced2 import _M11_safe_velocity
    from swarm_rescue.solutions.my_drone_red_rn7 import PassageMotion
    random.seed(742);np.random.seed(742)
    class Coarse:
        visits=np.zeros((1,1),int)
        def cell(self,p):return (0,0)
    class Probe(DroneAbstract):
        def __init__(self,**kw):
            super().__init__(**kw);self.narrow=PassageMotion();self.previous=None
            self.narrow.cooldown=0
        def define_message_for_all(self):return None
        def control(self):
            n=self.narrow;n.update(self.odometer_values())
            d,a=self.lidar_values(),self.lidar_rays_angles()
            force,self.previous=_M11_safe_velocity(d,a,0.,n.velocity,self.previous,180.)
            command=dict(forward=float(force[0]),lateral=float(force[1]),rotation=0.,grasper=0)
            local=n.command(d,a,Coarse(),np.zeros(2),0.,0.,self.semantic_values())
            if local is not None:command.update(local)
            return command
    pg=ClosedPlayground((700,420));length=160;start=time.perf_counter()
    for y in (width/2+3,-width/2-3):
        wall=NormalWall((0,y),(length,y),wall_thickness=6);pg.add(wall,wall.wall_coordinates)
    if deadend:
        wall=NormalWall((length,-width/2-3),(length,width/2+3),wall_thickness=6)
        pg.add(wall,wall.wall_coordinates)
    drone=Probe(identifier=0,misc_data=MiscData(size_area=(700,420),number_drones=1))
    pg.add(drone,((-55,0),0.));drone.set_competition_mode(True)
    if not noise:
        for sensor in drone.sensors:
            if hasattr(sensor,'_noise'):sensor._noise=False
    contacts=[0];handler=pg.space.add_collision_handler(CollisionTypes.DRONE,CollisionTypes.WALL)
    old=handler.begin
    def begin(a,s,d):
        contacts[0]+=1;return old(a,s,d) if old else True
    handler.begin=begin;pg.step();trace=[]
    try:
        for tick in range(steps):
            drone.elapsed_timestep=tick
            p=drone.true_position().copy();heading=float(drone.true_angle())
            command=run_control(drone)
            trace.append(dict(tick=tick,x=float(p[0]),y=float(p[1]),heading=heading,
                              command=command,active=bool(drone.narrow.path),health=float(drone.drone_health)))
            pg.step({drone:command})
            if p[0]>length+32:break
        inside=[r for r in trace if 0<=r['x']<=length]
        bypassed=any(abs(r['y'])>width/2 for r in inside)
        return dict(width=width,noise=noise,deadend=deadend,steps=len(trace),
                    seconds=time.perf_counter()-start,passed=trace[-1]['x']>length+32 and bool(inside) and not bypassed,
                    bypassed=bypassed,
                    contacts=contacts[0],health=float(drone.drone_health),
                    inside_max_lateral=max((abs(r['y']) for r in inside),default=None),
                    max_x=max(r['x'] for r in trace),stats=dict(drone.narrow.stats),
                    events=list(drone.narrow.events),trace=trace)
    finally:pg.close_window()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--case-json');args=p.parse_args()
    if args.case_json:
        if sys.stdin.readline().strip()!='GO':raise RuntimeError('No handshake')
        row=run_case(**json.loads(args.case_json))
        with args.output.open('x',encoding='utf-8') as f:json.dump(row,f,indent=2)
        return
    from swarm_rescue.solutions.devtools.optimization.evaluation.resources import run_limited
    args.output.mkdir(parents=True,exist_ok=False);rows=[]
    cases=[('w50_ideal',dict(width=50,noise=False)),('w50_noisy',dict(width=50)),
           ('w80_noisy',dict(width=80)),('w25_noisy',dict(width=25)),
           ('deadend_noisy',dict(width=55,deadend=True))]
    for name,params in cases:
        path=args.output/(name+'.json')
        argv=[MODULE,'--output',str(path),'--case-json',json.dumps(params)]
        boot=(f'import runpy,sys;sys.path[:]={sys.path!r};sys.prefix={sys.prefix!r};'
              f'sys.exec_prefix={sys.exec_prefix!r};sys.argv={argv!r};runpy.run_module({MODULE!r},run_name="__main__")')
        command=[sys.executable,'-B','-u','-S','-X','utf8','-c',boot]
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with path.with_suffix('.log').open('x',encoding='utf-8') as log:
            limits=run_limited(command,env,log,timeout=45)
        errors=path.with_suffix('.log').read_text(encoding='utf-8',errors='replace')
        row=json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(error=errors)
        row.update(case=name,valid=limits['exit_code']==0 and not any(x in errors for x in ('Traceback','Exception ignored from cffi callback')),resources=limits)
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('trace','events')}),flush=True)
    with (args.output/'results.json').open('x',encoding='utf-8') as f:json.dump(rows,f,indent=2)


if __name__=='__main__':main()
