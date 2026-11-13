"""Offline native-physics micro-scenes. Never imported by competition policy.

Stock ClosedPlayground, DroneBase hull, sensors, forces and damage are used.
No benchmark point/width/seed is passed into NarrowPassage. Fixture goal is an
odometric heading, not a true current pose. This is not a full-game score.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile
import numpy as np
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.drone.ground_truth_guard import run_control
from swarm_rescue.simulation.elements.normal_wall import NormalWall
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.definitions import CollisionTypes
from swarm_rescue.solutions.my_drone_blue_narrow import NarrowPassage,HULL,lateral_support
from swarm_rescue.solutions.my_drone_blue_transport import SteeringTable


class Probe(DroneAbstract):
    def __init__(self,variant,**kwargs):
        super().__init__(**kwargs)
        self.variant=variant;self.gate=NarrowPassage();self.steering=SteeringTable()
        self.heading=0.;self.last=None
    def define_message_for_all(self):return None
    def control(self):
        d=np.asarray(self.lidar_values(),float);a=np.asarray(self.lidar_rays_angles(),float)
        odo=self.odometer_values()
        compass=self.measured_compass_angle()
        if odo is not None and np.isfinite(odo).all():self.heading+=odo[2]
        if compass is not None:self.heading+=.2*((compass-self.heading+math.pi)%(2*math.pi)-math.pi)
        desired=(0-self.heading+math.pi)%(2*math.pi)-math.pi
        if self.variant=='narrow':
            cmd=self.gate.step(d,a,odo,self.elapsed_timestep,desired,health=self.drone_health)
            if cmd is not None:self.last=cmd;return cmd
        selected=self.steering.choose(d,a,desired)
        cmd=dict(forward=0.,lateral=0.,rotation=0.,grasper=0)
        if selected:
            angle,gap=selected;speed=.72*np.clip((gap-31)/65,0,1) if abs(angle)<=1.2 else 0.
            cmd.update(forward=float(speed*math.cos(angle)),lateral=float(speed*math.sin(angle)),
                       rotation=float(np.clip(angle*1.5,-.85,.85)))
        self.last=cmd;return cmd


def run_case(variant,width,length,offset,angle,seed,noise=True,dropout=False,steps=600):
    random.seed(seed);np.random.seed(seed)
    started=time.perf_counter();pg=ClosedPlayground((700,420));contacts=Counter()
    drone=Probe(variant,identifier=0,misc_data=MiscData(size_area=(700,420),number_drones=1))
    pg.add(drone,((-55.,offset),math.radians(angle)))
    drone.set_competition_mode(True)
    for side in (-1,1):
        wall=NormalWall(pos_start=(0,side*(width/2+3)),pos_end=(length,side*(width/2+3)),wall_thickness=6)
        pg.add(wall,wall.wall_coordinates)
    physical_walls=[e for e in pg.elements if isinstance(e,NormalWall) and abs(e.position.x-length/2)<1]
    faces=[]
    for wall in physical_walls:
        ys=[wall.pm_body.local_to_world(v).y for s in wall.pm_shapes for v in s.get_vertices()]
        faces.append(min(ys) if wall.position.y>0 else max(ys))
    actual_gap=max(faces)-min(faces)
    handler=pg.space.add_collision_handler(CollisionTypes.DRONE,CollisionTypes.WALL)
    original=handler.begin
    def begin(arbiter,space,data):
        contacts['contact_episodes']+=1
        return original(arbiter,space,data) if original else True
    handler.begin=begin
    if not noise:
        for sensor in drone.sensors:
            if hasattr(sensor,'_noise'):sensor._noise=False
    pg.step();trace=[];entered=False;passed=False;missing=0;escaped_side=False
    try:
        for tick in range(steps):
            drone.elapsed_timestep=tick
            # Sensor dropout fixture, not a runtime oracle; both algorithms see
            # identical availability schedule, all legal lidar remains enabled.
            if dropout:
                drone.gps()._disabled=20<=tick<steps-20
                drone.compass()._disabled=20<=tick<steps-20
            missing+=int(drone.measured_gps_position() is None)
            cmd=run_control(drone)
            pg.step({drone:cmd})
            p=drone.true_position();heading=float(drone.true_angle())
            if tick%5==0:trace.append(dict(step=tick,x=float(p[0]),y=float(p[1]),heading=heading,
                health=drone.drone_health,state=drone.gate.state,command=cmd,
                geometry=drone.gate.last_geometry,
                scan=np.asarray(drone.lidar_values()).tolist() if tick<50 else None))
            entered |= p[0]>15 and abs(p[1])<width/2
            if 15<p[0]<length-15 and abs(p[1])>width/2:escaped_side=True
            # Complete hull, not just the center, must have cleared the far end.
            if p[0]>length+32 and abs(p[1])<width/2+10 and entered and not escaped_side:
                passed=True;break
            if drone.drone_health<=0:break
        return dict(variant=variant,width=width,actual_gap=actual_gap,length=length,offset=offset,angle=angle,seed=seed,
                    noise=noise,dropout=dropout,steps=tick+1,seconds=time.perf_counter()-started,
                    entered=bool(entered),passed=bool(passed),bypassed=escaped_side,contact_episodes=contacts['contact_episodes'],
                    health_lost=25-drone.drone_health,missing_steps=missing,stats=dict(drone.gate.stats),
                    final_state=drone.gate.state,reason=drone.gate.reason,trace=trace)
    finally:pg.close_window()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--case-json')
    parser.add_argument('--wait-go',action='store_true')
    args=parser.parse_args()
    if args.case_json:
        if args.wait_go and sys.stdin.readline().strip()!='GO':raise RuntimeError('Missing resource guard handshake')
        row=run_case(**json.loads(args.case_json))
        args.output.write_text(json.dumps(row,indent=2),encoding='utf-8')
        return
    args.output.mkdir(parents=True,exist_ok=False)
    from .optimization.evaluation.runner import source_receipt,validate_sources,REPO
    from .optimization.evaluation.resources import run_limited
    receipt=source_receipt([])
    (args.output/'sources.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    with zipfile.ZipFile(args.output/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in receipt:z.write(REPO/name,name)
    from swarm_rescue.simulation.drone.drone_base import DroneBase
    base=DroneBase();actual=np.array([tuple(v) for v in base.pm_shapes[0].get_vertices()])
    np.testing.assert_array_equal(actual,HULL)
    geometry=dict(vertices=actual.tolist(),radius=base.radius,
                  widths={str(a):lateral_support(math.radians(a))[1]-lateral_support(math.radians(a))[0]
                          for a in (0,5,15,30,45,90)},native_engine=True)
    (args.output/'geometry.json').write_text(json.dumps(geometry,indent=2),encoding='utf-8')
    configs=[(27,90,0,0,False,False),(27,90,0,0,True,False),
             (35,90,0,0,True,False),(55,90,0,0,True,False)]
    if not args.quick:
        configs += [(27,90,3,0,True,False),(27,90,0,12,True,False),
                    (35,160,0,0,True,True),(55,160,5,12,True,True)]
    rows=[]
    for variant in ('compute_steering','narrow'):
        for config in configs:
            for seed in ([742] if args.quick else [742,743,744]):
                params=dict(zip(('width','length','offset','angle','noise','dropout'),config),variant=variant,seed=seed)
                case=args.output/f'case_{len(rows):03d}.json'
                # Arcade's context cache is process-global: isolate native scenes
                # just like full evaluation workers, preserving failed stderr.
                validate_sources(receipt)
                env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
                with case.with_suffix('.log').open('w',encoding='utf-8') as log:
                    resources=run_limited([sys.executable,'-B','-m',__spec__.name,'--output',str(case),
                                         '--case-json',json.dumps(params),'--wait-go'],env,log,timeout=30)
                log_text=case.with_suffix('.log').read_text(encoding='utf-8',errors='replace')
                invalid=resources['exit_code']!=0 or any(t in log_text for t in ('Traceback','Exception ignored from cffi callback','AssertionError'))
                if invalid:
                    row=dict(params,entered=False,passed=False,contact_episodes=None,health_lost=None,
                             error=log_text,trace=[])
                else:row=json.loads(case.read_text(encoding='utf-8'))
                row.update(resources=resources,valid=not invalid)
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k!='trace'}),flush=True)
                (args.output/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,len(configs),figsize=(3.3*len(configs),6),squeeze=False)
    for ax,(variant,config) in zip(axes.ravel(),[(v,c) for v in ('compute_steering','narrow') for c in configs]):
        width,length,offset,angle,noise,dropout=config
        group=[r for r in rows if (r['variant'],r['width'],r['length'],r['offset'],r['angle'],r['noise'],r['dropout'])==(variant,*config)]
        for r in group:ax.plot([p['x'] for p in r['trace']],[p['y'] for p in r['trace']],label=str(r['seed']))
        ax.plot([0,length],[width/2,width/2],'k',lw=4);ax.plot([0,length],[-width/2,-width/2],'k',lw=4)
        ax.set(xlim=(-70,length+30),ylim=(-65,65),title=f'{variant}\nW={width} noise={noise} pass={sum(r["passed"] for r in group)}/{len(group)}')
        ax.set_aspect('equal');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(args.output/'trajectories.png',dpi=120);plt.close(fig)


if __name__=='__main__':main()
