"""Small native B8 motion-component probes, never competition scores.

Stock physical shapes, force application, sensors, grasp and disposal callbacks.
The probe has a nominal relative travel direction; no truth is used by control.
Full B0/B8 search/planning/terminal integration is evaluated separately.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np


def run_case(variant, scenario, seed=742, steps=160):
    from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
    from swarm_rescue.simulation.drone.ground_truth_guard import run_control
    from swarm_rescue.simulation.elements.bomb import Bomb
    from swarm_rescue.simulation.elements.disposal_center import DisposalCenter
    from swarm_rescue.simulation.elements.normal_wall import NormalWall
    from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
    from swarm_rescue.simulation.utils.misc_data import MiscData
    from swarm_rescue.simulation.utils.definitions import CollisionTypes
    from swarm_rescue.solutions.my_drone_blue_transport import SteeringTable
    from swarm_rescue.solutions.blue_nav_b8 import SpeedGovernor, wrap, BODY_RADIUS, PAYLOAD_RADIUS
    random.seed(seed);np.random.seed(seed)
    loaded=scenario.startswith('payload'); started=time.perf_counter()

    class Probe(DroneAbstract):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            self.heading=0.;self.travel=np.zeros(2);self.governor=SpeedGovernor()
            self.steering=SteeringTable();self.last={};self.turned=False
        def define_message_for_all(self):return None
        def control(self):
            odo=self.odometer_values();compass=self.measured_compass_angle()
            if odo is not None and np.isfinite(odo).all():
                self.travel+=odo[0]*np.array([math.cos(self.heading+odo[1]),math.sin(self.heading+odo[1])])
                self.heading=wrap(self.heading+odo[2])
            if compass is not None and np.isfinite(compass):self.heading+=.2*wrap(compass-self.heading)
            # A relative fixture waypoint sequence, not current true position.
            self.turned |= bool(scenario=='payload_turn' and self.travel[0]>125)
            desired_heading=math.pi/2 if self.turned else 0.
            desired=wrap(desired_heading-self.heading)
            d=np.asarray(self.lidar_values(),float);a=np.asarray(self.lidar_rays_angles(),float)
            # Fixed-direction probe isolates braking rather than letting the
            # baseline steer around the test wall; production retains B0 steering.
            angle=desired
            x=d*np.cos(a-angle);y=d*np.sin(a-angle)
            mask=(x>0)&(np.abs(y)<(43 if loaded else 31))
            gap=float(x[mask].min()) if mask.any() else 230.
            speed=(.62 if loaded else .72)*np.clip((gap-(43 if loaded else 31))/65,0,1)
            cmd=dict(forward=float(speed*math.cos(angle)),lateral=float(speed*math.sin(angle)),
                     rotation=float(np.clip(angle*1.5,-.85,.85)),grasper=int(loaded))
            if variant=='b8':
                cmd=self.governor.update(cmd,d,a,odo,carrying=bool(self.grasped_bombs()),
                    turn=abs(desired),pose_uncertain=compass is None)
            self.last=cmd.copy();return cmd

    pg=ClosedPlayground((800,500));contacts=Counter()
    drone=Probe(identifier=0,misc_data=MiscData(size_area=(800,500),number_drones=1))
    pg.add(drone,((-240.,0.),0.));drone.set_competition_mode(True)
    center=DisposalCenter((40,70));pg.add(center,((335.,170.),0.))
    bomb=None
    if loaded:
        bomb=Bomb(center);pg.add(bomb,((-215.,0.),0.))
    if scenario=='empty_wall':
        wall=NormalWall(pos_start=(5,-230),pos_end=(5,230),wall_thickness=6)
        pg.add(wall,wall.wall_coordinates)
    for kind in (CollisionTypes.WALL,CollisionTypes.DRONE):
        handler=pg.space.add_collision_handler(CollisionTypes.DRONE,kind);original=handler.begin
        def begin(arbiter,space,data,original=original,kind=kind):
            contacts[str(kind)]+=1
            return original(arbiter,space,data) if original else True
        handler.begin=begin
    pg.step({drone:dict(forward=0.,lateral=0.,rotation=0.,grasper=int(loaded))})
    # Reach the grasper's real contact shell with ordinary commands; a bounding
    # circle overlap alone is NOT evidence that the two native hulls touch.
    # This is fixture setup, separately recorded, not a scored pickup benchmark.
    setup_steps=0
    if loaded:
        while not drone.grasped_bombs() and setup_steps<30:
            pg.step({drone:dict(forward=.12,lateral=0.,rotation=0.,grasper=1)})
            setup_steps+=1
    # Geometry verification is outside run_control; it is never given to policy.
    hull=np.array([tuple(v) for v in drone.base.pm_shapes[0].get_vertices()])
    actual_radius=float(np.linalg.norm(hull,axis=1).max())
    assert abs(actual_radius-BODY_RADIUS)<.001
    bomb_radius=max(np.linalg.norm(tuple(v)) for v in bomb.pm_shapes[0].get_vertices()) if bomb else None
    acquired=bool(drone.grasped_bombs());losses=0;was_held=acquired;trace=[];crossed=None
    try:
        for tick in range(steps):
            drone.elapsed_timestep=tick
            if scenario=='payload_turn' and 30<=tick<100:
                drone.gps()._disabled=True;drone.compass()._disabled=True
            cmd=run_control(drone);pg.step({drone:cmd})
            held=bool(drone.grasped_bombs());acquired|=held
            if was_held and not held and not (bomb and bomb.disposed):losses+=1
            was_held=held
            p=drone.true_position();v=drone.true_velocity()
            if crossed is None and p[0]>=-80:crossed=tick+1
            if tick%2==0:
                trace.append(dict(step=tick,x=float(p[0]),y=float(p[1]),speed=float(np.linalg.norm(v)),
                    command=cmd,health=int(drone.drone_health),held=held,
                    regulator=dict(drone.governor.last),
                    bomb=None if bomb is None or bomb.disposed else bomb.true_position().tolist()))
            if drone.drone_health<=0:break
        return dict(variant=variant,scenario=scenario,seed=seed,steps=tick+1,
            seconds=time.perf_counter()-started,contact_episodes=sum(contacts.values()),health_lost=25-drone.drone_health,
            acquired=acquired,retained=was_held,losses=losses,disposed=bool(bomb and bomb.disposed),
            first_160px_step=crossed,final=drone.true_position().tolist(),
            body_radius=actual_radius,bomb_radius=bomb_radius,payload_bound=PAYLOAD_RADIUS,
            fixture_ready=bool(not loaded or acquired),setup_steps=setup_steps,
            scope='motion component only, not whole policy or a score',trace=trace)
    finally:pg.close_window()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--red-idle-confirmed',action='store_true');p.add_argument('--case-json');p.add_argument('--wait-go',action='store_true')
    p.add_argument('--scenarios',nargs='+',choices=['empty_open','empty_wall','payload_open','payload_turn'],
                   default=['empty_open','empty_wall','payload_open','payload_turn'])
    a=p.parse_args()
    if a.case_json:
        if a.wait_go and sys.stdin.readline().strip()!='GO':raise RuntimeError('Missing resource handshake')
        with a.output.open('x',encoding='utf-8') as f:json.dump(run_case(**json.loads(a.case_json)),f,indent=2)
        return
    if not a.red_idle_confirmed:p.error('Confirm no competing red simulation before native tests')
    a.output.mkdir(parents=True,exist_ok=False)
    from .optimization.evaluation.resources import run_limited
    rows=[]
    for scenario in a.scenarios:
        for variant in ('b0_motion','b8'):
            config=dict(variant=variant,scenario=scenario);path=a.output/f'{scenario}_{variant}.json'
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
            with path.with_suffix('.log').open('x',encoding='utf-8') as log:
                resource=run_limited([sys.executable,'-B','-m',__spec__.name,'--output',str(path),
                    '--case-json',json.dumps(config),'--wait-go'],env,log,timeout=40)
            log=path.with_suffix('.log').read_text(encoding='utf-8',errors='replace')
            valid=resource.get('exit_code')==0 and path.exists() and not any(x in log for x in ('Traceback','Exception ignored from cffi callback','AssertionError'))
            row=json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(config,error=log)
            row.update(valid=valid,resources=resource)
            row['scenario_valid']=bool(valid and row.get('fixture_ready',False))
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ('trace','resources')},ensure_ascii=False),flush=True)
    with (a.output/'results.json').open('x',encoding='utf-8') as f:json.dump(rows,f,indent=2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(a.scenarios),figsize=(4*len(a.scenarios),4),squeeze=False)
    for ax,scenario in zip(axes[0],a.scenarios):
        for row in rows:
            if row['scenario']!=scenario:continue
            trace=row.get('trace',[]);ax.plot([r['x'] for r in trace],[r['y'] for r in trace],label=row['variant'])
        if scenario=='empty_wall':ax.axvline(2,color='black')
        ax.set(title=scenario,xlim=(-250,300),ylim=(-80,230));ax.legend();ax.set_aspect('equal')
    fig.suptitle('Native motion components only: not full B0/B8 policy results')
    fig.tight_layout();fig.savefig(a.output/'native_paths.png',dpi=120);plt.close(fig)


if __name__=='__main__':main()
