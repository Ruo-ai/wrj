"""Red-owned native physics probes; truth is diagnostic only, never policy input."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np

MODULE = 'swarm_rescue.solutions.devtools.red_stopturn_native'


def motion_case(group='B', noise=True, seed=742, steps=420, scene='corner'):
    """Native component fixture, NOT a full A3 exploration/deployment game.

    A uses the existing pose filter + local safe-velocity primitive. All groups
    use exactly the same A3 mapper. Only this offline fixture knows its route.
    Truth is sampled outside run_control and cannot influence the commands.
    """
    from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
    from swarm_rescue.simulation.drone.ground_truth_guard import run_control
    from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
    from swarm_rescue.simulation.utils.misc_data import MiscData
    from swarm_rescue.simulation.elements.normal_wall import NormalWall
    from swarm_rescue.simulation.elements.sensor_disablers import NoGpsZone
    from swarm_rescue.simulation.utils.definitions import CollisionTypes
    from swarm_rescue.simulation.reporting.place_score_manager import walls_and_boundary_grid
    from swarm_rescue.tools.map_exploration_scoring import MapExplorationScorer, breakdown_to_dict
    from swarm_rescue.solutions.my_drone_red_advanced2 import _M01_PoseFilter, _M11_safe_velocity
    from swarm_rescue.solutions.my_drone_red_mapping import SubmissionMapper
    from swarm_rescue.solutions.my_drone_red_stopturn import SegmentMotion, body_motion, wrap
    random.seed(seed); np.random.seed(seed)
    size = (800,600)
    walls = [((-330,-160),(40,-160)),((-330,-40),(-80,-40)),
             ((40,-160),(40,70)),((-80,-40),(-80,190)),
             ((-80,190),(280,190)),((40,70),(280,70))]
    goals = [(-20,-100),(-20,130),(210,130)]
    start = (-290,-100)
    if scene == 'deadend':
        walls = [((-330,-160),(40,-160)),((-330,-40),(40,-40)),((40,-160),(40,-40))]
        goals = [(0,-100),(-240,-100)]
    elif scene == 'narrow':
        walls = [((-330,-120.5),(280,-120.5)),((-330,-79.5),(280,-79.5))]
        goals = [(200,-100)]  # 35px free width, no artificial guaranteed passage
    class Probe(DroneAbstract):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.nav = _M01_PoseFilter(); self.mapper = SubmissionMapper(size)
            self.motion = SegmentMotion(size,radar=group=='C'); self.motion.record_events = True
            self.goal_index = 0; self.motion_velocity = np.zeros(2); self.previous = None
        def define_message_for_all(self): return None
        def control(self):
            gps, compass, odo = self.measured_gps_position(),self.measured_compass_angle(),self.odometer_values()
            d,a,objects = self.lidar_values(),self.lidar_rays_angles(),self.semantic_values()
            self.nav.update(gps,compass,odo); self.mapper.update(gps,compass,odo,d,a,objects)
            self.motion.pose.update(gps,compass,odo,d,a,objects)
            measured = body_motion(odo)
            if measured is not None: self.motion_velocity = .4*self.motion_velocity+.6*measured
            if self.nav.position is None: return dict(forward=0,lateral=0,rotation=0,grasper=0)
            delta = np.array(goals[min(self.goal_index,len(goals)-1)])-self.nav.position
            if np.linalg.norm(delta)<22 and self.goal_index<len(goals):
                self.goal_index += 1
                delta = np.array(goals[min(self.goal_index,len(goals)-1)])-self.nav.position
            if self.goal_index == len(goals):
                v = np.clip(-.55*self.motion_velocity,-1,1)
                self.motion.step(d,a,0,allow=False)
                return dict(forward=float(v[0]),lateral=float(v[1]),rotation=0,grasper=0)
            angle = wrap(math.atan2(delta[1],delta[0])-self.nav.heading)
            force, chosen = _M11_safe_velocity(d,a,angle,self.motion_velocity,self.previous,float(np.linalg.norm(delta)))
            self.previous = chosen
            cmd = dict(forward=float(force[0]),lateral=float(force[1]),rotation=0.,grasper=0)
            if group != 'A':
                replacement = self.motion.step(d,a,chosen,float(np.linalg.norm(delta)),objects)
                if replacement is not None: cmd.update(replacement)
            return cmd
    pg = ClosedPlayground(size); started = time.perf_counter()
    for begin,end in walls:
        wall = NormalWall(begin,end,wall_thickness=6)
        pg.add(wall,wall.wall_coordinates)
    pg.add(NoGpsZone(size=(300,460)),((-10,0),0.))
    drone = Probe(identifier=0,misc_data=MiscData(size_area=size,number_drones=1))
    pg.add(drone,(start,0.)); drone.set_competition_mode(True)
    if not noise:
        for sensor in drone.sensors:
            if hasattr(sensor,'_noise'): sensor._noise=False
    contacts = [0]
    handler = pg.space.add_collision_handler(CollisionTypes.DRONE,CollisionTypes.WALL)
    original_begin = handler.begin
    def count_contact(arbiter,space,data):
        contacts[0] += 1
        return original_begin(arbiter,space,data) if original_begin else True
    handler.begin = count_contact
    pg.step(); rows=[]; observation=np.zeros((600,800),np.uint8)
    try:
        truth,boundary = walls_and_boundary_grid(pg)
        for tick in range(steps):
            drone.elapsed_timestep = tick
            # Capture aligned truth BEFORE control/physics; never pass it in.
            true_position=drone.true_position().copy(); true_heading=float(drone.true_angle())
            command=run_control(drone)
            pose=drone.motion.pose; mapper=drone.mapper
            hitpoints=drone.lidar()._hitpoints
            if hitpoints is not None:
                # Actual engine ray endpoints, not expanded trajectory coverage.
                hit=np.asarray(hitpoints); valid=hit[:,8]!=0
                xy=np.rint(np.column_stack((hit[valid,2]+400,300-hit[valid,3]))).astype(int)
                xy=xy[(xy[:,0]>=0)&(xy[:,0]<800)&(xy[:,1]>=0)&(xy[:,1]<600)]
                observation[xy[:,1],xy[:,0]]=1
            row=dict(step=tick,true_position=true_position.tolist(),true_heading=true_heading,
                position=pose.position.tolist(),heading=pose.heading,nav_position=drone.nav.position.tolist(),
                map_position=mapper.position.tolist(),absolute=list(pose.absolute),state=drone.motion.state,
                command=command,odometry=drone.odometer_values().tolist(),health=float(drone.drone_health),
                goal_index=drone.goal_index,position_error=float(np.linalg.norm(pose.position-true_position)),
                nav_error=float(np.linalg.norm(drone.nav.position-true_position)),
                map_error=float(np.linalg.norm(mapper.position-true_position)),
                heading_error=abs(wrap(pose.heading-true_heading)),
                clearance=drone.motion.last_clearance,braking_distance=drone.motion.last_brake_distance)
            rows.append(row)
            pg.step({drone:command})
            if drone.goal_index==len(goals): break
        prediction=mapper.to_pixels(mapper.mask())
        # This is the official map FORMULA on a synthetic fixture, not an official round score.
        scoring=breakdown_to_dict(MapExplorationScorer().score(truth,prediction,len(rows)/30))
        missing=[r for r in rows if not all(r['absolute'])]
        mean=lambda k: float(np.mean([r[k] for r in missing])) if missing else None
        return dict(scene=scene,group=group,noise=noise,seed=seed,steps=len(rows),
            seconds=time.perf_counter()-started,complete=drone.goal_index==len(goals),goals_reached=drone.goal_index,
            collision_contacts=contacts[0],health=float(drone.drone_health),
            outage_steps=len(missing),outage_pose_mean=mean('position_error'),outage_nav_mean=mean('nav_error'),
            outage_mapper_mean=mean('map_error'),outage_heading_mean_rad=mean('heading_error'),
            distance=float(np.linalg.norm(np.diff([r['true_position'] for r in rows],axis=0),axis=1).sum()),
            stats=dict(drone.motion.stats),pose_stats=dict(pose.stats),events=list(drone.motion.events),
            map_formula=scoring,trace=rows,
            arrays=dict(truth=truth,boundary=boundary,prediction=prediction,observed_endpoints=observation*truth))
    finally: pg.close_window()


def calibration(mode, seed=742):
    from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
    from swarm_rescue.simulation.drone.ground_truth_guard import run_control
    from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
    from swarm_rescue.simulation.utils.misc_data import MiscData
    random.seed(seed); np.random.seed(seed)
    class Probe(DroneAbstract):
        def define_message_for_all(self): return None
        def control(self):
            odo = self.odometer_values()
            v = np.zeros(2)
            if odo is not None and np.isfinite(odo).all():
                v = odo[0]*np.array([math.cos(odo[1]-odo[2]), math.sin(odo[1]-odo[2])])
            move = np.array([1., 0.]) if self.elapsed_timestep < 40 else (
                np.clip(-.55*v, -1, 1) if mode == 'brake' else np.zeros(2))
            return dict(forward=float(move[0]), lateral=float(move[1]), rotation=0., grasper=0)
    pg = ClosedPlayground((900, 500)); started = time.perf_counter()
    drone = Probe(identifier=0, misc_data=MiscData(size_area=(900, 500), number_drones=1))
    pg.add(drone, ((-240., 0.), 0.)); drone.set_competition_mode(True)
    for sensor in drone.sensors:
        if hasattr(sensor, '_noise'): sensor._noise = False
    pg.step(); rows = []
    try:
        for tick in range(100):
            drone.elapsed_timestep = tick
            odo = drone.odometer_values().copy(); command = run_control(drone)
            before = drone.true_position().copy(); pg.step({drone: command})
            after = drone.true_position().copy()
            rows.append(dict(step=tick,position=after.tolist(),heading=float(drone.true_angle()),
                distance=float(np.linalg.norm(after-before)),odometry=odo.tolist(),command=command,
                actual_velocity=drone.true_velocity().tolist(),health=float(drone.drone_health)))
        stop = next((r['step'] for r in rows[40:] if abs(r['actual_velocity'][0]) < .45), None)
        anchor = np.array(rows[39]['position'])
        dist = float(np.linalg.norm(np.array(rows[stop]['position'])-anchor)) if stop else None
        return dict(scene='calibration',mode=mode,seed=seed,noise=False,seconds=time.perf_counter()-started,
            stop_step=stop,brake_distance_px=dist,speed_at_brake=rows[39]['actual_velocity'][0],trace=rows)
    finally: pg.close_window()


def command_for(params, path):
    argv = [MODULE, '--output', str(path), '--case-json', json.dumps(params), '--wait-go']
    bootstrap = (f'import runpy,sys;sys.path[:]={sys.path!r};sys.prefix={sys.prefix!r};'
                 f'sys.exec_prefix={sys.exec_prefix!r};sys.argv={argv!r};'
                 f'runpy.run_module({MODULE!r},run_name="__main__")')
    return [sys.executable, '-B', '-u', '-S', '-X', 'utf8', '-c', bootstrap]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--case-json'); p.add_argument('--wait-go', action='store_true')
    p.add_argument('--suite', choices=['calibration','motion'], default='calibration')
    args = p.parse_args()
    if args.case_json:
        if args.wait_go and sys.stdin.readline().strip() != 'GO': raise RuntimeError('Missing parent handshake')
        if os.name != 'nt':
            from swarm_rescue.solutions.devtools.optimization.evaluation.resources import linux_limit
            linux_limit(4_000_000_000)
        params=json.loads(args.case_json)
        row = motion_case(**params) if 'group' in params else calibration(**params)
        arrays=row.pop('arrays',None)
        if arrays is not None:
            np.savez_compressed(args.output.with_suffix('.npz'),**arrays)
            row['sampled_observed_wall_pixels']=int(arrays['observed_endpoints'].sum())
        with args.output.open('x', encoding='utf-8') as f: json.dump(row, f, indent=2, allow_nan=False)
        return
    from swarm_rescue.solutions.devtools.optimization.evaluation.resources import run_limited
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    cases = [(mode,dict(mode=mode)) for mode in ('coast','brake')]
    if args.suite=='motion':
        cases=[(f'corner_{group}_{"noisy" if noise else "ideal"}',dict(group=group,noise=noise))
               for noise,groups in [(False,'AB'),(True,'ABC')] for group in groups]
        cases += [(scene+'_B_noisy',dict(group='B',scene=scene)) for scene in ('deadend','narrow')]
    for name,params in cases:
        path = args.output/(name+'.json')
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        with path.with_suffix('.log').open('w', encoding='utf-8') as log:
            resources = run_limited(command_for(params,path),env,log,timeout=45)
        errors = path.with_suffix('.log').read_text(encoding='utf-8',errors='replace')
        valid = resources['exit_code']==0 and not any(s in errors for s in ('Traceback','Exception ignored from cffi callback'))
        row = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else dict(case=name,error=errors)
        row.update(valid=valid,resources=resources); rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('trace','events')}),flush=True)
    (args.output/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')


if __name__ == '__main__': main()
