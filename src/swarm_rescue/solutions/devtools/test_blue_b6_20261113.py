import copy
import math
from types import SimpleNamespace as NS
import numpy as np
import pytest
from swarm_rescue.solutions.my_drone_blue_b6_local_20261113 import (
    PrincipalMatcher,BlueLocalWindow,MyDroneBlueB6Local,MyDroneBlueB6Off,MyDroneBlueB6Private,PortalObservation)
from swarm_rescue.solutions.my_drone_eval_blue import RescueGrid,MyDroneBlueAdapted
from swarm_rescue.solutions.semifinal_localization import scan_points
from swarm_rescue.solutions.devtools.blue_b6_fixtures_20261113 import cast,ANGLES,polygon


def test_corridor_weak_axis_is_not_falsely_corrected():
    walls=[((-600,-40),(600,-40)),((-600,40),(600,40))]
    scan=scan_points(cast(walls,np.zeros(2),0.),ANGLES)
    matcher=PrincipalMatcher((768,768));matcher.add(scan,np.zeros(2),0.)
    delta,info=matcher.match(scan,np.array([5.,4.]),.03)
    assert delta is not None
    assert info['rank']==1 and abs(delta[0])<1.
    assert delta[1]<-1 and delta[2]<0


def test_diagonal_corridor_tests_principal_direction_not_separate_axes():
    theta=math.pi/4;c,s=math.cos(theta),math.sin(theta);rot=np.array([[c,-s],[s,c]])
    walls=[(rot@np.array([-600,y]),rot@np.array([600,y])) for y in (-40,40)]
    scan=scan_points(cast(walls,np.zeros(2),theta),ANGLES)
    matcher=PrincipalMatcher((768,768));matcher.add(scan,np.zeros(2),theta)
    delta,info=matcher.match(scan,rot@np.array([5.,4.]),theta+.03)
    assert delta is not None and info['rank']==1
    projection=np.array(info['projection']);tangent=rot@np.array([1.,0.])
    assert np.linalg.norm(projection@tangent)<.25


def test_matches_prior_before_inserting_current_scan_and_bounds_history():
    window=BlueLocalWindow();events=[]
    window.matcher.match=lambda *a:(events.append('match') or (None,dict(rank=0,reason='test')))
    window.matcher.add=lambda *a:events.append('add')
    window.update(np.zeros(2),0.,np.zeros((20,2)),None,None,0)
    assert events==['match','add']
    window.origin=np.zeros(2);window.started=0
    window.update(np.array([101.,0.]),0.,np.zeros((20,2)),None,None,6)
    assert window.started==6 and window.matcher.frames.maxlen==8


def test_insufficient_scan_cannot_claim_global_confidence():
    window=BlueLocalWindow()
    for tick in range(30):window.update(np.zeros(2),0.,np.zeros((0,2)),None,None,tick)
    assert not window.ready and window.stats['accepted']==0


def make_drone(cls):
    from swarm_rescue.simulation.utils.misc_data import MiscData
    d=cls(identifier=0,misc_data=MiscData(size_area=(320,256),number_drones=1))
    d._inputs=NS(gps=np.zeros(2),compass=0.,odo=np.zeros(3),held=[])
    d.measured_gps_position=lambda:d._inputs.gps
    d.measured_compass_angle=lambda:d._inputs.compass
    d.odometer_values=lambda:d._inputs.odo
    d.lidar_values=lambda:np.full(181,230.)
    d.lidar_rays_angles=lambda:ANGLES
    d.semantic_values=lambda:[]
    d.grasped_bombs=lambda:d._inputs.held
    return d


def test_off_switch_returns_exact_B0_final_controls():
    b0=make_drone(MyDroneBlueAdapted);off=make_drone(MyDroneBlueB6Off)
    for tick in range(1,70):
        for d in (b0,off):d.elapsed_timestep=tick;d._inputs.gps=None if 20<tick<50 else np.zeros(2);d._inputs.compass=None if 20<tick<50 else 0.
        assert b0.control()==off.control()
        np.testing.assert_array_equal(b0._grid.free,off._grid.free)
        np.testing.assert_array_equal(b0._grid.evidence,off._grid.evidence)


def test_private_map_does_not_pollute_old_map_or_stop_relative_navigation():
    d=make_drone(MyDroneBlueB6Private);d.elapsed_timestep=1;d.control()
    trusted=d._grid;before=trusted.evidence.copy()
    d._inputs.gps=None;d._inputs.compass=None;d.elapsed_timestep=2
    command=d.control();assert d._grid is not trusted
    assert any(abs(command[k])>0 for k in ('forward','lateral','rotation'))
    d._grid.evidence[4,4]=8
    np.testing.assert_array_equal(trusted.evidence,before)
    assert d.define_message_for_all() is None
    d._inputs.gps=np.zeros(2);d._inputs.compass=0.
    for tick in range(3,12):d.elapsed_timestep=tick;d.control()
    assert d._grid is trusted and d._b6_book is None
    assert d._grid.evidence[4,4]==before[4,4]


def test_relock_does_not_clear_actual_possession_or_release_payload():
    d=make_drone(MyDroneBlueB6Private);d.elapsed_timestep=1;d.control()
    d._inputs.held=[object()];d._inputs.gps=None;d._inputs.compass=None
    d.elapsed_timestep=2;assert d.control()['grasper']==1
    d._inputs.gps=np.zeros(2);d._inputs.compass=0.
    for tick in range(3,15):d.elapsed_timestep=tick;assert d.control()['grasper']==1
    assert d._was_carrying
    d._inputs.held=[];d.elapsed_timestep=15;d.control()
    assert not d._was_carrying


def test_portal_never_admits_payload_or_marks_unknown_as_free():
    grid=RescueGrid((320,256));before=grid.free.copy();portal=PortalObservation()
    walls=[((40,-17.5),(180,-17.5)),((40,17.5),(180,17.5))]
    d=cast(walls,np.zeros(2),0.)
    assert portal.propose(grid,np.zeros(2),0.,d,ANGLES,carrying=True) is None
    assert portal.reason=='payload_not_validated'
    for _ in range(4):assert portal.propose(grid,np.zeros(2),0.,d,ANGLES) is None
    np.testing.assert_array_equal(grid.free,before)


def test_probe_packing_keeps_raw_evidence_and_invalid_scan_values():
    from swarm_rescue.solutions.devtools.blue_b6_diagnostics_20261113 import pack,unpack
    array=np.array([[0,.25,2,8],[np.nan,np.inf,-np.inf,.5]],np.float32)
    np.testing.assert_array_equal(unpack(pack(array)),array)


def test_new_policy_has_no_truth_file_or_map_specific_online_reads():
    from pathlib import Path
    import swarm_rescue.solutions.my_drone_blue_b6_local_20261113 as policy
    source=Path(policy.__file__).read_text(encoding='utf-8')
    for forbidden in ('true_position','true_angle','Map08','Map09','read_text','np.load','requests.','http','1560','936'):
        assert forbidden not in source


def test_supported_portal_only_uses_observed_safe_viewpoint():
    grid=RescueGrid((512,256));grid.free[:,:19]=True
    before=grid.free.copy();evidence=grid.evidence.copy();portal=PortalObservation()
    d=cast([((48,-20),(180,-20)),((48,20),(180,20))],np.zeros(2),0.)
    assert portal.propose(grid,np.zeros(2),0.,d,ANGLES) is None
    assert portal.propose(grid,np.zeros(2),0.,d,ANGLES) is None
    point=portal.propose(grid,np.zeros(2),0.,d,ANGLES)
    assert point is not None and 0<point[0]<28 and grid.layers()[0][grid.cell(point)]
    np.testing.assert_array_equal(grid.free,before)
    np.testing.assert_array_equal(grid.evidence,evidence)


def test_portal_goal_precedes_planning_and_expires_without_loop(monkeypatch):
    from swarm_rescue.solutions.my_drone_blue_b6_local_20261113 import MyDroneBlueB6Portal
    from swarm_rescue.solutions.my_drone_blue_narrow import MyDroneBlueNarrow
    d=make_drone(MyDroneBlueB6Portal);d._position=np.zeros(2);d._mode='search';d._tick=7
    d._grid.free[:]=True;d._b6_portal_point=np.array([24.,8.]);d._b6_portal_until=90
    goals=[]
    def navigation(self,goal,*args):
        goals.append(goal);return dict(forward=.1,lateral=0.,rotation=0.,grasper=0)
    monkeypatch.setattr(MyDroneBlueNarrow,'_navigate',navigation)
    d._navigate(None,d.lidar_values(),ANGLES)
    assert len(goals)==1;np.testing.assert_equal(goals[0],[24.,8.])
    d._tick=91;d._navigate(None,d.lidar_values(),ANGLES)
    assert goals[-1] is None and d._b6_portal.reason=='observation_timeout'
    assert d._b6_portal_cooldown==181


def test_portal_closes_on_new_obstruction_and_payload(monkeypatch):
    from swarm_rescue.solutions.my_drone_blue_b6_local_20261113 import MyDroneBlueB6Portal
    from swarm_rescue.solutions.my_drone_blue_narrow import MyDroneBlueNarrow
    d=make_drone(MyDroneBlueB6Portal);d._position=np.zeros(2);d._tick=7;d._mode='search'
    d._b6_portal_point=np.array([24.,8.]);d._b6_portal_until=90
    monkeypatch.setattr(MyDroneBlueNarrow,'_navigate',lambda *a,**kw:{})
    d._navigate(None,d.lidar_values(),ANGLES)
    assert d._b6_portal_point is None and d._b6_portal.reason=='approach_blocked'
    d._b6_portal_point=np.array([24.,8.])
    d._navigate(None,d.lidar_values(),ANGLES,True)
    assert d._b6_portal_point is None and not d.PAYLOAD_NARROW


def test_extended_probe_preserves_controls_and_captures_raw_snapshot():
    from swarm_rescue.solutions.devtools.blue_b6_diagnostics_20261113 import instrument,unpack
    b0=make_drone(MyDroneBlueAdapted);probe=make_drone(instrument(MyDroneBlueAdapted))
    for tick in range(1,122):
        b0.elapsed_timestep=probe.elapsed_timestep=tick
        assert b0.control()==probe.control()
    sample=probe._b6_observations[0]
    assert sample['step']==120 and len(sample['lidar'])==181
    assert sample['component_size']>0 and sample['source']
    assert unpack(sample['own_evidence']).dtype==np.float32
    b0.collide_wall();probe.collide_wall()
    assert probe.drone_health==b0.drone_health
    assert probe._b6_collisions['wall_pre_solve_callbacks']==1


def test_diagnostics_keep_removed_drone_snapshots():
    from swarm_rescue.solutions.devtools.blue_b6_diagnostics_20261113 import Observer
    o=Observer();d=NS(identifier=7,_b6_observations=[{'step':42}],_b6_collisions={},_b6_diagnostic_seconds=.01)
    o._b6_drones={'7':d}
    data=o.export(NS(_drones=[]))
    assert data['b6_diagnostics']['drones']['7']['snapshots']==[{'step':42}]


def test_independent_adapter_preserves_registry_file_and_excludes_invalid_scores(monkeypatch):
    from pathlib import Path
    from swarm_rescue.solutions.devtools.optimization.evaluation import identities
    from swarm_rescue.solutions.devtools.blue_b6_run_20261113 import register_local,valid_result,worker_command
    content=Path(identities.__file__).read_bytes();original=identities.BLUE_CONTROLLERS
    monkeypatch.setattr(identities,'BLUE_CONTROLLERS',original)
    register_local()
    assert identities.load_blue('b6_local') is MyDroneBlueB6Local
    assert identities.load_blue('adapted') is MyDroneBlueAdapted
    assert Path(identities.__file__).read_bytes()==content
    valid=dict(status='ok',exit_code=0,blue_score=0.,walltime_limited=False)
    assert valid_result(valid)  # Zero-disposal/zero-score complete games remain.
    for field in ('walltime_limited','controller_crashed','engine_callback_errors','smoke'):
        assert not valid_result(dict(valid,**{field:True}))
    assert not valid_result(dict(valid,exit_code=1))
    assert '-B' in worker_command(Path('request.json'))


def test_absolute_recovery_discards_local_scan_history_before_reusing_reference():
    window=BlueLocalWindow();scan=scan_points(cast([((-600,-40),(600,-40)),((-600,40),(600,40))],np.zeros(2),0.),ANGLES)
    window.update(np.zeros(2),0.,scan,np.zeros(2),0.,0)
    assert window.matcher.distance is not None
    window.update(np.zeros(2),0.,scan,None,None,1)
    window.update(np.array([30.,0.]),0.,scan,np.zeros(2),0.,2)
    assert window.matcher.distance is None and not window.ready
    assert window.stats['absolute_recovery_reset']==1
    for tick in range(3,13):window.update(np.zeros(2),0.,scan,np.zeros(2),0.,tick)
    assert window.ready and window.matcher.distance is not None


def test_pose_correction_is_same_for_empty_and_payload_given_same_sensors():
    empty=make_drone(MyDroneBlueB6Local);loaded=make_drone(MyDroneBlueB6Local)
    scan=cast([((-600,-40),(600,-40)),((-600,40),(600,40))],np.zeros(2),0.)
    for drone in (empty,loaded):drone.lidar_values=lambda:scan
    for tick in range(1,30):
        for drone in (empty,loaded):
            drone.elapsed_timestep=tick
            drone._inputs.gps=None if tick>8 else np.zeros(2)
            drone._inputs.compass=None if tick>8 else 0.
        loaded._inputs.held=[object()]
        empty.control();loaded_command=loaded.control()
        np.testing.assert_allclose(empty._position,loaded._position)
        assert loaded_command['grasper']==1 and empty._heading==loaded._heading
