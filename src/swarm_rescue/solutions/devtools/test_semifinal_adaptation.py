"""Current-engine regressions for the two retained semifinal controllers."""
import importlib,math
import numpy as np
import pytest
from swarm_rescue.simulation.utils.constants import MAX_RANGE_LIDAR_SENSOR

@pytest.fixture(params=['advanced2','advanced5_narrow'])
def strategy(request):
    return importlib.import_module('swarm_rescue.solutions.my_drone_red_'+request.param)


def test_range_returns_do_not_create_walls(strategy):
    a=np.linspace(-math.pi,math.pi,181)
    rng=np.random.default_rng(742)
    fine=strategy._M03_ObservationMapV2((900,900))
    coarse=strategy._M09_ExplorationGrid((900,900))
    for _ in range(100):
        d=MAX_RANGE_LIDAR_SENSOR+rng.normal(0,2.5,len(a))
        fine.observe(np.zeros(2),0,d,a)
        coarse.observe(np.zeros(2),0,d,a)
    assert not fine.mask().any()
    assert not (coarse.obstacle>=2).any()
    assert fine.scans==100 and coarse.free.any()


def test_real_wall_is_retained(strategy):
    a=np.linspace(-.6,.6,61);d=100/np.cos(a)
    m=strategy._M03_ObservationMapV2((900,900))
    for _ in range(4):m.observe(np.zeros(2),0,d,a)
    assert m.mask().sum()>15


def test_invalid_and_range_overflow_never_explore_beyond_sensor(strategy):
    m=strategy._M09_ExplorationGrid((900,900));a=np.linspace(-math.pi,math.pi,181)
    d=np.full(181,10000.);d[:3]=[np.nan,np.inf,-1]
    m.observe(np.zeros(2),0,d,a)
    yy,xx=np.where(m.free)
    # Coarse cells cover a small area around each sampled point.
    assert max(np.linalg.norm(m.point((int(y),int(x)))) for y,x in zip(yy,xx))<MAX_RANGE_LIDAR_SENSOR+20
    assert not m.obstacle.any()


def test_pose_loss_dead_reckoning_and_recovery(strategy):
    f=strategy._M01_PoseFilter()
    assert f.update(None,None,[1,0,0]) is None
    assert f.update([np.nan,0],np.nan,None) is None
    f.update([0,0],0,None)
    for _ in range(40):f.update(None,None,[1,0,0])
    assert np.allclose(f.position,[40,0]) and f.absolute_missing_steps==40
    for _ in range(80):f.update([45,2],.1,[0,0,0])
    assert np.linalg.norm(f.position-[45,2])<.01 and abs(f.heading-.1)<.001
    assert f.absolute_missing_steps==0
    old=f.position.copy();f.update(None,None,[np.nan,0,0])
    assert np.array_equal(f.position,old)


def test_semantic_noise_is_zero_mean_current_engine():
    from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
    from types import SimpleNamespace
    # Exercise the actual noise method without creating an engine window.
    sensor=SimpleNamespace(_values=[DroneSemanticSensor.Data(50,0,None,False)]*10000,
        _std_dev_noise=2.5,Data=DroneSemanticSensor.Data)
    state=np.random.get_state()
    try:
        np.random.seed(742);DroneSemanticSensor._apply_noise(sensor)
    finally:np.random.set_state(state)
    errors=np.array([x.distance-50 for x in sensor._values])
    assert abs(errors.mean())<.1 and 2.4<errors.std()<2.6


def test_narrow_no_hit_and_invalid_rays():
    from swarm_rescue.solutions import my_drone_red_advanced5_narrow as s
    a=np.linspace(-math.pi,math.pi,181);d=np.full(181,230.)
    d[20:23]=[np.nan,np.inf,-2]
    m=s.NarrowLocalMap((900,900))
    for _ in range(5):m.observe(np.zeros(2),0,d,a,np.zeros(181,bool))
    assert not m.hits.any() and not m.squeeze.any()


@pytest.mark.parametrize('map_name',['Map08','Map09'])
@pytest.mark.parametrize('selector',['advanced2','advanced5_narrow'])
def test_native_control_short_run(map_name,selector):
    from swarm_rescue.solutions.devtools.evaluate_red import configure_ray_backend
    from swarm_rescue.solutions.devtools.evaluate_verified_revision import load_controller,controller_identity
    from swarm_rescue.simulation.elements.sensor_disablers import ZoneType
    from swarm_rescue import launcher
    configure_ray_backend('cpu')
    np.random.seed(742)
    cls=load_controller(controller_identity(selector))
    world=getattr(launcher,map_name)(cls,(ZoneType.NO_COM_ZONE,ZoneType.NO_GPS_ZONE),5)
    try:
        for drone in world.drones:drone.enable_bomb_deployer(world,2)
        world.playground.step()
        for step in range(1,61):
            commands={}
            for drone in world.drones:
                drone.elapsed_timestep=step;drone.elapsed_walltime=step/30
                # Simulate sensor loss at the public sensor API, then restoration.
                gps,compass=drone.measured_gps_position,drone.measured_compass_angle
                if 20<=step<40:
                    drone.measured_gps_position=lambda:None;drone.measured_compass_angle=lambda:None
                try:commands[drone]=drone.control()
                finally:drone.measured_gps_position=gps;drone.measured_compass_angle=compass
                assert all(math.isfinite(float(v)) for v in commands[drone].values())
            world.playground.step(all_commands=commands)
        assert all(d._steps==60 for d in world.drones)
    finally:world.playground.window.close()
