from types import SimpleNamespace as NS
import numpy as np
from swarm_rescue.solutions.devtools.blue_lifecycle import LifecycleObserver,instrument


def test_passive_probe_preserves_final_command():
    class Base:
        def __init__(self):
            self._center_hit=None;self._route=[];self._target=None;self._position=np.zeros(2)
            self._mode='search';self._delivery_ticks=0;self._peers={}
            self.elapsed_timestep=7;self.identifier=0;self.drone_health=25
        def grasped_bombs(self):return []
        def measured_gps_position(self):return np.zeros(2)
        def measured_compass_angle(self):return 0.
        def control(self):return dict(forward=.2,lateral=-.1,rotation=.3,grasper=1)
    ref=Base();probed=instrument(Base)()
    assert ref.control()==probed.control()
    assert probed._probe_frame['source']=='early_return'


def test_observer_requires_public_visibility_and_confirms_only_official_disposal():
    bomb=NS(uid=41,position=(20.,0.),disposed=False)
    sensor=NS(ray_angles=[0.],_hitpoints=np.array([[0,0,0,0,0,0,0,0,41,20.]]))
    frame=dict(step=1,mode='collect',target=[20.,0.],position=[0.,0.],command=dict(grasper=1),
               center_distance=None,carrying=False,source='navigation',terminal_state=None)
    d=NS(identifier=0,drone_health=25,grasped_bombs=lambda:[],semantic_values=lambda:[],semantic=lambda:sensor,
         _probe_frame=frame,true_position=lambda:np.zeros(2))
    gui=NS(_the_map=NS(bombs=[bomb]),_drones=[d],_elapsed_timestep=1)
    obs=LifecycleObserver();obs.begin(gui);obs.end(gui)
    assert obs.rows[0]['first_seen'] is None  # renderer alone is not a legal discovery
    d.semantic_values=lambda:[NS(angle=0.,entity_type=NS(name='BOMB'))]
    obs.begin(gui);d.grasped_bombs=lambda:[bomb];obs.end(gui)
    assert obs.rows[0]['first_seen']==1 and len(obs.rows[0]['pickups'])==1
    obs.begin(gui);d.grasped_bombs=lambda:[];obs.end(gui)
    assert len(obs.rows[0]['losses'])==1 and obs.rows[0]['disposed'] is None
    bomb.disposed=True;obs.begin(gui);obs.end(gui)
    assert obs.rows[0]['disposed']==1
    phase=dict(obs.rows[0]['phase_steps']);gui._elapsed_timestep=2
    obs.begin(gui);obs.end(gui)
    assert obs.rows[0]['phase_steps']==phase
