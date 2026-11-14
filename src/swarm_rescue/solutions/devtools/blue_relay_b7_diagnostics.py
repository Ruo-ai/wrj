"""Blue-only passive extension of B6/blue_lifecycle, identical for all groups."""
import time
import base64
import numpy as np
from .blue_b6_diagnostics_20261113 import instrument as previous_instrument,Observer as PreviousObserver
from swarm_rescue.solutions.my_drone_blue_relay_b7 import RelayView


def instrument(base):
    parent=previous_instrument(base)
    class Probe(parent):
        def __init__(self,**kwargs):
            super().__init__(**kwargs);self._b7_probe_view=RelayView(self._grid)
            self._b7_good=0;self._b7_seconds=0.;self._b7_packet_keys=set();self._b7_coverage=None

        def define_message_for_all(self):
            message=super().define_message_for_all()
            relay=getattr(self,'_relay',None);meta=getattr(message,'relay',{})
            for name in ('offer','status'):
                job=meta.get(name)
                if relay is not None and job:
                    key=(tuple(job['token']),job['phase'],name)
                    if key not in self._b7_packet_keys:
                        self._b7_packet_keys.add(key)
                        relay.event(name+'_packet_built',self._tick,job,'construction_is_not_delivery')
            return message

        def control(self):
            command=super().control();started=time.perf_counter()
            tick=int(self.elapsed_timestep);position=getattr(self,'_position',None)
            gps=self.measured_gps_position();heading=self.measured_compass_angle()
            good=(position is not None and gps is not None and heading is not None
                  and np.isfinite(gps).all() and np.isfinite(heading) and np.linalg.norm(gps-position)<15)
            self._b7_good=self._b7_good+1 if good else 0
            if self._b7_good>=8 and tick%12==0 and not self.semantic_is_disabled():
                self._b7_probe_view.observe(position,self._heading,self.lidar_values(),self.lidar_rays_angles())
                view=self._b7_probe_view
                self._b7_coverage=dict(shape=view.shape,packed=base64.b64encode(np.packbits(view.seen)).decode('ascii'),
                    observed_cell_samples=view.observed_samples,repeated_cell_samples=view.repeated_samples)
            lease=getattr(self,'_relay',None)
            self._probe_frame['b7']=dict(out=dict(lease.out) if lease and lease.out else None,
                incoming=dict(lease.inc) if lease and lease.inc else None,
                assessment=getattr(self,'_relay_assessment',{}),reasons=dict(getattr(self,'_relay_reasons',{})))
            self._b7_seconds+=time.perf_counter()-started
            return command
    Probe.__name__=base.__name__;Probe.__module__=base.__module__
    return Probe


class Observer(PreviousObserver):
    def export(self,gui):
        data=super().export(gui)
        data['b7_relay']=dict(scope='OFFLINE diagnosis; no truth supplied to task policy',
            coverage_scope='estimated-grid semantic opportunity every 12 steps with reliable GPS; not exact search coverage',
            drones={str(d.identifier):dict(events=list(getattr(getattr(d,'_relay',None),'events',[])),
                stats=dict(getattr(getattr(d,'_relay',None),'stats',{})),
                reasons=dict(getattr(d,'_relay_reasons',{})),coverage=getattr(d,'_b7_coverage',None),
                diagnostic_seconds=getattr(d,'_b7_seconds',0.),health=float(d.drone_health)) for d in gui._drones})
        return data
