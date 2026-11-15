"""Passive B8 fields on the existing B6 / blue_lifecycle observer."""
from collections import Counter
import copy
import time
import numpy as np
from .blue_b6_diagnostics_20261113 import instrument as previous_instrument, Observer as PreviousObserver


def instrument(base):
    parent=previous_instrument(base)
    class Probe(parent):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            self._b8_probe_rows=[];self._b8_probe_seconds=0.;self._b8_phase_steps=Counter()
        def control(self):
            cmd=super().control();started=time.perf_counter()
            phase=getattr(self,'_mode','unknown');self._b8_phase_steps[phase]+=1
            info=copy.deepcopy(getattr(self,'_b8_info',{}))
            self._probe_frame['b8']=info
            if int(self.elapsed_timestep)%12==0:
                odo=self.odometer_values()
                self._b8_probe_rows.append(dict(step=int(self.elapsed_timestep),mode=phase,
                    health=float(self.drone_health),carrying=bool(self.grasped_bombs()),
                    odometry=None if odo is None else np.asarray(odo).tolist(),command=dict(cmd),navigation=info))
                self._b8_probe_rows=self._b8_probe_rows[-460:]
            self._b8_probe_seconds+=time.perf_counter()-started
            return cmd
    Probe.__name__=base.__name__;Probe.__module__=base.__module__
    return Probe


class Observer(PreviousObserver):
    def export(self,gui):
        data=super().export(gui)
        data['b8_navigation']=dict(scope='legal-input probe, events do not confirm disposal',
            drones={key:dict(events=list(getattr(d,'_b8_events',[])),stats=dict(getattr(d,'_b8_stats',{})),
                samples=getattr(d,'_b8_probe_rows',[]),phase_steps=dict(getattr(d,'_b8_phase_steps',{})),
                control_seconds=getattr(d,'_b8_seconds',None),probe_seconds=getattr(d,'_b8_probe_seconds',0.),
                health=float(d.drone_health)) for key,d in self._b6_drones.items()})
        return data
