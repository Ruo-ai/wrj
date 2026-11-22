"""Frontier34 R1: event-field compatibility fix, with unchanged flight policy.

The native event emitter reserves ``kind`` and writes its own ``step``. Upload
receipts and frontier diagnostics use those field names too; rename only these
diagnostic keys, without altering the receipt or control state.
The frozen first attempt remains available as invalid-run evidence.
"""
from swarm_rescue.solutions.my_drone_red_frontier34 import MyDroneRedFrontier34


class MyDroneRedFrontier34R1(MyDroneRedFrontier34):
    def _f34_event(self, reason, **data):
        if 'kind' in data:
            data['receipt_kind'] = data.pop('kind')
        if 'step' in data:
            data['control_step'] = data.pop('step')
        return super()._f34_event(reason, **data)

    def _event(self, kind, **data):
        if 'step' in data:
            data['control_step'] = data.pop('step')
        return super()._event(kind, **data)
