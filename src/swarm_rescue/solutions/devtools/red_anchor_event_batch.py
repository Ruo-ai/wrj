"""Red-only evaluator wrapper fixing optional A5 event activation.

The retained evaluator set the mapper flag inside its initializer, before the
controller overwrote it with its class default. Set the controller flag in this
offline process instead. Mapping decisions and existing results are untouched.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import sys

MODULE = 'swarm_rescue.solutions.devtools.red_anchor_event_batch'


@contextmanager
def anchor_event_recording(enabled):
    from swarm_rescue.solutions.my_drone_red_mapping_anchor import (
        MyDroneRedMappingAnchor, MyDroneRedMappingAnchorOff,
    )
    classes = (MyDroneRedMappingAnchor, MyDroneRedMappingAnchorOff)
    previous = [c.RECORD_ANCHORS for c in classes]
    try:
        for c in classes:
            c.RECORD_ANCHORS = bool(enabled)
        yield
    finally:
        for c, flag in zip(classes, previous):
            c.RECORD_ANCHORS = flag


def main():
    from swarm_rescue.solutions.devtools import red_mapping_anchor_batch as driver
    previous = driver.MODULE
    worker = previous_run = None
    if '--worker' in sys.argv:
        request = Path(sys.argv[sys.argv.index('--worker')+1]).resolve()
        scope = next(p/'red_scope.json' for p in request.parents if (p/'red_scope.json').is_file())
        enabled = json.loads(scope.read_text(encoding='utf-8')).get('anchor_diagnostics', False)
        from swarm_rescue.solutions.devtools.optimization.evaluation import worker
        previous_run = worker.run
        def diagnosed_run(request_path):
            # The retained driver reads GO and installs OS limits before run().
            # Do not import the controller before that parent/worker handshake.
            with anchor_event_recording(enabled):
                return previous_run(request_path)
        worker.run = diagnosed_run
    try:
        driver.MODULE = MODULE  # Children must also enter this wrapper.
        driver.main()
    finally:
        driver.MODULE = previous
        if worker is not None:
            worker.run = previous_run


if __name__ == '__main__':
    main()
