"""RL13 red-only single-factor comparisons; default prepares, --run executes serially."""
from contextlib import contextmanager
import sys
from swarm_rescue.solutions.devtools import red_rn7_batch as driver
from swarm_rescue.solutions.devtools.red_rx10_diagnostics import recording as exploration
from swarm_rescue.solutions.devtools.red_lines11_diagnostics import recording as inputs
from swarm_rescue.solutions.devtools.red_local13_diagnostics import recording as local


def main():
    old_module,old_extra,old_record=driver.MODULE,driver.EXTRA,driver.narrow_recording
    @contextmanager
    def diagnosed(path):
        with old_record(path),exploration(path.with_name('exploration_diagnostics.json')),inputs(path.with_name('line_diagnostics.json')),local(path.with_name('local13_diagnostics.json')):
            yield
    try:
        driver.MODULE='swarm_rescue.solutions.devtools.red_local13_batch'
        driver.EXTRA={
            'rl11_lines':('my_drone_red_lines11','MyDroneRedLines11'),
            'rl13_local':('my_drone_red_local13','MyDroneRedLocal13'),
            'rl13_entry':('my_drone_red_entry13','MyDroneRedEntry13'),
            'rl13_segments':('my_drone_red_entry13','MyDroneRedSegments13'),
            'rl13_off':('my_drone_red_local13','MyDroneRedLocal13Off')}
        driver.narrow_recording=diagnosed
        if '--worker' not in sys.argv and '--controllers' not in sys.argv:
            sys.argv+=['--controllers','rl11_lines','rl13_local']
        driver.main()
    finally:driver.MODULE,driver.EXTRA,driver.narrow_recording=old_module,old_extra,old_record


if __name__=='__main__':main()
