"""RX10 red-only comparisons. Explicit --run; no default batch execution."""
from contextlib import contextmanager
import sys
from swarm_rescue.solutions.devtools import red_rn7_batch as driver
from swarm_rescue.solutions.devtools.red_rx10_diagnostics import recording


def main():
    old_module,old_extra,old_record=driver.MODULE,driver.EXTRA,driver.narrow_recording
    @contextmanager
    def diagnosed(destination):
        with old_record(destination),recording(destination.with_name('exploration_diagnostics.json')):yield
    try:
        driver.MODULE='swarm_rescue.solutions.devtools.red_rx10_batch'
        driver.EXTRA={
            'rn7_b':('my_drone_red_rn7','MyDroneRedRN7B'),
            'rx10_off':('my_drone_red_branches10','MyDroneRedBranches10'),
            'rx10_branch':('my_drone_red_branches10','MyDroneRedRX10Branch'),
            'rx10_finish':('my_drone_red_branches10','MyDroneRedRX10Finish'),
            'rx10_combined':('my_drone_red_branches10','MyDroneRedRX10Combined')}
        driver.narrow_recording=diagnosed
        if '--worker' not in sys.argv and '--controllers' not in sys.argv:
            sys.argv+=['--controllers','advanced2','rn7_b','rx10_branch','rx10_finish']
        driver.main()
    finally:driver.MODULE,driver.EXTRA,driver.narrow_recording=old_module,old_extra,old_record


if __name__=='__main__':main()
