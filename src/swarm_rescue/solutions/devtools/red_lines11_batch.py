"""RL11 serial red comparisons; default prepare-only, --run starts explicitly."""
from contextlib import contextmanager
import sys
from swarm_rescue.solutions.devtools import red_rn7_batch as driver
from swarm_rescue.solutions.devtools.red_rx10_diagnostics import recording as evidence
from swarm_rescue.solutions.devtools.red_lines11_diagnostics import recording as accuracy


def main():
    old_module,old_extra,old_record=driver.MODULE,driver.EXTRA,driver.narrow_recording
    @contextmanager
    def diagnosed(destination):
        with old_record(destination),evidence(destination.with_name('exploration_diagnostics.json')),accuracy(destination.with_name('line_diagnostics.json')):
            yield
    try:
        driver.MODULE='swarm_rescue.solutions.devtools.red_lines11_batch'
        driver.EXTRA={
            'rn7_b':('my_drone_red_rn7','MyDroneRedRN7B'),
            'rl11_lines':('my_drone_red_lines11','MyDroneRedLines11'),
            'rl11_slow':('my_drone_red_lines11','MyDroneRedLines11Slow'),
            'rl11_combined':('my_drone_red_lines11','MyDroneRedLines11Combined'),
            'rl11_off':('my_drone_red_lines11','MyDroneRedLines11Off')}
        driver.narrow_recording=diagnosed
        if '--worker' not in sys.argv and '--controllers' not in sys.argv:
            sys.argv+=['--controllers','rn7_b','rl11_lines','rl11_slow']
        driver.main()
    finally:driver.MODULE,driver.EXTRA,driver.narrow_recording=old_module,old_extra,old_record


if __name__=='__main__':main()
