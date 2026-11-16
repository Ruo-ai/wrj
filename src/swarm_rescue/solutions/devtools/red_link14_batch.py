"""RL14 independent single-factor serial full red comparisons, --run opts in."""
from contextlib import contextmanager
import sys
from swarm_rescue.solutions.devtools import red_rn7_batch as driver
from swarm_rescue.solutions.devtools.red_rx10_diagnostics import recording as exploration
from swarm_rescue.solutions.devtools.red_lines11_diagnostics import recording as inputs
from swarm_rescue.solutions.devtools.red_local13_diagnostics import recording as coverage
from swarm_rescue.solutions.devtools.red_link14_diagnostics import recording as chain


def main():
    saved=driver.MODULE,driver.EXTRA,driver.narrow_recording
    @contextmanager
    def diagnosed(path):
        with saved[2](path),exploration(path.with_name('exploration_diagnostics.json')),inputs(path.with_name('line_diagnostics.json')),coverage(path.with_name('local13_diagnostics.json')),chain(path.with_name('link14_diagnostics.json')):
            yield
    try:
        driver.MODULE='swarm_rescue.solutions.devtools.red_link14_batch'
        driver.EXTRA={'rl11_lines':('my_drone_red_lines11','MyDroneRedLines11'),
            'rl14_delivery':('my_drone_red_link14','MyDroneRedDelivery14'),
            'rl14_opening':('my_drone_red_link14','MyDroneRedOpening14'),
            'rl14_relative':('my_drone_red_link14','MyDroneRedRelative14'),
            'rl14_off':('my_drone_red_link14','MyDroneRedLink14Off')}
        driver.narrow_recording=diagnosed
        if '--worker' not in sys.argv and '--controllers' not in sys.argv:
            sys.argv+=['--controllers','rl11_lines','rl14_delivery','rl14_opening','rl14_relative']
        driver.main()
    finally:driver.MODULE,driver.EXTRA,driver.narrow_recording=saved


if __name__=='__main__':main()
