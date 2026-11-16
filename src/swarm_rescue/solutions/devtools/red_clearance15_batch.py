"""RC15 / RL11 serial full red-only comparison; --run opts in."""
from contextlib import contextmanager
import sys
from swarm_rescue.solutions.devtools import red_rn7_batch as driver
from swarm_rescue.solutions.devtools.red_rx10_diagnostics import recording as exploration
from swarm_rescue.solutions.devtools.red_lines11_diagnostics import recording as inputs
from swarm_rescue.solutions.devtools.red_local13_diagnostics import recording as coverage
from swarm_rescue.solutions.devtools.red_clearance15_diagnostics import recording as narrow

def main():
    saved=driver.MODULE,driver.EXTRA,driver.narrow_recording
    @contextmanager
    def diagnosed(path):
        with saved[2](path),exploration(path.with_name('exploration_diagnostics.json')),inputs(path.with_name('line_diagnostics.json')),coverage(path.with_name('local13_diagnostics.json')),narrow(path.with_name('clearance15_diagnostics.json')):yield
    try:
        driver.MODULE='swarm_rescue.solutions.devtools.red_clearance15_batch'
        driver.EXTRA={'rl11_lines':('my_drone_red_lines11','MyDroneRedLines11'),
            'rc15_clearance':('my_drone_red_clearance15','MyDroneRedClearance15'),
            'rc15_off':('my_drone_red_clearance15','MyDroneRedClearance15Off')}
        driver.narrow_recording=diagnosed
        if '--worker' not in sys.argv and '--controllers' not in sys.argv:sys.argv+=['--controllers','rl11_lines','rc15_clearance']
        driver.main()
    finally:driver.MODULE,driver.EXTRA,driver.narrow_recording=saved

if __name__=='__main__':main()
