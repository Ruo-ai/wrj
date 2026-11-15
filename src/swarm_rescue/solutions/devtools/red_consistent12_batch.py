"""Red-only RL11/RL12 driver, unchanged common evaluator and official entry."""
from contextlib import contextmanager
import sys
from swarm_rescue.solutions.devtools import red_rn7_batch as driver
from swarm_rescue.solutions.devtools.red_rx10_diagnostics import recording as exploration
from swarm_rescue.solutions.devtools.red_lines11_diagnostics import recording as inputs
from swarm_rescue.solutions.devtools.red_consistent12_diagnostics import recording as navigation


def main():
    old_module,old_extra,old_record=driver.MODULE,driver.EXTRA,driver.narrow_recording
    @contextmanager
    def diagnosed(path):
        with old_record(path),exploration(path.with_name('exploration_diagnostics.json')),inputs(path.with_name('line_diagnostics.json')),navigation(path.with_name('navigation_diagnostics.json')):
            yield
    try:
        driver.MODULE='swarm_rescue.solutions.devtools.red_consistent12_batch'
        driver.EXTRA={
            'rl11_lines':('my_drone_red_lines11','MyDroneRedLines11'),
            'rl12_lag':('my_drone_red_consistent12','MyDroneRedConsistent12'),
            'rl12_heading':('my_drone_red_consistent12','MyDroneRedHeading12'),
            'rl12_stable':('my_drone_red_consistent12','MyDroneRedStable12'),
            'rl12_nav':('my_drone_red_navlines12','MyDroneRedNavLines12'),
            'rl12_off':('my_drone_red_consistent12','MyDroneRedConsistent12Off')}
        driver.narrow_recording=diagnosed
        if '--worker' not in sys.argv and '--controllers' not in sys.argv:
            sys.argv+=['--controllers','rl11_lines','rl12_nav']
        driver.main()
    finally:driver.MODULE,driver.EXTRA,driver.narrow_recording=old_module,old_extra,old_record


if __name__=='__main__':main()
