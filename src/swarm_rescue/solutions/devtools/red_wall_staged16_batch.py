"""RL11 / RL16-A / RL16-B serial red-only runner; --run explicitly required."""
from contextlib import contextmanager
import sys
from swarm_rescue.solutions.devtools import red_rn7_batch as driver
from swarm_rescue.solutions.devtools.red_rx10_diagnostics import recording as exploration
from swarm_rescue.solutions.devtools.red_lines11_diagnostics import recording as inputs
from swarm_rescue.solutions.devtools.red_local13_diagnostics import recording as coverage
from swarm_rescue.solutions.devtools.red_wall_staged16_diagnostics import recording as experiment


def main():
    saved=driver.MODULE,driver.EXTRA,driver.narrow_recording
    @contextmanager
    def diagnosed(path):
        with saved[2](path),exploration(path.with_name('exploration_diagnostics.json')),inputs(path.with_name('line_diagnostics.json')),coverage(path.with_name('local13_diagnostics.json')),experiment(path.with_name('wall_staged16_diagnostics.json')):yield
    try:
        driver.MODULE='swarm_rescue.solutions.devtools.red_wall_staged16_batch'
        driver.EXTRA={'rl11_lines':('my_drone_red_lines11','MyDroneRedLines11'),
            'rl16_wall':('my_drone_red_wall16','MyDroneRedWall16'),
            'rl16_staged':('my_drone_red_staged16','MyDroneRedStaged16'),
            'rl16_wall_off':('my_drone_red_wall16','MyDroneRedWall16Off'),
            'rl16_staged_off':('my_drone_red_staged16','MyDroneRedStaged16Off')}
        driver.narrow_recording=diagnosed
        if '--worker' not in sys.argv and '--controllers' not in sys.argv:sys.argv+=['--controllers','rl11_lines','rl16_wall','rl16_staged']
        driver.main()
    finally:driver.MODULE,driver.EXTRA,driver.narrow_recording=saved


if __name__=='__main__':main()
