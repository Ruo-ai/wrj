"""New-output-only RL11 / RL17 serial red comparison; no formal entry change."""
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import sys

from swarm_rescue.solutions.devtools import red_rn7_batch as driver
from swarm_rescue.solutions.devtools.red_rx10_diagnostics import recording as exploration
from swarm_rescue.solutions.devtools.red_lines11_diagnostics import recording as inputs
from swarm_rescue.solutions.devtools.red_local13_diagnostics import recording as coverage
from swarm_rescue.solutions.devtools.red_p2l17_diagnostics import recording as matching


def main():
    worker='--worker' in sys.argv
    if not worker:
        if '--full-pairs' in sys.argv:raise SystemExit('RL17 is red-only')
        if '--output' not in sys.argv:
            sys.argv+=['--output',str(Path('output/sf13')/('rl17_manual_'+datetime.now().strftime('%Y%m%d_%H%M%S')))]
        output=Path(sys.argv[sys.argv.index('--output')+1])
        if output.exists():raise SystemExit('Use a new output directory; RL17 does not resume old batches')
        if '--maps' not in sys.argv:sys.argv+=['--maps','Map06','Map08','Map09']
        if '--controllers' not in sys.argv:sys.argv+=['--controllers','rl11_lines','rl17_repair']
    saved=driver.MODULE,driver.EXTRA,driver.narrow_recording
    @contextmanager
    def diagnosed(path):
        with saved[2](path),exploration(path.with_name('exploration_diagnostics.json')),inputs(path.with_name('line_diagnostics.json')),coverage(path.with_name('local13_diagnostics.json')),matching(path.with_name('p2l17_diagnostics.json')):
            yield
    try:
        driver.MODULE='swarm_rescue.solutions.devtools.red_p2l17_batch'
        driver.EXTRA={'rl11_lines':('my_drone_red_lines11','MyDroneRedLines11'),
            'rl17_p2l':('my_drone_red_p2l17','MyDroneRedP2L17'),
            'rl17_repair':('my_drone_red_repair17','MyDroneRedRepair17'),
            'rl17_off':('my_drone_red_p2l17','MyDroneRedP2L17Off')}
        driver.narrow_recording=diagnosed
        driver.main()
    finally:driver.MODULE,driver.EXTRA,driver.narrow_recording=saved
    if not worker:
        if '--run' in sys.argv and '--smoke' not in sys.argv:
            from swarm_rescue.solutions.devtools.red_p2l17_review import review
            review(output)
        else:print('Atlas command: python -B -m swarm_rescue.solutions.devtools.red_p2l17_review --batch '+str(output),flush=True)


if __name__=='__main__':main()
