"""Red-only RN7 A/B/C one-factor driver, using retained evaluator unchanged."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys

MODULE='swarm_rescue.solutions.devtools.red_rn7_batch'
EXTRA={name:('my_drone_red_rn7',cls) for name,cls in (
    ('rn7_a','MyDroneRedRN7A'),('rn7_b','MyDroneRedRN7B'),('rn7_c','MyDroneRedRN7C'),
    ('rn7_off','MyDroneRedRN7Off'))}


@contextmanager
def narrow_recording(destination):
    from swarm_rescue.simulation.gui_map import gui_sr
    original=gui_sr.run_control;rows={}
    def control(drone):
        command=original(drone)
        narrow=getattr(drone,'_rn7',None)
        if narrow is not None:
            rows[str(drone.identifier)]=dict(stats=dict(narrow.stats),events=list(narrow.events),
                                            episodes=narrow.episodes)
        return command
    gui_sr.run_control=control
    try:yield
    finally:
        gui_sr.run_control=original
        with Path(destination).open('x',encoding='utf-8') as f:
            json.dump(dict(schema='rn7-events-v1',drones=rows),f,indent=2)


def main():
    from swarm_rescue.solutions.devtools import red_mapping_cells_batch as driver
    original=(driver.MODULE,driver.EXTRA,driver.write_report);worker=old_run=None
    if '--worker' in sys.argv:
        from swarm_rescue.solutions.devtools.optimization.evaluation import worker
        old_run=worker.run
        def diagnosed_run(request):
            from swarm_rescue.solutions.devtools.red_stopturn_diagnostics import recording
            from swarm_rescue.solutions.devtools.optimization.evaluation.runner import read
            settings=read(request)
            if settings['phase']!='red':return old_run(request)
            run=Path(settings['run_dir'])
            with recording(run/'motion_diagnostics.json'),narrow_recording(run/'narrow_diagnostics.json'):
                return old_run(request)
        worker.run=diagnosed_run
    elif '--controllers' not in sys.argv:sys.argv+=['--controllers','rn7_a','rn7_b','rn7_c']
    def report(output,plan,pairs):
        original[2](output,plan,pairs)
        path=output/'report.md'
        text=path.read_text(encoding='utf-8').replace('# A3/A4 红队专项逐轮图册','# RN7 独立窄路候选逐轮图册')
        text=text.replace('碰撞次数未直接记录，标为未知；不能把健康损失或回调异常次数当碰撞次数。',
                          '实际接触次数见 motion_diagnostics.json；局部窄路事件见 narrow_diagnostics.json。')
        path.write_text(text,encoding='utf-8')
    try:
        driver.MODULE=MODULE;driver.EXTRA=EXTRA;driver.write_report=report;driver.main()
    finally:
        driver.MODULE,driver.EXTRA,driver.write_report=original
        if worker is not None:worker.run=old_run


if __name__=='__main__':main()
