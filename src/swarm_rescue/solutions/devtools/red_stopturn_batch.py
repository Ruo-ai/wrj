"""Red-owned S6 A/B/C driver; defaults to prepare-only, serial execution via --run.

Retained evaluator is reused by process-local registration, never file edits.
Passive diagnostics are mandatory for this dedicated comparison driver.
"""
from pathlib import Path
import sys

MODULE='swarm_rescue.solutions.devtools.red_stopturn_batch'
EXTRA={
    's6_motion':('my_drone_red_stopturn','MyDroneRedStopTurn'),
    's6_radar':('my_drone_red_stopturn','MyDroneRedStopTurnRadar'),
    's6_off':('my_drone_red_stopturn','MyDroneRedStopTurnOff'),
}


def main():
    from swarm_rescue.solutions.devtools import red_mapping_cells_batch as driver
    original=(driver.MODULE,driver.EXTRA,driver.write_report)
    worker=old_run=None
    if '--worker' in sys.argv:
        from swarm_rescue.solutions.devtools.optimization.evaluation import worker
        old_run=worker.run
        def diagnosed_run(request):
            from swarm_rescue.solutions.devtools.red_stopturn_diagnostics import recording
            from swarm_rescue.solutions.devtools.optimization.evaluation.runner import read
            settings=read(request)
            if settings['phase']!='red': return old_run(request)
            with recording(Path(settings['run_dir'])/'motion_diagnostics.json'):
                return old_run(request)
        worker.run=diagnosed_run
    elif '--controllers' not in sys.argv:
        sys.argv+=['--controllers','advanced2_mapping','s6_motion','s6_radar']
    def report(output,plan,pairs):
        original[2](output,plan,pairs)
        path=output/'report.md'
        content=path.read_text(encoding='utf-8').replace('# A3/A4 红队专项逐轮图册','# S6 A/B/C 红队专项逐轮图册')
        content=content.replace('碰撞次数未直接记录，标为未知；不能把健康损失或回调异常次数当碰撞次数。',
            '碰撞接触开始次数与对齐的逐步位置误差见各attempt/motion_diagnostics.json；全程聚合，每12步有界抽样，不是完整传感器回放。')
        path.write_text(content,encoding='utf-8')
    try:
        driver.MODULE=MODULE;driver.EXTRA=EXTRA;driver.write_report=report
        driver.main()
    finally:
        driver.MODULE,driver.EXTRA,driver.write_report=original
        if worker is not None: worker.run=old_run


if __name__=='__main__':main()
