"""Red-owned A3/A5 driver. No shared registry, entry, or engine file changes.

Default is prepare-only. --run explicitly starts serial simulations. --full-pairs
adds the frozen adapted blue after each valid red; otherwise D/full score are
unmeasured. Registrations below live only in this dedicated evaluator process.
"""
import argparse
import os
from pathlib import Path
import sys

from swarm_rescue.solutions.devtools import evaluate_verified_revision as ev
from swarm_rescue.solutions.devtools.optimization.evaluation import runner
from swarm_rescue.solutions.devtools.optimization.evaluation.joint_batch import arguments
from swarm_rescue.solutions.devtools.optimization.evaluation.atlas import round_section
from swarm_rescue.solutions.devtools.optimization.map_generator.append_maps import append_lock

MODULE = 'swarm_rescue.solutions.devtools.red_mapping_anchor_batch'
EXTRA = {
    'a4_cells': ('my_drone_red_mapping_cells', 'MyDroneRedMappingCells'),
    'a4_off': ('my_drone_red_mapping_cells', 'MyDroneRedMappingCellsOff'),
    'a5_anchor': ('my_drone_red_mapping_anchor', 'MyDroneRedMappingAnchor'),
    'a5_off': ('my_drone_red_mapping_anchor', 'MyDroneRedMappingAnchorOff'),
}


def register_local():
    ev.CONTROLLERS = {**ev.CONTROLLERS, **EXTRA}


def worker_command(request):
    argv = [MODULE, '--worker', str(request)]
    bootstrap = (f'import runpy,sys; sys.path[:]={sys.path!r}; sys.prefix={sys.prefix!r}; '
                 f'sys.exec_prefix={sys.exec_prefix!r}; sys.argv={argv!r}; '
                 f'runpy.run_module({MODULE!r},run_name="__main__")')
    return [sys.executable, '-B', '-u', '-S', '-X', 'utf8', '-X', 'faulthandler', '-c', bootstrap]


def write_report(output, plan, pairs):
    from swarm_rescue.solutions.devtools.optimization.evaluation.report import flatten
    rows = []
    lines = ['# A3/A5 红队专项逐轮图册', '',
             '冻结B0=MyDroneBlueAdapted（开发参考，未验收为强对手）。无蓝阶段时，难度及官方原始/有效总分均未评测。',
             '轨迹覆盖、激光观察范围与地图分不同；本图册不把轨迹扩张率当作雷达观测率。',
             '碰撞次数未直接记录，标为未知；不能把健康损失或回调异常次数当碰撞次数。', '',
             '|图/轮/红版本|红有效|M|W|T|D|官方原始总分|官方有效总分|弹数|红步/秒|',
             '|---|---|---:|---:|---:|---|---|---|---:|---|']
    for pair in pairs:
        row = flatten(pair)
        row['red_valid'] = runner.phase_valid(pair['red'], 'red') and not pair['red'].get('smoke')
        row['blue_evaluated'] = pair['blue'] is not None
        row['collision_count'] = None
        if pair['blue'] is None:
            row['failure_reason'] = None if row['red_valid'] else row['failure_reason']
            row['full_status'] = 'not_evaluated'
        rows.append(row)
        def f(value): return '未评测' if value is None else f'{value:.4f}'
        lines.append(f"|{row['map']}/r{row['round']}/{row['controller']}|{row['red_valid']}|"+
                     '|'.join(f(row[k]) for k in ('map_score','wall_score','time_score','difficulty_score','raw_total_score','total_score'))+
                     f"|{row['placed_count']}|{row['red_steps']}/{f(row['red_walltime'])}|")
    for pair in pairs:
        lines += round_section(pair, output)
    runner.write(output/'results.json', rows)
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    register_local()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--maps', nargs='+', default=['Map08'])
    parser.add_argument('--controllers', nargs='+', choices=['advanced2','advanced2_mapping',*EXTRA],
                        default=['advanced2_mapping','a5_anchor'])
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--anchor-diagnostics', action='store_true', help='Bounded recovery events and mapper timing; recorded in red_scope.json')
    parser.add_argument('--full-pairs', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.worker:
        if sys.stdin.readline().strip() != 'GO':
            raise SystemExit('Parent did not authorize startup')
        if os.name != 'nt':
            from swarm_rescue.solutions.devtools.optimization.evaluation.resources import linux_limit
            linux_limit(int(os.environ['G3_MEMORY_BYTES']))
        from swarm_rescue.solutions.devtools.optimization.evaluation.worker import run
        scope_file = next(p/'red_scope.json' for p in args.worker.parents if (p/'red_scope.json').is_file())
        scope = runner.read(scope_file)
        if scope.get('anchor_diagnostics'):
            import time
            from swarm_rescue.solutions.my_drone_red_mapping_anchor import StableAnchorMapper
            instances = []
            original_init, original_update = StableAnchorMapper.__init__, StableAnchorMapper.update
            def diagnosed_init(self, *a, **kw):
                original_init(self, *a, **kw)
                self.RECORD_ANCHORS = True
                self._diagnostic_seconds = 0.
                self._diagnostic_calls = 0
                self._diagnostic_max_pending_array_bytes = 0
                instances.append(self)
            def diagnosed_update(self, *a, **kw):
                before = time.perf_counter()
                try:
                    return original_update(self, *a, **kw)
                finally:
                    self._diagnostic_seconds += time.perf_counter()-before
                    self._diagnostic_calls += 1
                    if self.step % 32 == 0:
                        n = sum(f[1].nbytes+f[2].nbytes+f[5].nbytes for f in self.pending)
                        self._diagnostic_max_pending_array_bytes = max(self._diagnostic_max_pending_array_bytes, n)
            StableAnchorMapper.__init__, StableAnchorMapper.update = diagnosed_init, diagnosed_update
            try:
                code = run(args.worker)
            finally:
                runner.write(args.worker.parent/'anchor_diagnostics.json', dict(
                    scope='red mapper only; bounded events, no complete sensor replay',
                    pending_bytes='Arrays sampled every 32 steps; excludes Python objects and matcher; not process RSS',
                    mappers=[dict(stats=m.stats, events=list(m.anchor_events),
                         update_seconds=m._diagnostic_seconds, update_calls=m._diagnostic_calls,
                         maximum_pending_array_bytes_sampled=m._diagnostic_max_pending_array_bytes,
                         pending_scans=len(m.pending)) for m in instances]))
            raise SystemExit(code)
        raise SystemExit(run(args.worker))
    if args.output is None:
        parser.error('--output is required; old batches are never selected automatically')
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    task = arguments(args.output.resolve(), blue='adapted', smoke=args.smoke)
    task.map_names = args.maps
    task.controllers = args.controllers
    task.rounds = args.rounds
    task.diagnostic_steps = 128
    scope = dict(anchor_diagnostics=args.anchor_diagnostics, full_pairs=args.full_pairs, smoke=args.smoke, red_phases=len(args.maps)*args.rounds*len(args.controllers),
                 maximum_blue_phases=len(args.maps)*args.rounds*len(args.controllers) if args.full_pairs else 0)
    scope_path = task.output/'red_scope.json'
    if task.output.exists() and (not scope_path.is_file() or runner.read(scope_path) != scope):
        raise ValueError('Different/missing red-owned scope; use a new output directory')
    output, plan = runner.create_or_resume(task)
    if not scope_path.exists(): runner.write(scope_path, scope)
    if not args.run:
        print('Prepared only:', output, scope, flush=True)
        return
    runner.worker_command = worker_command
    pairs = []
    with append_lock(output):
        for item in plan['schedule']:
            runner.validate_sources(plan['source_sha256'])
            print(item['name'], item['controller'], 'red', flush=True)
            runner.write(output/'progress.json', dict(active=item, completed=len(pairs), expected=len(plan['schedule'])))
            red, red_run = runner.execute_phase(output, plan, item, 'red')
            blue, blue_run = None, None
            if args.full_pairs and (args.smoke or (runner.phase_valid(red,'red') and red.get('placed_count',0)>0)):
                runner.validate_sources(plan['source_sha256'])
                blue, blue_run = runner.execute_phase(output, plan, item, 'blue', red_run)
            pairs.append(dict(**item, red=red, blue=blue, red_run=str(red_run),
                              blue_run=str(blue_run) if blue_run else None))
            write_report(output, plan, pairs)
        runner.write(output/'progress.json', dict(active=None, completed=len(pairs), expected=len(plan['schedule']), complete=True))
    print('Completed:', output, flush=True)


if __name__ == '__main__':
    main()
