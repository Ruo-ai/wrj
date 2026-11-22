"""Resume the fixed schedule after coordinator-only exclusive-write failure.

The frozen worker, contestant, plan, first flight and random draw are untouched.
This coordinator uses the existing mutable JSON writer for cumulative receipts.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
from swarm_rescue.solutions.devtools import delivery39r2_random3 as batch


def main():
    output=batch.OUT;plan=batch.validate(output);driver=batch.driver
    receipt=output/'coordinator_continuation.json'
    identity=dict(script=Path(__file__).resolve().relative_to(batch.ROOT).as_posix(),
        sha256=batch.sha(Path(__file__).resolve()),plan_sha256=batch.sha(output/'plan.json'),
        created_utc=datetime.now(timezone.utc).isoformat(),
        reason='Original coordinator used write_new for progress/pairs; stopped before second flight. Resume only unstarted slots with mutable cumulative-receipt writer.',
        worker_entrypoint_unchanged=str(batch.SCRIPT))
    if receipt.exists():
        saved=batch.read(receipt)
        assert saved['sha256']==identity['sha256'] and saved['plan_sha256']==identity['plan_sha256']
    else:batch.write(receipt,identity)
    pairs=batch.read(output/'pairs.json');delivery=batch.read(output/'delivery_checks.json')
    assert len(pairs)==len(delivery)
    for p,item in zip(pairs,plan['schedule']):
        assert (p['map'],p['round'],p['seed'],p['controller'])==(item['map'],item['round'],item['seed'],item['controller'])
        attempt=Path(p['red_run'])
        assert all(batch.sha(attempt/f)==v for f,v in batch.read(attempt/'integrity.json')['files'].items())
    driver.register();driver.use_ascii_temp();driver.runner.worker_command=driver.command
    write=driver.runner.write
    with driver.append_lock(output):
        for item in plan['schedule'][len(pairs):]:
            batch.validate(output)
            phase=output/'runs'/f"{item['map']}_r{item['round']:02d}"/batch.SELECTOR/'red'
            driver.no_retry_guard(phase)
            write(output/'progress.json',dict(complete=False,completed=len(pairs),expected=6,
                current_map=item['map'],current_seed=item['seed']))
            print('START',item['execution_index'],'/6',item['map'],item['seed'],flush=True)
            result,attempt=driver.runner.execute_phase(output,plan,item,'red',retry_failed=False)
            issues=driver.phase_issues(result,attempt,plan,'red')
            for name in ('entry37_diagnostics.json','continuous38_diagnostics.json','delivery39_diagnostics.json'):
                if not (attempt/name).is_file():issues.append('missing_'+name)
                elif batch.read(attempt/name)['errors']:issues.append('observer_errors_'+name)
            pairs.append(dict(item,red=result,red_run=str(attempt),blue=None,blue_run=None,
                red_issues=issues,blue_issues=['not_requested']))
            write(output/'pairs.json',pairs)
            check=batch.delivery_check(attempt,issues)
            delivery.append(dict(map=item['map'],round=item['round'],seed=item['seed'],**check))
            write(output/'delivery_checks.json',delivery)
            print('FINISHED',item['execution_index'],'/6',item['map'],item['seed'],
                'map',result.get('phase1',{}).get('score_map'),'delivery',check,flush=True)
    batch.validate(output)
    done=dict(complete=True,new_red_runs=len(pairs),new_blue_runs=0,retries=0,
        valid_red=sum(not p['red_issues'] for p in pairs),delivery_complete=sum(x['passed'] for x in delivery),
        total_score=None,completed=len(pairs),expected=6,coordinator_only_resume=True)
    write(output/'completion.json',done);write(output/'progress.json',done)
    print('COMPLETE',json.dumps(done),flush=True)


if __name__=='__main__':main()
