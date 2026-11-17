"""Offline single-factor recovery weights, with fixed recorded matcher decisions.

No controller candidate is registered. The second evidence grid never enters
navigation, scan matching, position correction, or communication. It replaces
only .8 at recovery with frame[4], for BOTH free erasure and wall insertion.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from swarm_rescue.solutions.devtools.red_pair19_batch import (
    REPO, read, write, digest, source_files, environment)
from swarm_rescue.solutions.devtools.red_recovery18_audit import (
    load_npz, semantic_objects, legal_arguments, evidence_grid, Metrics, radio_states)
from swarm_rescue.solutions.my_drone_red_repair17 import RepairMapper17
from swarm_rescue.solutions.my_drone_red_mapping import anchor_frames, wall_points, free_points
from swarm_rescue.solutions.semifinal_localization import transform, wrap
from swarm_rescue.solutions.devtools.optimization.evaluation import runner
from swarm_rescue.solutions.devtools.red_stopturn_match_report import COMPARISON_HEADER_HEIGHT

HISTORY=REPO/'output/sf13/rl17_m89_once_20261116'
AUDIT18=REPO/'output/sf13/rl18_audit_20261116/stage_a_v2'


def recovery_ticks(data):
    good=data[:,1].astype(bool)&data[:,2].astype(bool)
    return (np.flatnonzero(good[1:]&~good[:-1])+2).tolist()


def corrected_frames(mapper, args):
    gps,compass,odom,*_=args
    pos,heading=mapper.position.copy(),mapper.heading
    if odom is not None and np.isfinite(odom).all():
        pos+=odom[0]*np.array([np.cos(heading+odom[1]),np.sin(heading+odom[1])])
        heading=wrap(heading+odom[2])
    return anchor_frames(mapper.pending,mapper.start,pos,heading,gps,compass,mapper.step+1)


def metric_count(metric, pred):
    result=metric.count(pred)
    result['missed']=int(metric.surface.sum())-result['hits']
    return result


def change(a,b):
    return {k:b[k]-a[k] for k in ('hits','missed','charged','map_score')}


def check_radio(events, submission):
    """Accepted source versions must have been published and cannot be future."""
    published={}; states={}
    for e in events:
        if e['event']=='local_publish':
            key=(e['source'],e['version'])
            if e['holder']!=e['source'] or e['version']>e['step']:
                raise ValueError('Invalid publication')
            if key in published and published[key]['sha256']!=e['sha256']:
                raise ValueError('Publication version changed payload')
            published[key]=e
    for e in events:
        if e['event'] in ('local_publish','received'):
            origin=published.get((e['source'],e['version']))
            if origin is None or origin['step']>e['step'] or origin['sha256']!=e['sha256'] or e['version']>e['step']:
                raise ValueError('Missing origin, future information or payload mismatch')
    states=radio_states(events,submission['drone'],[submission['step']])[submission['step']]
    if states!={int(k):v for k,v in submission['sources'].items()}:
        raise ValueError('Submission sources disagree with accepted version log')
    return published,states


def prepare(output):
    if output.exists():raise FileExistsError(output)
    rows=[];inputs={}
    history_plan=read(HISTORY/'plan.json');plan_sha=digest(HISTORY/'plan.json')
    for item in history_plan['schedule']:
        if item['controller']!='rl17_repair':continue
        run=HISTORY/'runs'/f"{item['name']}_r{item['round']:02d}"/item['controller']/'red/attempt_001'
        result,checked=runner.load_done(run.parent,plan_sha)
        if checked!=run or not runner.phase_valid(result,'red'):raise ValueError('Invalid historical input')
        d=read(AUDIT18/f"{item['name']}_rl17_repair/results.json")
        if not d['valid']:raise ValueError('RL18 baseline audit invalid')
        for path,sha in d['source_code'].items():
            if digest(path)!=sha:raise ValueError('Historical implementation changed: '+path)
        for path,sha in d['source_hashes'].items():
            if digest(run/path)!=sha:raise ValueError('Historical tape changed: '+path)
        for path in run.rglob('*'):
            if path.is_file():inputs[str(path)]=digest(path)
        inputs[str(AUDIT18/f"{item['name']}_rl17_repair/results.json")]=digest(AUDIT18/f"{item['name']}_rl17_repair/results.json")
        meta=read(run/'line_diagnostics.json'); detail=read(run/'p2l17_diagnostics.json');diag=read(run/'exploration_diagnostics.json')
        if not meta['mapper_inputs_complete'] or meta['errors'] or detail['errors'] or diag['errors']:raise ValueError('Incomplete diagnostics')
        check_radio(detail['radio'],diag['submissions'][0])
        with np.load(run/'mapper_legal_inputs.npz',allow_pickle=False) as tape:
            events=[dict(owner=int(k[6:]),step=tick) for k in tape.files if k.startswith('drone_') for tick in recovery_ticks(tape[k])]
        rows.append(dict(map=item['name'],round=item['round'],seed=item['seed'],run=str(run),events=events))
    if not rows:raise ValueError('No RL17-R inputs')
    output.mkdir(parents=True)
    plan=dict(schema='recovery-weight19-v1',rows=rows,input_sha256=inputs,
        source_sha256=source_files([]),environment=environment(),
        event_count=sum(len(r['events']) for r in rows),variants=['current_uniform_0.8','preserve_frame_weight'],
        frozen_factors=['all trajectories','recorded actual scan matcher return values',
            'all positions/headings','anchor_frames endpoint correction','normal scans and pending logic',
            'published source/version and actual accepted receipt timing','submission holder/time and postprocessing'],
        scope='conditional offline map counterfactual, not a new flight or verified radio experiment')
    write(output/'plan.json',plan);write(output/'prepared.json',dict(plan_sha256=digest(output/'plan.json'),executed=0))
    print('Prepared offline:',len(rows),'historical tapes,',plan['event_count'],'recovery transitions; 0 flight',flush=True)


def validate_prepared(folder,plan):
    if digest(folder/'plan.json')!=read(folder/'prepared.json')['plan_sha256']:raise ValueError('Plan changed')
    runner.validate_sources(plan['source_sha256'])
    if environment()!=plan['environment']:raise ValueError('Dependency environment changed')
    for path,sha in plan['input_sha256'].items():
        if digest(path)!=sha:raise ValueError('Input changed: '+path)


def replay(row, out, smoke=False):
    out.mkdir(parents=True,exist_ok=False); started=time.perf_counter()
    run=Path(row['run']);result=read(run/'result.json');request=read(run/'request.json')
    inputs=load_npz(run/'mapper_legal_inputs.npz');stages=load_npz(run/'exploration_stages.npz')
    recorded=load_npz(run/'p2l17_diagnostics.npz')
    meta=read(run/'line_diagnostics.json');diag=read(run/'exploration_diagnostics.json');detail=read(run/'p2l17_diagnostics.json')
    objects=semantic_objects(inputs,meta);h,w=result['expected_shape']
    png=np.array(Image.open(next((run/'artifacts').glob('*_exploration_diff.png'))).convert('RGB'))
    truth=np.all(png[COMPARISON_HEADER_HEIGHT:COMPARISON_HEADER_HEIGHT+h,:w]==40,axis=2).astype(np.uint8)
    metric=Metrics(truth,(0,0,w-1,h-1));submission=diag['submissions'][0]
    published,source_state=check_radio(detail['radio'],submission)
    versions=defaultdict(set)
    for owner,version in published:versions[owner].add(version)
    for owner,version in source_state.items():versions[owner].add(version)
    for owner,info in diag['drones'].items():
        versions[int(owner)].update(x['published_sequence'] for x in info['history'])
    decisions={(x['drone'],x['step']):x['decision'] for x in detail['matches']}
    baseline_snaps={};candidate_snaps={};events=[];local=[];checks=[];arrays={}
    for ident in sorted(diag['drones'],key=int)[:1] if smoke else sorted(diag['drones'],key=int):
        owner=int(ident);mapper=RepairMapper17((w,h));candidate=np.zeros(mapper.shape,np.float32)
        data=inputs['drone_'+ident];angles=inputs['angles_'+ident];record=recorded['pose_'+ident]
        if not np.array_equal(data[:,0],np.arange(1,len(data)+1)):raise ValueError('Non-contiguous tape')
        if not np.array_equal(record[:,0],data[:,0]):raise ValueError('Pose/tape alignment')
        stop=min(len(data), (recovery_ticks(data)[0]+4 if recovery_ticks(data) else 64)) if smoke else len(data)
        def fixed_match(points,position,heading):
            decision=decisions[owner,mapper.step]
            offset=decision['offset']
            if decision['accepted']!=(offset is not None):raise ValueError('Incomplete matching decision')
            return (None if offset is None else np.asarray(offset,float)),decision['info']
        mapper.matcher.match=fixed_match
        max_pose_delta=0.;matched_publications=0;recoveries=set(recovery_ticks(data))
        for index,rowdata in enumerate(data[:stop]):
            tick=int(rowdata[0]);args=legal_arguments(rowdata,angles,objects[owner,tick])
            good=bool(rowdata[1] and rowdata[2]);recovery=tick in recoveries
            if recovery:
                before=mapper.mask();provisional=evidence_grid(mapper,mapper.pending)
                candidate_before=(candidate>=2.4)|(provisional>=2.4)
                frames=corrected_frames(mapper,args)
                counts=dict(Counter(str(f[4]) for f in frames))
                promoted=evidence_grid(mapper,mapper.pending,mapper.hits,.8)
                anchored=evidence_grid(mapper,frames,mapper.hits,.8)
                candidate=evidence_grid(mapper,frames,candidate)  # THE ONLY experimental change
            mapper.update(*args)
            actual=np.r_[mapper.position,mapper.heading]
            delta=float(np.max(np.abs(actual-record[index,4:7])))
            max_pose_delta=max(max_pose_delta,delta)
            if delta>1e-9:raise ValueError(f'Baseline pose mismatch d{owner} t{tick}: {delta}')
            if good and tick%4==0:
                mapper._erase(candidate,transform(free_points(args[3],args[4]),mapper.position,mapper.heading),1.)
                mapper._insert(candidate,transform(wall_points(args[3],args[4],args[5]),mapper.position,mapper.heading),1.)
            if recovery:
                # Check the counterfactual .8 implementation against original hits.
                expected=anchored.copy()
                if tick%4==0:
                    mapper._erase(expected,transform(free_points(args[3],args[4]),mapper.position,mapper.heading),1.)
                    mapper._insert(expected,transform(wall_points(args[3],args[4],args[5]),mapper.position,mapper.heading),1.)
                if not np.array_equal(expected,mapper.hits):raise ValueError('Recovery implementation mismatch')
                stage_masks=dict(baseline_before=before,candidate_before=candidate_before,
                    baseline_after=mapper.mask(),candidate_after=candidate>=2.4,
                    baseline_unanchored_promoted=promoted>=2.4,baseline_anchored=anchored>=2.4)
                scores={key:metric_count(metric,mapper.to_pixels(value)) for key,value in stage_masks.items()}
                events.append(dict(owner=owner,step=tick,frame_weights=counts,pending_count=len(frames),
                    scores=scores,weight_delta=change(scores['baseline_after'],scores['candidate_after']),
                    baseline_projection_delta=change(scores['baseline_unanchored_promoted'],scores['baseline_anchored'])))
                for k,v in stage_masks.items():arrays[f'd{owner}_t{tick}_{k}']=v
            if tick in versions[owner]:
                provisional=evidence_grid(mapper,mapper.pending)
                baseline_snaps[owner,tick]=mapper.mask().copy()
                candidate_snaps[owner,tick]=(candidate>=2.4)|(provisional>=2.4)
        for event in diag['drones'][ident]['history']:
            v=event['published_sequence']
            if v>stop:continue
            if not np.array_equal(baseline_snaps[owner,v],stages[f"published_{owner}_{event['step']}"]):
                raise ValueError('Saved published map differs')
            matched_publications+=1
        if not smoke:
            if not np.array_equal(mapper.mask(),stages['local_final_'+ident]):raise ValueError('Local final mismatch')
            version=source_state.get(owner)
            if version is not None and not np.array_equal(baseline_snaps[owner,version],stages[f"submitter_{submission['drone']}_source_{owner}"]):raise ValueError('Received source mask mismatch')
            provisional=evidence_grid(mapper,mapper.pending)
            a=mapper.to_pixels(mapper.mask());b=mapper.to_pixels((candidate>=2.4)|(provisional>=2.4))
            sa,sb=metric_count(metric,a),metric_count(metric,b)
            local.append(dict(owner=owner,baseline=sa,candidate=sb,delta=change(sa,sb),
                end_step=len(data),pending_count=len(mapper.pending)))
            arrays['local_baseline_'+ident]=a;arrays['local_candidate_'+ident]=b
        checks.append(dict(owner=owner,steps=stop,pose_max_abs=max_pose_delta,saved_publications_matched=matched_publications))
    final=None;merged=[]
    if not smoke:
        if len(events)!=len(row['events']):raise ValueError('Recovery event omitted')
        ticks={submission['step']}
        ticks.update(e['step'] for e in events if e['step']<=submission['step'])
        for e in events:
            receipts=[x['step'] for x in detail['radio'] if x['holder']==submission['drone'] and
                x['event'] in ('received','local_publish') and x['source']==e['owner'] and x['version']>=e['step'] and x['step']<=submission['step']]
            e['first_holder_receipt']=min(receipts) if receipts else None
            if receipts:ticks.add(min(receipts))
        for tick,state in radio_states(detail['radio'],submission['drone'],ticks).items():
            a=mapper.to_pixels(np.logical_or.reduce([baseline_snaps[o,v] for o,v in state.items()]))
            b=mapper.to_pixels(np.logical_or.reduce([candidate_snaps[o,v] for o,v in state.items()]))
            sa,sb=metric_count(metric,a),metric_count(metric,b)
            merged.append(dict(step=tick,actual_sources=state,baseline=sa,candidate=sb,delta=change(sa,sb)))
            if tick==submission['step']:
                actual=np.asarray(read(next((run/'artifacts').glob('*_exploration.json')))['grid'],np.uint8)
                if not np.array_equal(a,actual):raise ValueError('Baseline actual submitted pixels differ')
                if abs(sa['map_score']-result['phase1']['score_map'])>1e-6:raise ValueError('Official map score mismatch')
                final=merged[-1];arrays['submitted_baseline']=a;arrays['submitted_candidate']=b
                for label,pixels in [('baseline',a),('candidate',b)]:Image.fromarray(metric.rgb(pixels)).save(out/f'final_{label}.png')
    np.savez_compressed(out/'maps.npz',**arrays)
    report=dict(source=str(run),map=request['map'],round=request['round'],seed=request['seed'],
        scope='partial_real_tape_smoke_no_final_submission' if smoke else 'offline_conditional_submission_counterfactual',
        valid_full_replay=not smoke,checks=checks,event_count=len(events),expected_events=len(row['events']),
        events=events,per_drone_final=local,holder_timeline=merged,final=final,
        wall_seconds=time.perf_counter()-started,blue_difficulty=None,full_competition_score=None,
        limitations=['Recorded matcher decisions are replayed, not re-optimized; timing budgets cannot change the control arm.',
            'No actual candidate packets were transmitted. Payload length, navigation, source/version and accepted schedule are held fixed.',
            'All saved masks, every recorded pose and baseline submission must match; original full intermediate payload bytes are unavailable.',
            'Local intermediate metrics are not team scores. Truth is used only for scoring after evidence construction.',
            'Later-event candidate deltas include accumulated effects of earlier recoveries under the same single-factor policy.'])
    write(out/'results.json',report)
    print(row['map'],'events',len(events),'full',not smoke,'seconds',round(report['wall_seconds'],2),flush=True)
    return report


def report_markdown(output, results, smoke):
    lines=['# 恢复权重单因素离线实验','',
        '**暂无新飞行图。** 历史RL17-R轨迹/匹配/位置/发布接收时序固定，只改恢复帧墙命中和自由擦除权重。',
        '基线恢复权重0.8；候选保留每帧原值。正常帧1.0与缺测缓存逻辑不变。蓝队难度和完整总分未评测。',
        '这是条件离线提交图，不是新飞行或实际无线通信试验。无未来版本或所有机终图的理想化合并。']
    if smoke:lines+=['','**仅真实输入带片段冒烟，未完成64事件实验，不生成有效最终提交图结论。**']
    for r in results:
        label=f"{r['map']} / 历史 RL17-R / r{r['round']} / seed {r['seed']}"
        lines+=['',f'## {label}','',f"事件 {r['event_count']}/{r['expected_events']}；完整重建={r['valid_full_replay']}。"]
        if r['final']:
            lines+=['','|范围|基线M|候选M|Δ命中|Δ漏画|Δ受罚|ΔM|','|---|---|---|---|---|---|---|']
            for name,x in [('条件最终提交图',r['final'])]+[(f"单机{v['owner']}终图（非提交）",v) for v in r['per_drone_final']]:
                d=x['delta'];lines.append(f"|{name}|{x['baseline']['map_score']:.5f}|{x['candidate']['map_score']:.5f}|{d['hits']}|{d['missed']}|{d['charged']}|{d['map_score']:+.5f}|")
            for variant in ('baseline','candidate'):
                lines+=['',f"![{label} {variant} 离线图]({(output/r['map']/('final_'+variant+'.png')).as_posix()})",
                    '绿=真墙命中；蓝=漏画；红=受罚；黄=免罚预测；灰=真墙内部。']
        lines+=['','|机/恢复步|帧权重计数|Δ命中|Δ漏画|Δ受罚|ΔM|基线重投Δ受罚|首次提交者接收步|',
            '|---|---|---|---|---|---|---|---|']
        for e in r['events']:
            d=e['weight_delta'];lines.append(f"|{e['owner']}/{e['step']}|{e['frame_weights']}|{d['hits']}|{d['missed']}|{d['charged']}|{d['map_score']:+.5f}|{e['baseline_projection_delta']['charged']}|{e.get('first_holder_receipt','片段未评估')}|")
        lines+=['','上表前六项是累计候选与基线的单机恢复后差值；重投列是基线同权重下校正位姿的有序差值。不是最终团队因果分解。']
        if r['final'] and r['events']:
            # Requested positive/negative historical cases plus BOTH extremes
            # of the new factor. All events remain in the preceding table.
            wanted=(3,1755) if r['map']=='Map08' else (3,2793)
            selected=[e for e in r['events'] if (e['owner'],e['step'])==wanted]
            for e in (min(r['events'],key=lambda e:e['weight_delta']['map_score']),
                      max(r['events'],key=lambda e:e['weight_delta']['map_score'])):
                if e not in selected:selected.append(e)
            from swarm_rescue.solutions.devtools import red_stopturn_match_report as plot
            from swarm_rescue.solutions.devtools.red_lines11_review import FONT
            run=Path(r['source']);h,w=read(run/'result.json')['expected_shape']
            png=np.array(Image.open(next((run/'artifacts').glob('*_exploration_diff.png'))).convert('RGB'))
            truth=np.all(png[COMPARISON_HEADER_HEIGHT:COMPARISON_HEADER_HEIGHT+h,:w]==40,axis=2).astype(np.uint8)
            metric=Metrics(truth,(0,0,w-1,h-1));mapper=RepairMapper17((w,h))
            with np.load(output/r['map']/'maps.npz',allow_pickle=False) as maps:
                fig,axes=plot.plt.subplots(len(selected),3,figsize=(18,6*len(selected)),squeeze=False)
                for y,e in enumerate(selected):
                    for x,key in enumerate(('baseline_before','baseline_after','candidate_after')):
                        mask=maps[f"d{e['owner']}_t{e['step']}_{key}"];s=e['scores'][key]
                        axes[y,x].imshow(metric.rgb(mapper.to_pixels(mask)));axes[y,x].axis('off')
                        axes[y,x].set_title(f"d{e['owner']} t{e['step']} {key}\nM={s['map_score']:.4f}; hit={s['hits']}; missed={s['missed']}; charged={s['charged']}",fontsize=10)
                fig.suptitle(label+' / OFFLINE single-drone events, NOT team score',fontproperties=FONT,fontsize=14)
                fig.text(.5,.01,'Green=hit; blue=missed; red=charged; yellow=uncharged; gray=wall interior. Same scale per row.',ha='center')
                fig.tight_layout(rect=(0,.03,1,.95));path=output/r['map']/'event_cases.png';plot.save(fig,path)
            lines+=['',f'![{label} 指定正反案例及新权重两端案例]({path.as_posix()})',
                '展示指定历史事件及本图候选地图分差值的最大/最小事件，未省略其余事件。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if not a.run:return prepare(a.output.resolve())
    if not a.prepared:p.error('--prepared required')
    plan=read(a.prepared/'plan.json');validate_prepared(a.prepared,plan)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);write(out/'input_plan.json',plan)
    from swarm_rescue.solutions.devtools.red_p2l17_replay import ReplayProcess
    process=ReplayProcess();started=time.perf_counter();cpu=time.process_time()
    cv2.setNumThreads(1)
    results=[replay(r,out/r['map'],a.smoke) for r in (plan['rows'][:1] if a.smoke else plan['rows'])]
    validate_prepared(a.prepared,plan)
    write(out/'summary.json',dict(full=not a.smoke,planned_events=plan['event_count'],
        completed_events=sum(r['event_count'] for r in results),new_flights=0,
        resources=dict(wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
                       affinity=process.affinity,peak_rss_bytes=process.memory()),
        results=[dict(map=r['map'],events=r['event_count'],final=r['final']) for r in results]))
    report_markdown(out,results,a.smoke)


if __name__=='__main__':main()
