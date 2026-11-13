"""Offline actual-wall union audit; source artifacts and scores stay immutable."""
import gc
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .runner import REPO,read,write,receipt,file_hash
from .scenarios import resolve_map
from .replay import validate_replay_sources


def increments(weights,masks,order):
    covered=np.zeros(weights.shape,bool);rows=[]
    multiplicity=masks.sum(axis=0)
    for i in order:
        mask=masks[i]
        rows.append(dict(bomb=int(i)+1,standalone=float(weights[mask].sum()),
            marginal=float(weights[mask & ~covered].sum()),exclusive=float(weights[mask & (multiplicity==1)].sum())))
        covered |= mask
    return rows,float(weights[covered].sum())


def audit(batch):
    from ...evaluate_red import configure_ray_backend
    from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
    from swarm_rescue.simulation.elements.sensor_disablers import ZoneType
    from swarm_rescue.simulation.reporting.place_score_manager import walls_and_boundary_grid,PlaceScoreManager
    configure_ray_backend('cpu')
    plan=read(batch/'plan.json');validate_replay_sources(plan['source_sha256'])
    rows=read(batch/'results.json');results=[]
    for name in sorted({r['map'] for r in rows}):
        item=next(s for s in plan['schedule'] if s['name']==name)
        gc.collect()
        cls=resolve_map(dict(map_file=str(REPO/item['map_file']),official_map=item.get('official_map')))
        world=cls(DroneMotionless,tuple(ZoneType[z] for z in item['scenario']['zones']),5)
        try:walls,boundary=walls_and_boundary_grid(world.playground)
        finally:world.playground.window.close();world=None
        weights=np.where(boundary,.5,1.)
        radius=round(.12*min(walls.shape))
        for row in (r for r in rows if r['map']==name):
            run=Path(row['red_run'])
            if read(run/'integrity.json')!=receipt(run,file_hash(batch/'plan.json')):raise ValueError('Red attempt changed')
            record=dict(map=name,controller=row['controller'],round=row['round'],run=str(run),bombs=[],
                full_valid=row['valid_score'],status='unavailable')
            layouts=list((run/'artifacts').glob('*_bombs.json'));recordings=list((run/'artifacts/recordings').glob('*.npz'))
            if len(layouts)!=1 or len(recordings)!=1:
                results.append(record);continue
            bombs=read(layouts[0])['bombs']
            if not bombs:
                record.update(status='no_actual_bombs',weighted_union=0.)
                results.append(record);continue
            xy=np.array([(b['x'],b['y']) for b in bombs])
            with np.load(recordings[0],allow_pickle=False) as z:states=z['bomb_states']
            if states.shape[1]!=len(bombs):
                record['status']='recording_layout_count_mismatch';results.append(record);continue
            final=states[-1];cost=np.linalg.norm(xy[:,None]-np.column_stack((final['x'],final['y']))[None,:],axis=2)
            indices,slots=linear_sum_assignment(cost)
            if not final['active'].all() or cost[indices,slots].max()>2:
                record['status']='recording_layout_alignment_ambiguous';results.append(record);continue
            first=np.argmax(states['active'][:,slots],axis=0)
            masks=np.array([PlaceScoreManager.wall_destruction_masks(walls,[p.tolist()],radius,boundary)[2].astype(bool) for p in xy])
            contributions,total=increments(weights,masks,np.argsort(first,kind='stable'))
            confirmations=[]
            for line in (run/'run.log').read_text(encoding='utf-8',errors='replace').splitlines():
                if line.startswith('V3_EVENT '):
                    event=json.loads(line[9:]);fs1=event.get('fs1') or {}
                    if event['event']=='placement_confirmed' and fs1.get('request'):confirmations.append(fs1['request'])
            for contribution in contributions:
                i=contribution['bomb']-1
                t=int(first[i]);same=int((first==t).sum())
                actual=[float(states[t,slots[i]][k]) for k in ('x','y')]
                contribution.update(actual_initial=actual,actual_final=xy[i].tolist(),first_frame=t+1,
                    simultaneous_count=same,estimated_pose_error=None,drone=None)
                # Establish owner only from a unique adjacent-time confirmation.
                # Do not guess source assignment from nearest final position.
                matches=[e for e in confirmations if abs(e['request_step']-(t+1))<=1]
                if same==1 and len(matches)==1:
                    e=matches[0];contribution.update(drone=e['source'],request=e,
                        estimated_pose_error=float(np.linalg.norm(np.array(actual)-e['estimated_position'])))
            official=(read(run/'result.json').get('phase1') or {}).get('wall_breakdown') or {}
            expected=official.get('weighted_destroyed')
            record.update(status='ok' if expected is not None and abs(total-expected)<.01 else 'official_union_unverified',
                weighted_union=total,official_weighted_union=expected,bombs=contributions)
            results.append(record)
    write(batch/'placement_audit.json',results)
    lines=['# 红队实际落点与真实新增墙面积（离线审计）','',
        '只读取冻结录像、日志和布局；真值不输入在线策略。贡献按首次记录顺序排列；同帧落弹的内部顺序未知，逐弹新增面积依赖表中顺序。',
        '布局采用实际导出最终位置；初次出现坐标另列。没有唯一确认对应时不猜所属机号或定位误差。','']
    for r in results:
        lines += [f"## {r['map']} / {r['controller']} / r{r['round']}",'',
            f"审计状态：{r['status']}；真实加权并集 {r.get('weighted_union')}；官方记录 {r.get('official_weighted_union')}；有效完整分 {r['full_valid']}。",'',
            '|弹|新增墙面积|单弹面积|独占面积|首次帧/同帧弹数|机号|初次落点定位误差(px)|','|---|---|---|---|---|---|---|']
        for b in r['bombs']:
            lines.append(f"|{b['bomb']}|{b['marginal']}|{b['standalone']}|{b['exclusive']}|{b['first_frame']}/{b['simultaneous_count']}|{b['drone']}|{b['estimated_pose_error']}|")
    (batch/'placement_audit.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path)
    audit(p.parse_args().batch.resolve())
