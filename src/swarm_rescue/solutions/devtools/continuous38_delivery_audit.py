"""Post-run evidence audit and supplementary figures; never starts a flight."""
import os, json, hashlib, importlib.util
from pathlib import Path
os.environ['PYGLET_SHADOW_WINDOW']='false';os.environ['MPLBACKEND']='Agg'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from swarm_rescue.solutions.red_boundary33 import FinalBoundary33

ROOT=Path(__file__).resolve().parents[4]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    r=module('continuous38_supplement','output/sf13/continuous38_report_20261122.py')
    driver=module('continuous38_frozen_check','output/sf13/continuous38_run_20261122.py')
    driver.base.validate(r.BATCH)
    read=r.read;atlas=r.atlas;navplot=r.navplot;out=r.OUT
    data=read(out/'data.json');pairs=[read(r.OLD/'pairs.json')[0],read(r.BATCH/'pairs.json')[0]]
    loaded=[atlas.inspect(p,r.BATCH) for p in pairs];nav=[navplot.load(p) for p in pairs]
    checks=[];supp=[];requests=[]
    def check(name,condition):
        checks.append(dict(check=name,passed=bool(condition)));assert condition,name
    check('frozen plan, source snapshot and historical observations unchanged',True)
    check('exactly one new candidate attempt',len(list((r.BATCH/'runs').glob('**/attempt_*')))==1)
    for j,(p,(m,d),n) in enumerate(zip(pairs,loaded,nav)):
        m['version']=data['metrics'][j]['name'];run=Path(p['red_run']);req=read(run/'request.json');requests.append(req)
        check(f'{j}: native red valid and 10 actual bombs',m['valid_red'] and m['placed_count']==10)
        integrity=read(run/'integrity.json')
        check(f'{j}: all saved artifact hashes',all(atlas.sha(run/f)==value for f,value in integrity['files'].items()))
        ident=read(run/'loaded_identity.json')
        check(f'{j}: all loaded source hashes',all(atlas.sha(ROOT/f)==value for f,value in ident['loaded_sources'].items()))
        check(f'{j}: five correct actual controller instances',len(ident['instances'])==5 and all(x.endswith(req['controller_class']) for x in ident['instances'].values()))
        resource=read(run/'resources.json')
        check(f'{j}: 1 CPU / 4 GB and normal worker exit',resource['cpu_count']==1 and resource['memory_limit_bytes']==4000000000 and resource['exit_code']==0 and resource['status']=='exited')
        check(f'{j}: passive diagnostics have no errors',not read(run/'entry37_diagnostics.json')['errors'])
        check(f'{j}: received map bytes bind to submitted pixels',not m['audit_issues'])
        for f in ('mapper_legal_inputs.npz','dualmap32_snapshots.npz','exploration_stages.npz'):
            check(f'{j}: {f} saved',(run/f).stat().st_size>0)
        renderer=FinalBoundary33(d['shape'][::-1])
        actual=renderer.fuse({int(k):v[::3,::3] for k,v in n['sources'].items()})
        check(f'{j}: received-source renderer replay exactly matches actual submission',np.array_equal(actual,d['pred']))
        pre=renderer.last_pre_boundary_pixels;added=actual&~pre
        scorer=atlas.MapExplorationScorer();score=scorer.score(d['truth'],pre,m['red_seconds'])
        surface=scorer.split_regions(d['truth'])[0].astype(bool);far=scorer.distance_field(d['truth'])>scorer.config.dead_zone
        yy,xx=np.indices(d['shape']);regions=dict(northwest=(yy<563)&(xx<622),northcenter=(yy<563)&(xx>=622)&(xx<1478),northeast=(yy<563)&(xx>=1478),south=yy>=563)
        regional={key:dict(wall_hits=int((region&surface&actual).sum()),wall_misses=int((region&surface&~actual).sum()),charged_pixels=int((region&far&actual).sum())) for key,region in regions.items()}
        native=read(run/'result.json')
        supp.append(dict(version=m['version'],pre_boundary_map_score_offline=100*float(score.s_map),
            boundary_score_delta_offline=m['map_score']-100*float(score.s_map),added_pixels=int(added.sum()),
            added_wall_hits=int((added&surface).sum()),added_charged_pixels=int((added&far).sum()),regional=regional,
            native_wall_score=native['phase1']['score_wall'],native_time_score=native['phase1']['score_time'],
            native_wall_raw_coverage=native['phase1']['wall_breakdown']['coverage_fraction'],
            candidate_replay_is_offline_not_another_flight=True))
        Image.fromarray(np.uint8(~pre)*255).save(out/f'{j}_pre_boundary_offline.png')
    common_keys=('official_map','stage','zones','map','map_file','round','seed','phase','drones','bombs','recording','ray_backend','visual','smoke','diagnostic_steps','diagnostic_profile','lifecycle_diagnostics')
    check('all native driver scenario/resource inputs identical except candidate identity/output',all(requests[0][k]==requests[1][k] for k in common_keys))
    check('requested seed and both official zones active',requests[1]['seed']==1492628140 and sorted(requests[1]['zones'])==['NO_COM_ZONE','NO_GPS_ZONE'])
    check('shared navigation/mapping/stopturn pose exactly identical at observed controls',max(data['shared_pose_agreement'].values())==0)
    check('no complete total or blue result inferred',all(x['total_score'] is None for x in data['metrics']))
    atlas.write(out/'supplement.json',dict(scope='offline diagnostics of actual received source maps; no additional flight',metrics=supp))
    atlas.write(r.BATCH/'audit.json',dict(passed=True,checks=checks,count=len(checks),source_snapshot_sha256=atlas.sha(r.BATCH/'source_snapshot.zip')))

    # Add track identity legend and leave white space for it.
    fig,axs=plt.subplots(2,3,figsize=(20,11))
    for row,((m,d),n) in enumerate(zip(loaded,nav)):
        navplot.draw(axs[row,0],navplot.nav_colors(n,'4'),m['version']+' · 4号导航图\n灰未知／白自由／黑占据；赛末快照',d['shape'])
        atlas.draw_submission(axs[row,1],m,d);atlas.draw_tracks(axs[row,2],m,d,'red')
    fig.suptitle('Map09 · seed 1492628140 · 官方特殊区域开启\n导航为各机规划证据，未经最终规整/补边；提交与轨迹均为实际运行产物',fontsize=16)
    fig.subplots_adjust(top=.86,bottom=.07,left=.01,right=.99,hspace=.34,wspace=.09)
    fig.legend([Line2D([0],[0],color=atlas.COLORS[i]) for i in range(5)],[f'无人机{i}' for i in range(5)],loc='lower center',ncol=5,frameon=False)
    r.save(fig,'comparison_v2.png')
    fig,axs=plt.subplots(2,3,figsize=(18,10));samples=np.load(out/'sampled_drift.npz')
    for row,((m,d),n) in enumerate(zip(loaded,nav)):
        navplot.draw(axs[row,0],n['sources']['4'],m['version']+'\n提交者实收4号源图（融合前）',d['shape'],True)
        atlas.frame(axs[row,1],d,True)
        state=d['red'][:data['common_cutoff_elapsed']-1,4];axs[row,1].plot(state['x']+850,550-state['y'],color=atlas.COLORS[4],lw=.6)
        axs[row,1].set_title(m['version']+' · 4号真实轨迹（到冻结步）')
        a=samples[f'{row}_drone4']
        for index,label in ((3,'导航'),(5,'建图')):
            y=np.where(a[:,1]>0,abs(a[:,index]),np.nan)
            axs[row,2].plot(a[:,0],y,label=label,lw=1)
        axs[row,2].set(xlabel='控制步数',ylabel='航向误差 / 度',title=m['version']+' · 4号漂移',xlim=(0,4644),ylim=(0,37))
        axs[row,2].legend();axs[row,2].grid(alpha=.2)
    fig.suptitle('4号机实际飞行核对 · 两版漂移同坐标轴；有GPS/罗盘的区间断开',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.95));r.save(fig,'drone4_flight_v2.png')
    fig,axs=plt.subplots(1,2,figsize=(18,7))
    for j,(ax,(m,d)) in enumerate(zip(axs,loaded)):
        ax.imshow(Image.open(d['native_wall']));ax.set_axis_off()
        ax.set_title(m['version']+f"\n原生墙体分 {supp[j]['native_wall_score']:.5f}；原始覆盖率 {supp[j]['native_wall_raw_coverage']:.2%}",fontsize=13)
    fig.suptitle('原生实际布弹墙体覆盖图 · 红为爆炸覆盖区域；内墙/边界权重由原生计分器计算',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95));r.save(fig,'wall_coverage.png')
    print(json.dumps(dict(audit_checks=len(checks),supplement=supp),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
