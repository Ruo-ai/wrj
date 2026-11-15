"""RL13 full-map single-factor atlas, source/validity and delivery audit."""
import argparse
import json
from pathlib import Path
import numpy as np
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base
from swarm_rescue.solutions.devtools import red_rx10_review as evidence
from swarm_rescue.solutions.devtools import red_lines11_review as lines
from swarm_rescue.solutions.devtools import red_consistent12_review as panels

NAMES={'rl11_lines':'RL11 原版','rl13_local':'RL13-L 短局部图','rl13_entry':'RL13-E 入口墙对齐',
       'rl13_segments':'RL13-S 分段运动','rl13_off':'RL13 关闭对照'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);ph=lines.frozen(a.batch)
    rows=[];assets={};failed=[];panels.NAMES=NAMES
    raw_rows=base.read(a.batch/'results.json')
    for raw in raw_rows:
        try:
            row,data=base.review_row(raw,ph);evidence.audit(row,data);lines.accuracy(row,data)
            row['local13']=base.read(Path(row['attempt'])/'local13_diagnostics.json')
            gray=np.zeros_like(data['truth'],bool)
            for area in data['spec']['elements']:
                if area['type']=='no_gps_zone':
                    l,t=map(int,area['top_left']);r,b=map(int,area['bottom_right'])
                    gray[max(0,t):min(data['h'],b+1),max(0,l):min(data['w'],r+1)]=True
            row['gray_false_pixels']=int((gray&data['pred']&~data['truth'].astype(bool)).sum())
            with np.load(Path(row['attempt'])/'local13_diagnostics.npz') as f:
                visible=np.zeros(((data['h']+3)//4,(data['w']+3)//4),bool)
                for key in f.files:visible|=f[key].astype(bool)
            visible=np.repeat(np.repeat(visible,4,0),4,1)[:data['h'],:data['w']]
            free=~data['truth'].astype(bool)
            row['gray_sampled_visible_free_pixels']=int((gray&free&visible).sum())
            row['gray_free_pixels']=int((gray&free).sum())
            row['gray_sampled_visible_free_fraction']=int((gray&free&visible).sum())/max(1,int((gray&free).sum()))
            data['visible']=visible
            row['gray_observed_surface_fraction']=row['gray_sampled_observed']/max(1,row['gray_true_surface'])
            row['observation_fraction_limitation']='Actual wall endpoint sample every 40 steps; lower bound, not complete visibility.'
            rows.append(row);assets[row['map'],row['controller']]=data
        except Exception as exc:
            failed.append(dict(raw=raw,error=repr(exc)))
    maps=list(dict.fromkeys(r['map'] for r in rows));controllers=list(dict.fromkeys(r['controller'] for r in rows))
    doc=['# RL13 完整地图单因素对照','','每图每版一次完整红阶段。无蓝阶段，难度与完整官方原始/有效总分未评测。',
         '轨迹改变的组不能称相同输入；局部图组另核对完整合法输入和实际轨迹。','',
         '|图/版本|有效|M|W|T|弹|步|墙钟秒|墙/机接触|墙命中|收费误画|',
         '|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    for r in rows:
        doc.append(f"|{r['map']}/{NAMES[r['controller']]}|{r['red_valid']}|{r['map_score']:.4f}|{r['wall_score']:.4f}|{r['time_score']:.4f}|{r['placed_count']}|{r['red_steps']}|{r['red_walltime']:.3f}|{r['contacts_wall']}/{r['contacts_drone']}|{r['wall_hits']}|{r['charged_pixels']}|")
    equality=[]
    for m in maps:
        baseline=assets.get((m,'rl11_lines'))
        for c in controllers:
            if c=='rl11_lines' or (m,c) not in assets or baseline is None:continue
            data=assets[m,c];first=next(r for r in rows if r['map']==m and r['controller']=='rl11_lines')
            second=next(r for r in rows if r['map']==m and r['controller']==c)
            # NPZ compressed hashes may differ; compare the actual arrays.
            with np.load(Path(first['attempt'])/'mapper_legal_inputs.npz') as aa,np.load(Path(second['attempt'])/'mapper_legal_inputs.npz') as bb:
                same=set(aa.files)==set(bb.files) and all(np.array_equal(aa[k],bb[k],equal_nan=True) for k in aa.files)
            equality.append(dict(map=m,controller=c,legal_inputs_equal=same,
                trajectory_equal=bool(np.array_equal(baseline['trajectories'],data['trajectories'])),
                bombs_equal=baseline['bombs']==data['bombs']))
    for candidate in controllers:
        if candidate=='rl11_lines':continue
        doc+=['',f'## RL11 与 {NAMES[candidate]}','']
        for kind,label in [('exploration','实际雷达探索范围估计'),('mapping','完整提交建图'),('trajectories','完整实际轨迹'),('walls','官方墙体覆盖'),('bombs','实际炸弹位置')]:
            fig,axes=base.plt.subplots(len(maps),2,figsize=(18,7*len(maps)),squeeze=False)
            fig.subplots_adjust(left=.015,right=.985,top=.88,bottom=.065,wspace=.06,hspace=.44)
            for y,m in enumerate(maps):
                for x,c in enumerate(['rl11_lines',candidate]):
                    matching=[r for r in rows if r['map']==m and r['controller']==c]
                    if matching:
                        data=assets[m,c]
                        panels.panel(axes[y,x],'mapping' if kind=='exploration' else kind,matching[0],data)
                        if kind=='exploration':
                            pixels=np.ones((*data['truth'].shape,3))*.94
                            pixels[data['visible']]=[.57,.85,.73];pixels[data['truth'].astype(bool)]=[.12,.12,.12]
                            axes[y,x].imshow(pixels);base.areas(axes[y,x],data['spec'])
                    else:axes[y,x].axis('off');axes[y,x].set_title('无可用完整产物',fontproperties=lines.FONT)
            fig.suptitle('RL11 / '+NAMES[candidate]+'：'+label,fontproperties=lines.FONT,fontsize=22,y=.985)
            note={'exploration':'绿：雷达可见区采样估计（每8步/4px/射线间插值）；浅灰：未覆盖估计；黑：真墙。',
                  'mapping':'绿：命中；蓝：漏墙；深红：收费误画；黄：免罚误画。',
                  'trajectories':'轨迹不是雷达覆盖；灰：无GPS/罗盘；黄：无通信；颜色是录像槽位。',
                  'walls':'绿：覆盖内墙；蓝：覆盖边界墙；灰：未覆盖；红：爆炸范围。',
                  'bombs':'编号是实际落点产物顺序；圆为爆炸半径，不表示实际墙面重复程度。'}[kind]
            fig.text(.5,.025,note+' 每图每版一局，蓝队难度与完整总分未评测。',ha='center',fontproperties=lines.FONT,fontsize=13)
            filename=f'{candidate}_{kind}.png';base.save(fig,a.output/filename)
            doc+=['',f'![{label}]({(a.output/filename).resolve().as_posix()})']
    doc+=['','## 每组身份、诊断与回放','']
    for r in rows:
        run=Path(r['attempt']).resolve()
        doc+=[f"### {r['map']} / {NAMES[r['controller']]} / r{r['round']} / seed {r['red_seed']}",'',
            f"区域：{r['actual_conditions']}；有效：{r['red_valid']}；灰区真墙面命中 {r['gray_surface_hits']}/{r['gray_true_surface']}，灰区假墙像素 {r['gray_false_pixels']}。",
            f"每40步实际雷达端点采样累计墙面 {r['gray_sampled_observed']}，是观察下界；不能冒充完整传感器覆盖。",'',
            f"每8步/4px雷达可见区估计覆盖灰区自由像素 {r['gray_sampled_visible_free_pixels']}/{r['gray_free_pixels']}（{100*r['gray_sampled_visible_free_fraction']:.2f}%）；含射线间插值，非精确连续覆盖率。",'',
            '|来源|已收到版本|年龄步|最新本机格|已收到格|本机图未送达格|','|---|---:|---:|---:|---:|---:|']
        for s in r['source_snapshots']:
            doc.append(f"|{s['source']}|{s['sequence']}|{s['age']}|{s['own_latest_cells']}|{s['received_cells']}|{s['own_final_absent_from_source_snapshot']}|")
        doc+=['','```powershell',"Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",f".\\replay_map.cmd '{run}'",'```','']
    doc+=['','本批没有蓝队录像。所有失败和审计异常均保留在 summary.json，不因退出码为0自动判有效。',
          f'原始完整图册：[批次报告]({(a.batch/"report.md").resolve().as_posix()})。']
    with (a.output/'summary.json').open('x',encoding='utf-8') as f:
        json.dump(dict(rows=rows,failures=failed,input_comparisons=equality,plan_sha256=ph,blue_phases=0),f,indent=2,ensure_ascii=False)
    with (a.output/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(doc)+'\n')
    print(json.dumps(dict(rows=[dict(map=r['map'],controller=r['controller'],M=r['map_score'],valid=r['red_valid']) for r in rows],failures=failed,equality=equality),ensure_ascii=True))


if __name__=='__main__':main()
