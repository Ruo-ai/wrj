"""OFFLINE ONLY: reachable, weighted marginal-wall greedy pressure layouts.

The online controllers never import this module or read its artifacts.
This finite-grid construction is neither a proven optimum nor a red result.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from .runner import REPO,read,write,file_hash
from .scenarios import training_rows,scenario,resolve_map
from ..dataset_manager.index import resolve_path,path_name

MAPS=['Map08','Map09','yh_map_fs_1','yh_map_fs_2']
DATASET=REPO/'output/optimization/datasets/dataset_001'


def greedy(walls,boundary,allowed,radius,count=10):
    """Integer centers, 16 px candidate stride, exact pixel-area residual."""
    h,w=walls.shape;yy,xx=np.indices((h,w))
    residual=walls.astype(np.float32)*np.where(boundary,.5,1.).astype(np.float32)
    original=residual.copy();union=np.zeros_like(walls,bool)
    candidates=allowed.copy();candidates &= (xx%16==8)&(yy%16==8)
    ky,kx=np.ogrid[-radius:radius+1,-radius:radius+1]
    kernel=(kx*kx+ky*ky<=radius*radius).astype(np.float32)
    placements=[];gains=[]
    for _ in range(count):
        if not candidates.any():raise ValueError('Not enough reachable candidate locations')
        gains_map=cv2.filter2D(residual,-1,kernel,borderType=cv2.BORDER_CONSTANT)
        rank=np.where(candidates,gains_map,-np.inf)
        row,col=np.unravel_index(np.argmax(rank),rank.shape)
        disk=(xx-col)**2+(yy-row)**2<=radius*radius
        gains.append(float(residual[disk].sum()));union|=disk
        residual[disk]=0
        placements.append(dict(x=float(col-w/2),y=float(h/2-row),theta=0.))
        candidates &= (xx-col)**2+(yy-row)**2>=32**2
    return placements,gains,float(original[union].sum()),union


def generate(output):
    from ...evaluate_red import configure_ray_backend
    from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
    from swarm_rescue.simulation.elements.sensor_disablers import ZoneType
    from swarm_rescue.simulation.reporting.place_score_manager import walls_and_boundary_grid,PlaceScoreManager
    configure_ray_backend('cpu');cv2.setNumThreads(1)
    if output.exists():raise ValueError('Oracle output already exists; preserve it and use a new directory')
    output.mkdir(parents=True)
    rows=training_rows(MAPS,DATASET);records=[]
    for row in rows:
        spec=scenario(resolve_path(row['map_path']),row.get('official_map'))
        request=dict(map_file=str(resolve_path(row['map_path'])),official_map=row.get('official_map'))
        cls=resolve_map(request)
        world=cls(DroneMotionless,tuple(ZoneType[z] for z in spec['zones']),5)
        try:
            walls,boundary=walls_and_boundary_grid(world.playground)
            h,w=walls.shape
            # 22 px wall clearance for an empty drone. Carrying feasibility and
            # blue difficulty must still be measured, not assumed by this mask.
            clearance=cv2.distanceTransform(1-walls,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            allowed=(clearance>=22).astype(np.uint8)
            _,components=cv2.connectedComponents(allowed,connectivity=4)
            sx,sy=map(int,spec['spawn']);component=components[sy,sx]
            if component==0:raise ValueError('Spawn has no clearance-feasible component')
            allowed=(components==component)
            doc=read(resolve_path(row['map_path']))
            # Editor polygons are offline geometry, never online parameters.
            for element in doc['elements']:
                if element['type'] in ('disposal_center','start_area'):
                    l,t=element['top_left'];r,b=element['bottom_right'];pad=32
                    allowed[max(0,t-pad):min(h,b+pad),max(0,l-pad):min(w,r+pad)]=False
            radius=round(.12*min(w,h))
            bombs,gains,weighted,union=greedy(walls,boundary,allowed,radius)
            exact=PlaceScoreManager.wall_destruction_masks(walls,[[b['x'],b['y']] for b in bombs],radius,boundary)[2]
            verified=float(np.where(boundary,.5,1.)[exact.astype(bool)].sum())
            if abs(weighted-verified)>.01:raise ValueError('Exact official union disagrees with greedy area')
            world.clear_bombs();world.spawn_bombs(bombs)
            if world._number_bombs!=10:raise ValueError('Native spawning rejected oracle layout')
            layout=output/(row['name']+'_bombs.json')
            write(layout,dict(map_name=cls.__name__,zones_config=spec['zones'],bombs=bombs))
            record=dict(name=row['name'],map=cls.__name__,map_row=row,scenario=spec,
                layout=path_name(layout),layout_sha256=file_hash(layout),marginal_weighted_area=gains,
                total_weighted_area=weighted,weighted_coverage=weighted/float(np.where(boundary,.5,1.)[walls.astype(bool)].sum()),
                actual_count=10,blast_radius=radius,method='reachable-22px/stride-16px/marginal-wall-greedy-v1',
                scope='offline_pressure_layout; not online red result; no optimality or difficulty proof')
            plot(output/(row['name']+'.png'),walls,exact,bombs,spec,record)
            records.append(record)
            print(row['name'],weighted,gains,flush=True)
        finally:world.playground.window.close()
    write(output/'manifest.json',dict(maps=records,tool_sha256=file_hash(__file__),
        source='original implementation; Nemhauser/Wolsey/Fisher 1978 idea; no third-party code copied'))
    lines=['# 离线上帝视角强布局候选','',
        '真实墙几何仅用于离线构造。没有红队在线探索/布弹过程，不产生红队成绩；不是已证明的最优或最难布局。',
        '22像素墙间距、出生连通分量、16像素候选间隔、10弹逐次新增加权墙面积贪心；炸弹间距32像素。携弹可行性及蓝队难度待测。','']
    for r in records:
        lines += [f"## {r['name']}",'',f"真实加权覆盖面积 {r['total_weighted_area']:.1f}；覆盖比例 {100*r['weighted_coverage']:.2f}%（不是官方归一墙体分）。",
            f"实际区域：{r['scenario']['zones']}。原生装载10弹通过；未运行蓝队正式回合。",
            f"逐弹新增面积：{r['marginal_weighted_area']}。",'',f"![离线强布局 {r['name']}](<{(output/(r['name']+'.png')).as_posix()}>)",'']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return records


def plot(path,walls,destroyed,bombs,spec,record):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Rectangle
    h,w=walls.shape
    rgb=np.ones((h,w,3));rgb[walls.astype(bool)]=(.25,.25,.25);rgb[destroyed.astype(bool)]=(.9,.2,.15)
    fig,ax=plt.subplots(figsize=(11,7));ax.imshow(rgb)
    for i,b in enumerate(bombs,1):
        x,y=b['x']+w/2,h/2-b['y'];ax.plot(x,y,'bo',ms=4)
        ax.text(x,y,str(i),fontsize=8);ax.add_patch(Circle((x,y),record['blast_radius'],fill=False,color='royalblue',alpha=.35))
    ax.plot(*spec['spawn'],'g*',ms=14,label='Spawn')
    doc=read(resolve_path(record['map_row']['map_path']))
    for e in doc['elements']:
        if e['type'] in ('no_gps_zone','no_com_zone','disposal_center'):
            l,t=e['top_left'];r,b=e['bottom_right']
            color={'no_gps_zone':'gray','no_com_zone':'gold','disposal_center':'green'}[e['type']]
            ax.add_patch(Rectangle((l,t),r-l,b-t,color=color,alpha=.18,label=e['type']))
    ax.set(title=record['name']+' / OFFLINE greedy wall layout (not a red score)',xlim=(0,w),ylim=(h,0))
    ax.legend(loc='upper right',fontsize=7);fig.tight_layout();fig.savefig(path,dpi=110);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    generate(parser.parse_args().output.resolve())
