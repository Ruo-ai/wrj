"""RL17 same-input offline replay: fixed actual submitted source versions.

Truth is read only for post-update metrics. Full pose traces are retained;
12-step aligned legacy audit samples supply error metrics without inventing
an unverified replay-slot-to-drone association.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from swarm_rescue.solutions.my_drone_red_lines11 import WallLineMapper
from swarm_rescue.solutions.my_drone_red_p2l17 import PointLineMapper17
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base


def write(path, obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)


def summary(values):
    a=np.asarray(values)
    return dict(count=len(a),mean=float(a.mean()) if len(a) else None,
                p95=float(np.percentile(a,95)) if len(a) else None,
                maximum=float(a.max()) if len(a) else None)


class ReplayProcess:
    """Stdlib-only affinity and process peak working set (no new dependency)."""
    def __init__(self):
        if os.name=='nt':
            import ctypes as c
            self.c=c;self.k=c.WinDLL('kernel32',use_last_error=True)
            self.k.GetCurrentProcess.restype=c.c_void_p
            self.handle=self.k.GetCurrentProcess()
            self.k.GetProcessAffinityMask.argtypes=[c.c_void_p,c.c_void_p,c.c_void_p]
            self.k.SetProcessAffinityMask.argtypes=[c.c_void_p,c.c_size_t]
            allowed,system=c.c_size_t(),c.c_size_t()
            if not self.k.GetProcessAffinityMask(self.handle,c.byref(allowed),c.byref(system)):raise c.WinError()
            mask=allowed.value&-allowed.value
            if not self.k.SetProcessAffinityMask(self.handle,mask):raise c.WinError()
            checked=c.c_size_t()
            if not self.k.GetProcessAffinityMask(self.handle,c.byref(checked),c.byref(system)):raise c.WinError()
            assert checked.value==mask
            self.affinity=[mask.bit_length()-1]
            class Memory(c.Structure):
                _fields_=[('cb',c.c_ulong),('faults',c.c_ulong)]+[(n,c.c_size_t) for n in (
                    'peak_rss','rss','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
            self.Memory=Memory
            self.k.K32GetProcessMemoryInfo.argtypes=[c.c_void_p,c.c_void_p,c.c_ulong]
        else:
            os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});self.affinity=list(os.sched_getaffinity(0))

    def memory(self):
        if os.name=='nt':
            m=self.Memory();m.cb=self.c.sizeof(m)
            if not self.k.K32GetProcessMemoryInfo(self.handle,self.c.byref(m),m.cb):raise self.c.WinError()
            return int(m.peak_rss)
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


def replay(run,out,repair=False):
    out.mkdir(parents=True,exist_ok=False)
    process=ReplayProcess()
    cv2.setNumThreads(1)
    meta=base.read(run/'line_diagnostics.json');diag=base.read(run/'exploration_diagnostics.json')
    motion=base.read(run/'motion_diagnostics.json');result=base.read(run/'result.json')
    assert meta['mapper_inputs_complete'] and not meta['errors'] and not diag['errors']
    with np.load(run/'mapper_legal_inputs.npz') as f:inputs={k:f[k] for k in f.files}
    with np.load(run/'exploration_stages.npz') as f:stages={k:f[k] for k in f.files}
    kinds={v:k for k,v in meta['kinds'].items()};objects={}
    for owner,tick,kind,distance,angle in inputs['semantic']:
        objects.setdefault((int(owner),int(tick)),[]).append(SimpleNamespace(
            entity_type=SimpleNamespace(name=kinds[int(kind)]),distance=float(distance),angle=float(angle)))
    submission=diag['submissions'][0];h,w=result['expected_shape']
    actual=np.asarray(base.read(next((run/'artifacts').glob('*_exploration.json')))['grid'],np.uint8)
    png=np.asarray(Image.open(next((run/'artifacts').glob('*_exploration_diff.png'))).convert('RGB'))
    truth=np.all(png[base.COMPARISON_HEADER_HEIGHT:base.COMPARISON_HEADER_HEIGHT+h,:w]==40,axis=2).astype(np.uint8)
    scorer=base.MapExplorationScorer();surface,_=scorer.split_regions(truth)
    distance=scorer.distance_field(truth)
    np.save(out/'truth.npy',truth)
    records=[]
    from swarm_rescue.solutions.my_drone_red_repair17 import RepairMapper17
    candidates=[('rl11_lines',WallLineMapper),('rl17_repair',RepairMapper17)] if repair else [('rl11_lines',WallLineMapper),('rl17_p2l',PointLineMapper17)]
    for name,cls in candidates:
        started=time.perf_counter();cpu=time.process_time();sources={};exact={};received={};stats={};arrays={}
        errors_p=[];errors_h=[];latencies=[];peak_pending=0;max_cache_bytes=0
        counts=Counter();peak_rss=process.memory()
        with (out/f'{name}_matches.jsonl').open('x',encoding='utf-8') as events:
            for ident in diag['drones']:
                mapper=cls((w,h));data=inputs['drone_'+ident];angles=inputs['angles_'+ident]
                assert np.array_equal(data[:,0],np.arange(1,len(data)+1))
                samples={r['tick']:r for r in motion['drones'][ident]['samples']}
                poses=[]
                for row in data:
                    tick=int(row[0]);gps=row[5:7].copy() if row[1] else None
                    compass=float(row[7]) if row[2] else None;odom=row[8:11].copy() if row[3] else None
                    lidar=row[11:].copy() if row[4] else None
                    before=time.perf_counter()
                    mapper.update(gps,compass,odom,lidar,angles,objects.get((int(ident),tick),()))
                    latencies.append(time.perf_counter()-before)
                    poses.append([*(mapper.position if mapper.position is not None else [np.nan,np.nan]),mapper.heading])
                    peak_pending=max(peak_pending,len(mapper.pending))
                    if tick%8==0 and mapper.missing:
                        info=dict(getattr(mapper.matcher,'last_info',{}))
                        counts['rl11_line_'+info.get('reason','none')]+=1
                        extra=getattr(mapper.matcher,'last_p2l',{})
                        events.write(json.dumps(dict(drone=ident,step=tick,rl11_or_accepted=info,p2l=extra))+'\n')
                    if tick%120==0:
                        peak_rss=max(peak_rss,process.memory())
                        memory=sum(f[1].nbytes+f[2].nbytes+f[5].nbytes for f in mapper.pending)
                        max_cache_bytes=max(max_cache_bytes,memory)
                    if tick in samples and (not row[1] or not row[2]):
                        s=samples[tick]
                        errors_p.append(float(np.linalg.norm(mapper.position-s['truth'])))
                        errors_h.append(abs((mapper.heading-s['true_heading']+np.pi)%(2*np.pi)-np.pi)*180/np.pi)
                        if name=='rl11_lines':
                            assert np.allclose(mapper.position,s['mapper']['position'],atol=1e-7)
                    if tick==submission['sources'].get(ident):sources[ident]=mapper.mask().copy()
                arrays['poses_'+ident]=np.array(poses)
                arrays['local_'+ident]=mapper.mask().copy()
                exact[ident]=bool(np.array_equal(arrays['local_'+ident],stages['local_final_'+ident]))
                if ident in sources:
                    arrays['source_'+ident]=sources[ident]
                    received[ident]=bool(np.array_equal(sources[ident],stages[f"submitter_{submission['drone']}_source_{ident}"]))
                stats[ident]=dict(mapper=mapper.stats,line=dict(mapper.matcher.stats),
                    p2l=dict(getattr(mapper.matcher,'p2l_stats',{})),p2l_seconds=getattr(mapper.matcher,'p2l_seconds',0.))
        merged=np.zeros(mapper.shape,bool)
        for mask in sources.values():merged|=mask
        assert len(sources)==len(submission['sources'])
        pixels=mapper.to_pixels(merged);np.save(out/f'{name}_pixels.npy',pixels)
        np.savez_compressed(out/f'{name}_maps_poses.npz',**arrays)
        score=scorer.score(truth,pixels,result['elapsed_walltime'])
        record=dict(mapper=name,wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
            peak_rss_bytes=peak_rss,cpu_affinity=process.affinity,max_pending=peak_pending,
            max_pending_arrays_bytes_sampled=max_cache_bytes,update_seconds=summary(latencies),
            local_exact=exact,received_exact=received,final_exact=bool(np.array_equal(pixels,actual)),
            offline_map_score=score.s_map*100,credit=score.credit,penalty=score.penalty,
            wall_hits=int((surface.astype(bool)&pixels.astype(bool)).sum()),
            charged_pixels=int((pixels.astype(bool)&(distance>scorer.config.dead_zone)).sum()),
            outage_position_sampled=summary(errors_p),outage_heading_sampled_deg=summary(errors_h),stats=stats,
            match_reason_counts=dict(counts))
        records.append(record);write(out/f'{name}_metrics.json',record)
        print(json.dumps({k:v for k,v in record.items() if k not in ('stats','match_reason_counts')},ensure_ascii=False),flush=True)
    exact=records[0]['final_exact'] and all(records[0]['local_exact'].values()) and all(records[0]['received_exact'].values())
    write(out/'results.json',dict(records=records,exact_rl11_reproduction=exact,
        input_sha256=base.digest(run/'mapper_legal_inputs.npz'),source=str(run.resolve()),source_sequences=submission['sources'],
        candidate_sha256={n:base.digest(Path(__file__).parents[1]/n) for n in ('my_drone_red_p2l17.py','my_drone_red_repair17.py')},
        limitation='Same legal tape and actual received source versions. Offline formula only, no flight/radio execution. Pose errors: aligned every-12-step samples, not every step. RSS includes tape, scorer and diagnostic arrays.'))
    if not exact:raise RuntimeError('RL11 did not reproduce exactly; do not claim valid same-input result')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repair',action='store_true')
    a=p.parse_args();replay(a.attempt,a.output,a.repair)
