"""RL16-B: bounded local outage blocks, graded validation, source-aware fusion.

RL11 legal pose/line matching and navigation are reused. No truth/map identity.
Heuristic thresholds are not probabilities or a claim of globally correct SLAM.
"""
from collections import Counter,deque
from types import SimpleNamespace
import math
import numpy as np
from swarm_rescue.solutions.my_drone_red_lines11 import MyDroneRedLines11,WallLineMapper,fit_lines,line_angle
from swarm_rescue.solutions.my_drone_red_mapping import MappingSnapshots,anchor_frames,free_points
from swarm_rescue.solutions.my_drone_red_advanced2 import _M01_SnapshotStore,_M08_MyDroneRedV4
from swarm_rescue.solutions.my_drone_red_stopturn import available_pose
from swarm_rescue.solutions.semifinal_localization import transform,wrap


def overlap_check(lines,references,distance=8.,angle=.065):
    """No global search: finite overlap, unique correspondence, geometry rank.

    Repeated parallel corridor walls cannot prove along-corridor alignment.
    Current/peer lines are never added to the reference before this check.
    """
    if not len(lines) or not len(references):return dict(matches=0,rank=0,ambiguous=0,residual=None)
    refs=np.asarray(references,float);rv=refs[:,1]-refs[:,0];rl=np.linalg.norm(rv,axis=1)
    valid=rl>35;refs=refs[valid];rv=rv[valid];rl=rl[valid]
    if not len(refs):return dict(matches=0,rank=0,ambiguous=0,residual=None)
    t=rv/rl[:,None];n=np.column_stack((-t[:,1],t[:,0]));rc=refs.mean(1);ra=np.arctan2(t[:,1],t[:,0])
    selected={};ambiguous=0
    for line in lines:
        v=line[1]-line[0];length=float(np.linalg.norm(v))
        if length<35:continue
        da=np.abs(line_angle(ra-math.atan2(v[1],v[0])))
        delta=line.mean(0)-rc;normal=np.abs((delta*n).sum(1));along=np.abs((delta*t).sum(1))
        good=(normal<distance)&(da<angle)&(along<(rl+length)/2-20)
        score=np.where(good,normal+50*da,np.inf);order=np.argsort(score)
        if not np.isfinite(score[order[0]]):continue
        if len(order)>1 and score[order[1]]-score[order[0]]<2:
            ambiguous+=1;continue
        k=int(order[0])
        if k not in selected or score[k]<selected[k]:selected[k]=float(score[k])
    if not selected:return dict(matches=0,rank=0,ambiguous=ambiguous,residual=None)
    values=np.linalg.eigvalsh(n[list(selected)].T@n[list(selected)])
    rank=2 if values[0]>.12*values.sum() else 1
    return dict(matches=len(selected),rank=rank,ambiguous=ambiguous,residual=float(np.mean(list(selected.values()))))


class StagedMapper(WallLineMapper):
    BLOCK_SCANS=24
    MAX_BLOCKS=24
    MAX_AGE=720
    def __init__(self,size,identifier=0):
        super().__init__(size);self.identifier=identifier
        self.blocks=[];self.open_block=None;self.episode=None;self.serial=0;self.block_serial=0
        self.good_count=0;self.last_trusted=None;self.recovery=None;self.progress=0.
        self.validated=np.zeros(self.shape,bool);self.staged=np.zeros(self.shape,bool)
        self.free_support=np.zeros(self.shape,np.uint8);self.witnesses={}
        self.stage_stats=Counter();self.stage_events=deque(maxlen=1000);self.version=0
        self.absolute_witness=None;self._finalized=False

    def note(self,event,**kw):
        self.stage_stats[event]+=1;self.stage_events.append(dict(step=self.step,event=event,**kw))

    def new_block(self):
        self.block_serial+=1;e=self.episode
        self.open_block=dict(id=self.block_serial,episode=e['id'],origin=e['position'].copy(),heading=e['heading'],
            start=e['step'],references=[x.copy() for x in e['references']],frames=[],lines=[],
            matched=0,match_checks=0,distance=0.,last_position=None,recovery=None)

    def global_frames(self,block,recovered=True):
        result=[(t,points,transform(np.asarray([pos]),block['origin'],block['heading'])[0],
            wrap(angle+block['heading']),weight,free) for t,points,pos,angle,weight,free in block['frames']]
        r=block.get('recovery')
        if recovered and r is not None:
            result=anchor_frames(result,(block['start'],block['origin']),r['pred_position'],r['pred_heading'],
                r['position'],r['heading'],r['step'])
        return result

    def world_lines(self,block):
        return [transform(line,block['origin'],block['heading']) for line in block['lines'][-24:]]

    def render(self,block,recovered=True,threshold=3.):
        grid=np.zeros(self.shape,np.float32)
        for _,points,pos,angle,_,free in self.global_frames(block,recovered):
            self._erase(grid,transform(free,pos,angle),1.)
            self._insert(grid,transform(points,pos,angle),1.)
        return grid>=threshold

    def seal(self):
        b=self.open_block
        if b is None or not b['frames']:return
        b['end']=b['frames'][-1][0];self.blocks.append(b);self.open_block=None
        raw=self.render(b,False,2.4);self.staged|=raw
        self.note('block_staged',block=b['id'],scans=len(b['frames']),cells=int(raw.sum()))
        self.stage_stats['maximum_blocks']=max(self.stage_stats['maximum_blocks'],len(self.blocks))
        self.stage_stats['maximum_frames']=max(self.stage_stats['maximum_frames'],sum(len(x['frames']) for x in self.blocks))

    def accept_witness(self,packet,now,number):
        if not isinstance(packet,tuple) or len(packet)!=7:return False
        source,seq,origin,parents,sigma_p,sigma_a,raw=packet
        if (type(sigma_p) not in (int,float) or type(sigma_a) not in (int,float)
                or not 0<=sigma_p<=8 or not 0<=sigma_a<=.10):
            self.stage_stats['witness_invalid_or_correlated']+=1;return False
        if (type(source)is not int or source==self.identifier or not 0<=source<number or type(seq)is not int
            or not 0<=now-seq<=240 or origin!='absolute_local' or parents!=(source,)
            or not np.isfinite([sigma_p,sigma_a]).all() or sigma_p>8 or sigma_a>.10
            or not isinstance(raw,tuple) or not 2<=len(raw)<=12):
            self.stage_stats['witness_invalid_or_correlated']+=1;return False
        previous=self.witnesses.get(source)
        if previous and seq<=previous['step']:
            self.stage_stats['witness_duplicate']+=1;return False
        try:segments=np.asarray(raw,float)
        except (ValueError,TypeError):return False
        if segments.shape!=(len(raw),2,2) or not np.isfinite(segments).all():return False
        self.witnesses[source]=dict(step=seq,lines=segments,parents=parents)
        self.note('witness_received',source=source,sequence=seq);return True

    def grade(self,b,deadline=False):
        n=len(b['frames']);ratio=b['matched']/max(1,b['match_checks'])
        local_age=b['end']-b['start'];travel=float(b['distance'])
        sigma=math.sqrt(25.+.04*local_age+(travel*math.radians(4))**2)
        check=overlap_check(self.world_lines(b),b['references'])
        info=dict(scans=n,match_ratio=ratio,sigma_position=sigma,entry=check)
        if n<6:return None,info
        r=b.get('recovery')
        # A stable independent absolute recovery is stronger than self matches.
        # Long/nonuniform drift and gross anchor disagreement remain deferred.
        if r is not None and r['stable'] and local_age<=600 and ratio>=.35:
            info['recovery_distance']=r['distance_error'];info['recovery_angle']=r['angle_error']
            if r['distance_error']<=70 and abs(r['angle_error'])<=.25:
                return 'absolute_recovery',info
        # A peer is only a supporting independent absolute observation. It
        # cannot replace a plausible own entry anchor, nor prove parallel axes.
        if sigma<=18 and ratio>=.6 and check['matches']>=1 and check['ambiguous']==0:
            for source,w in self.witnesses.items():
                if self.step-w['step']>240:continue
                peer=overlap_check(self.world_lines(b),w['lines'])
                if peer['matches']>=2 and peer['rank']==2 and peer['ambiguous']==0 and peer['residual']<=5:
                    info['peer']=dict(source=source,step=w['step'],check=peer)
                    return 'entry_plus_independent_peer',info
        # No second aircraft is required for a short, well constrained segment.
        if local_age<=120 and sigma<=12 and ratio>=.65 and check['matches']>=1 and check['ambiguous']==0 and check['residual']<=4:
            return 'short_entry_supported',info
        if deadline and sigma<=18 and ratio>=.75 and check['rank']==2 and check['matches']>=2 and check['ambiguous']==0:
            return 'deadline_high_repeat_subset',info
        return None,info

    def validate(self,deadline=False,capacity=False):
        keep=[]
        for index,b in enumerate(self.blocks):
            expired=self.step-b['end']>self.MAX_AGE
            grade,info=self.grade(b,deadline or expired or capacity)
            if grade is not None:
                mask=self.render(b,True,6. if grade=='deadline_high_repeat_subset' else 3.)
                # Confirmed free evidence vetoes tentative wall insertion only;
                # old trusted hits are neither erased nor globally translated.
                conflict=mask&(self.free_support>=4)&(self.hits<1.)
                mask&=~conflict;self.validated|=mask;self.version+=1
                self.note('block_validated',block=b['id'],grade=grade,cells=int(mask.sum()),conflict_cells=int(conflict.sum()),evidence=info)
            elif deadline or expired or (capacity and index==0):
                mask=self.render(b,False,2.4)
                self.note('block_not_submitted',block=b['id'],cells=int(mask.sum()),
                    reason='deadline' if deadline else ('age' if expired else 'capacity'),evidence=info)
            else:keep.append(b)
        self.blocks=keep

    def update(self,gps,compass,odom,d,a,objects=()):
        good=all(available_pose(gps,compass));self.good_count=self.good_count+1 if good else 0
        previous=self.position.copy() if self.position is not None else None;old_heading=self.heading
        if not good and self.missing==0 and previous is not None:
            self.serial+=1;self.recovery=None
            trusted=self.last_trusted
            self.episode=dict(id=self.serial,step=self.step,position=previous,heading=old_heading,
                references=[] if trusted is None else [x.copy() for x in trusted['lines']],distance=0.)
            self.note('outage_started',episode=self.serial,entry_position=previous.tolist(),entry_heading=old_heading,
                reference_count=len(self.episode['references']),position_sigma=5.,heading_sigma=math.radians(4))
        if good and self.missing and previous is not None and self.episode is not None:
            end=previous.copy();angle=old_heading
            if odom is not None and np.shape(odom)==(3,) and np.isfinite(odom).all():
                length,alpha,theta=map(float,odom);end+=length*np.array([math.cos(angle+alpha),math.sin(angle+alpha)]);angle=wrap(angle+theta)
            self.seal();self.recovery=dict(episode=self.episode['id'],pred_position=end,pred_heading=angle,
                step=self.step+1,relative=np.zeros(2),relative_heading=0.,positions=[],headings=[])
            self.episode=None
        # Parent still performs exactly RL11's odometry/scan matching. Its outage
        # pending list is drained below, so it cannot silently publish/re-anchor
        # those observations into its trusted hits at the first recovered frame.
        super().update(gps,compass,odom,d,a,objects)
        if not good and self.episode is not None and odom is not None and np.shape(odom)==(3,) and np.isfinite(odom).all():
            self.episode['distance']+=abs(float(odom[0]))
        if not good and self.pending:
            if self.episode is not None:
                if self.open_block is None:self.new_block()
                b=self.open_block
                for t,points,pos,heading,weight,free in self.pending:
                    relative=transform(np.asarray([pos-b['origin']]),np.zeros(2),-b['heading'])[0]
                    b['frames'].append((t,points.copy(),relative,wrap(heading-b['heading']),weight,free.copy()))
                    b['distance']=float(self.episode['distance'])
                    b['last_position']=pos.copy()
                    if t%8==0:
                        b['match_checks']+=1;b['matched']+=int(self.last_match==t)
                        for line in self.matcher.lines[:6]:b['lines'].append(transform(line,relative,wrap(heading-b['heading'])))
                        b['lines']=b['lines'][-48:]
                if len(b['frames'])>=self.BLOCK_SCANS:self.seal()
            self.pending.clear()
        if good and self.step%4==0:
            pts=transform(free_points(d,a),self.position,self.heading);indices=self._indices(pts)
            self.free_support.ravel()[indices]=np.minimum(8,self.free_support.ravel()[indices]+1)
            self.free_support[self.hits>=1.]=0
            if self.good_count>=8:
                segments=[transform(v,self.position,self.heading) for v in fit_lines(d,a,objects)[:12]]
                self.last_trusted=dict(step=self.step,lines=segments)
                self.absolute_witness=(self.identifier,self.step,'absolute_local',(self.identifier,),5.,math.radians(4),
                    tuple(tuple(tuple(float(x) for x in point) for point in line) for line in segments))
        if good and self.recovery is not None:
            r=self.recovery
            if self.step>r['step'] and odom is not None and np.shape(odom)==(3,) and np.isfinite(odom).all():
                length,alpha,theta=map(float,odom)
                r['relative']+=length*np.array([math.cos(r['relative_heading']+alpha),math.sin(r['relative_heading']+alpha)])
                r['relative_heading']=wrap(r['relative_heading']+theta)
            anchor_heading=wrap(float(compass)-r['relative_heading'])
            anchor=np.asarray(gps)-transform(np.asarray([r['relative']]),np.zeros(2),anchor_heading)[0]
            r['positions'].append(anchor);r['headings'].append(anchor_heading)
            if len(r['positions'])>=8:
                position=np.median(r['positions'],axis=0);center=r['headings'][0]
                headings=np.array([wrap(v-center) for v in r['headings']]);heading=wrap(center+float(np.median(headings)))
                stable=bool(max(np.linalg.norm(p-position) for p in r['positions'])<=8 and np.ptp(headings)<=.10)
                certificate=dict(step=r['step'],position=position,heading=heading,stable=stable,
                    pred_position=r['pred_position'],pred_heading=r['pred_heading'],
                    distance_error=float(np.linalg.norm(position-r['pred_position'])),angle_error=wrap(heading-r['pred_heading']))
                for b in self.blocks:
                    if b['episode']==r['episode']:b['recovery']=certificate
                self.note('recovery_checked',stable=stable,position_spread=float(max(np.linalg.norm(p-position) for p in r['positions'])),
                    distance_error=certificate['distance_error'],angle_error=certificate['angle_error'])
                self.recovery=None;self.validate()
        if self.step%48==0:self.validate()
        if len(self.blocks)>self.MAX_BLOCKS:self.validate(capacity=True)

    def set_progress(self,progress):
        self.progress=float(progress)
        if progress>=.78 and not self._finalized:
            self.seal();self.validate(deadline=True);self._finalized=True

    def mask(self):return (self.hits>=2.4)|self.validated


class ValidatedSnapshots(MappingSnapshots):
    """Keep source identity; veto tentative walls using received reliable free.

    Trusted baseline masks retain RL11 semantics; only newly validated outage
    masks are conflict-gated. This is not a universal inter-robot SLAM merger.
    """
    def __init__(self,shape,number,mapper):
        super().__init__(shape,number,mapper);self.trusted={};self.tentative={};self.frees={};self.conflict_cells={}
    def publish(self,source,sequence,mask):
        parts=[mask,self.mapper.hits>=2.4,self.mapper.validated,(self.mapper.free_support>=4)&(self.mapper.hits<1.)]
        payload=b''.join(np.packbits(np.asarray(p,np.uint8).ravel()).tobytes() for p in parts)
        self.accept((int(source),int(sequence),payload))
    def accept(self,snapshot):
        if not isinstance(snapshot,tuple) or len(snapshot)!=3:return False
        source,sequence,payload=snapshot;b=self.bytes_per_mask
        if not isinstance(payload,bytes) or len(payload)!=4*b:return False
        if not _M01_SnapshotStore.accept(self,(source,sequence,payload[:b])):return False
        self.snapshots[source]=snapshot
        masks=[np.unpackbits(np.frombuffer(payload[i*b:(i+1)*b],np.uint8))[:math.prod(self.shape)].reshape(self.shape).astype(bool) for i in (1,2,3)]
        self.trusted[source],self.tentative[source],self.frees[source]=masks
        self.mapping_masks[source]=masks[0]|masks[1]
        return True
    def submission(self):
        result=np.zeros(self.shape,bool);self.conflict_cells={}
        for source,trusted in self.trusted.items():
            veto=np.zeros(self.shape,bool)
            for other,free in self.frees.items():
                if other!=source:veto|=free
            tentative=self.tentative[source];self.conflict_cells[source]=int((tentative&veto).sum())
            result|=trusted|(tentative&~veto)
        return result


class MyDroneRedStaged16(MyDroneRedLines11):
    STAGED_SUBMISSION=True
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if self.STAGED_SUBMISSION:
            self._submission_mapper=StagedMapper(self.size_area,int(self.identifier or 0))
            self._radio_maps=ValidatedSnapshots(self._map.shape,self._n,self._submission_mapper)

    def _update_observation(self):
        super()._update_observation()
        if self.STAGED_SUBMISSION:self._submission_mapper.set_progress(self._progress())

    def define_message_for_all(self):
        original=super().define_message_for_all()
        if not self.STAGED_SUBMISSION:return original
        m=self._submission_mapper;w=m.absolute_witness
        if w is not None and m.step-w[1]>120:w=None
        return ('red-staged16',original,w)

    def _receive(self):
        if not self.STAGED_SUBMISSION:return super()._receive()
        stripped=[]
        for sender,message in self.communicator.received_messages:
            if isinstance(message,tuple) and len(message)==3 and message[0]=='red-staged16':
                _,original,witness=message
                if (isinstance(witness,tuple) and len(witness)==7 and isinstance(original,tuple) and len(original)==4
                    and original[0]=='red-v4' and isinstance(original[2],tuple) and len(original[2])>0 and witness[0]==original[2][0]):
                    self._submission_mapper.accept_witness(witness,self.elapsed_timestep,self._n)
                message=original
            stripped.append((sender,message))
        proxy=SimpleNamespace(_n=self._n,elapsed_timestep=self.elapsed_timestep,_status=self._status,
            _peers=self._peers,_radio_maps=self._radio_maps,_explore=self._explore,
            _valid_peer=self._valid_peer,_bomb_claims=self._bomb_claims,_bomb_records=self._bomb_records,
            communicator=SimpleNamespace(received_messages=stripped))
        _M08_MyDroneRedV4._receive(proxy)

    def _submit(self):
        if self.STAGED_SUBMISSION:
            self._submission_mapper.seal();self._submission_mapper.validate(deadline=True)
        return super()._submit()


class MyDroneRedStaged16Off(MyDroneRedStaged16):STAGED_SUBMISSION=False
