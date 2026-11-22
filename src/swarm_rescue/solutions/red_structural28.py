"""Observation-derived structural mapping and source-local evidence consolidation.

No map identity, seed, true geometry, true pose, or shared drone state is used.
The orientation prior is learned from coherent orthogonal observed line families;
unsupported geometry keeps the original RL17 evidence. Geometry is deliberately
conservative: one source's model can never erase another source's residual map.
"""
from collections import Counter, deque
import math

import cv2
import numpy as np

from swarm_rescue.solutions.my_drone_red_lines11 import fit_lines
from swarm_rescue.solutions.my_drone_red_repair17 import RepairMapper17
from swarm_rescue.solutions.red_fusion27 import Fusion27


def quarter_angle(angle):
    """Smallest signed orientation difference modulo a right angle."""
    return (float(angle)+math.pi/4) % (math.pi/2)-math.pi/4


def coherent_axis(lines, heading=0.):
    """Require two observed perpendicular families, never assume world axes."""
    if len(lines) < 2:
        return None
    vectors=np.asarray([line[1]-line[0] for line in lines],float)
    lengths=np.linalg.norm(vectors,axis=1)
    valid=lengths>=45.
    if valid.sum()<2:
        return None
    angles=np.arctan2(vectors[valid,1],vectors[valid,0])+float(heading)
    weights=np.minimum(lengths[valid],160.)
    z=np.sum(weights*np.exp(4j*angles))/weights.sum()
    if abs(z)<.96:
        return None
    axis=float(np.angle(z)/4)
    families=np.rint((angles-axis)/(math.pi/2)).astype(int)%2
    if len(set(families.tolist()))<2:
        return None
    return axis,float(abs(z))


class StructuralHeading28:
    """A causal absolute direction reference from stable legal sensor frames."""
    BOOTSTRAP_FRAMES=16
    MAX_TURN_RATE=.035
    TURN_SETTLE_STEPS=8
    MAX_CORRECTION=math.radians(12)

    def __init__(self):
        self.samples=deque(maxlen=96)
        self.axis=None
        self.settle=0
        self.stats=Counter()
        self.last_correction_applied=False
        self.angular_history=deque(maxlen=3)

    def motion(self,odometry):
        valid=(odometry is not None and np.shape(odometry)==(3,)
               and np.isfinite(odometry).all())
        rotation=float(odometry[2]) if valid else 0.
        self.angular_history.append(rotation)
        # Odometry angular increments contain noise even in a stationary body.
        # Do not turn one isolated noisy reading into a multi-frame blackout.
        confirmed=(len(self.angular_history)==3 and
                   abs(float(np.mean(self.angular_history)))>self.MAX_TURN_RATE)
        turning=valid and (abs(rotation)>=.08 or confirmed)
        if turning:
            self.settle=self.TURN_SETTLE_STEPS
            self.stats['turning_frames']+=1
        elif self.settle:
            self.settle-=1
        return bool(turning or self.settle)

    def observe(self,lines,compass):
        self.last_correction_applied=False
        if compass is None or np.ndim(compass)!=0 or not np.isfinite(compass) or self.settle:
            return compass
        info=coherent_axis(lines,float(compass))
        if info is None:
            self.stats['unsupported_frame']+=1
            return compass
        measured,concentration=info
        if self.axis is not None and abs(quarter_angle(measured-self.axis))>self.MAX_CORRECTION:
            self.stats['inconsistent_frame']+=1
            return compass
        self.samples.append(complex(np.exp(4j*measured)))
        average=np.mean(self.samples)
        if len(self.samples)<self.BOOTSTRAP_FRAMES or abs(average)<.94:
            self.stats['bootstrap_frames']+=1
            return compass
        self.axis=float(np.angle(average)/4)
        adjustment=quarter_angle(self.axis-measured)
        if abs(adjustment)>self.MAX_CORRECTION:
            self.stats['correction_bound']+=1
            return compass
        self.stats['corrected_frames']+=1
        self.stats['last_correction_milliradian']=int(round(adjustment*1000))
        self.stats['reference_milliradian']=int(round(self.axis*1000))
        self.stats['reference_frames']=len(self.samples)
        self.last_correction_applied=True
        return float(compass)+adjustment


class StructuralMapper28(RepairMapper17):
    """RL17 local mapper with a learned heading reference and immutable cutoff.

    Fast-turn scans are not inserted into the submission map. Odometry/GPS pose
    updates continue, and this does not touch the controller's navigation map.
    The correction is only observable heading; no invented corridor-axis shift.
    """
    def __init__(self,size):
        super().__init__(size)
        self.structure28=StructuralHeading28()
        self.final_mask=None
        self.final_mapper_step=None

    def update(self,gps,compass,odometry,distances,angles,objects=()):
        if self.final_mask is not None:
            self.stats['after_cutoff_observations_not_mapped']=self.stats.get('after_cutoff_observations_not_mapped',0)+1
            return
        unstable=self.structure28.motion(odometry)
        absolute=(gps is not None and np.shape(gps)==(2,) and np.isfinite(gps).all()
                  and compass is not None and np.ndim(compass)==0 and np.isfinite(compass))
        corrected=False
        if absolute and not unstable and (self.step+1)%4==0:
            compass=self.structure28.observe(fit_lines(distances,angles,objects),compass)
            corrected=self.structure28.last_correction_applied
            if corrected and self.position is not None and self.missing==0:
                rotation=(float(odometry[2]) if odometry is not None and np.shape(odometry)==(3,)
                          and np.isfinite(odometry).all() else 0.)
                # Apply the observable geometric heading to this scan, not
                # only to a low-gain compass filter after its walls were drawn.
                # On outage recovery RL17 must retain its integrated endpoint
                # heading until it has anchored the buffered scan trajectory.
                self.heading=(float(compass)-rotation+math.pi)%(2*math.pi)-math.pi
        if absolute and unstable:
            # Keep localization advancing while preventing a turn smear from
            # becoming permanently reinforced wall evidence. No scans are
            # invented or replayed with a later pose.
            distances=None
            self.stats['structural_turn_frames_gated']=self.stats.get('structural_turn_frames_gated',0)+1
            if (self.step+1)%4==0:
                self.stats['structural_turn_scans_suppressed']=self.stats.get('structural_turn_scans_suppressed',0)+1
        super().update(gps,compass,odometry,distances,angles,objects)
        for key,value in self.structure28.stats.items():
            self.stats['structural_'+key]=int(value)
        self.stats['structural_absolute_correction']=int(corrected)

    def mask(self):
        if self.final_mask is not None:
            return self.final_mask.copy()
        return super().mask()

    def seal(self):
        if self.final_mask is None:
            self.final_mask=super().mask().copy()
            self.final_mask.setflags(write=False)
            self.final_mapper_step=self.step


class Fusion28(Fusion27):
    """Regularize persistent local bands before conservative source fusion.

    Unlike Fusion27, a strip is explained only in its originating source.
    The output preserves every unparticipating source's residual evidence.
    Band width comes from observed transverse occupancy, with quantization
    allowance. Cross-source consolidation envelopes all participating bands;
    it does not pretend that disagreeing centers are accurately localized.
    """
    SNAP_ANGLE=math.radians(2.5)
    MIN_HALF_BAND=2.0  # coarse cells; uncertainty floor, not a truth wall width
    PERSISTENCE=.55

    def _observed_axis(self,lines):
        # Normalize each origin's total weight so a prolific source is not a
        # majority vote. Require both direction families in the observations.
        if len(lines)<4:
            return None
        weights=[];angles=[]
        totals={}
        for line in lines:
            totals[line['source']]=totals.get(line['source'],0.)+min(line['length'],80.)
        for line in lines:
            angles.append(math.atan2(line['direction'][1],line['direction'][0]))
            weights.append(min(line['length'],80.)/totals[line['source']])
        angles=np.asarray(angles);weights=np.asarray(weights)
        z=np.sum(weights*np.exp(4j*angles))/weights.sum()
        if abs(z)<.96:
            return None
        axis=float(np.angle(z)/4)
        family=np.rint((angles-axis)/(math.pi/2)).astype(int)%2
        if min(int((family==i).sum()) for i in (0,1))<2:
            return None
        return axis,float(abs(z))

    def _band(self,mask,line,axis):
        model=dict(line)
        direction=line['direction'].copy()
        snapped=False
        if axis is not None:
            angle=math.atan2(direction[1],direction[0])
            correction=quarter_angle(axis[0]-angle)
            if abs(correction)<=self.SNAP_ANGLE:
                direction=self._direction([math.cos(angle+correction),math.sin(angle+correction)])
                snapped=abs(correction)>1e-8
        normal=np.array([-direction[1],direction[0]])
        extent=(line['ends']-line['center'])@direction
        ends=line['center']+np.array([extent.min(),extent.max()])[:,None]*direction
        radius=float(line['radius'])+float(np.max(np.abs((line['ends']-line['center'])@normal)))
        bounds0=np.floor(ends.min(axis=0)-radius-2).astype(int)
        bounds1=np.ceil(ends.max(axis=0)+radius+3).astype(int)
        x0,y0=np.maximum(bounds0,0);x1,y1=np.minimum(bounds1,[mask.shape[1],mask.shape[0]])
        yy,xx=np.nonzero(mask[y0:y1,x0:x1])
        points=np.column_stack((xx+x0,yy+y0)).astype(float)-line['center']
        along=points@direction;perp=points@normal
        select=(along>=extent.min()+2)&(along<=extent.max()-2)&(np.abs(perp)<=radius)
        if select.sum()<24:
            return None
        bins=np.rint(perp[select]).astype(int)
        low=int(math.floor(-radius));high=int(math.ceil(radius))
        histogram=np.bincount(bins-low,minlength=high-low+1)
        persistent=np.flatnonzero(histogram>=max(4.,self.PERSISTENCE*float(histogram.max())))+low
        if len(persistent)<2:
            return None
        # Retain the measured persistent envelope, not a skeleton of fixed
        # thickness. One coarse cell in total covers bin/raster uncertainty.
        lower=float(persistent.min())-.5
        upper=float(persistent.max())+.5
        center=(lower+upper)/2
        half=max(self.MIN_HALF_BAND,(upper-lower)/2)
        lower,upper=center-half,center+half
        model.update(direction=direction,normal=normal,ends=ends,radius=radius,
                     lower=lower,upper=upper,snapped=snapped,
                     length=float(extent.max()-extent.min()))
        return model

    @staticmethod
    def _protect(mask,long_lines):
        """Keep compact components and independently oriented short structures."""
        encoded=np.asarray(mask,np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(encoded,connectivity=8)
        keep=np.zeros(count,np.uint8)
        keep[1:]=stats[1:,cv2.CC_STAT_AREA]<=150
        protected=keep[labels].astype(bool)
        if not long_lines:
            return mask.copy()
        skeleton=cv2.ximgproc.thinning(encoded*255)
        proposals=cv2.HoughLinesP(skeleton,1,np.pi/180,threshold=5,minLineLength=4,maxLineGap=1)
        if proposals is None:
            return protected
        distance=cv2.distanceTransform(encoded,cv2.DIST_L2,5)
        draw=np.zeros(mask.shape,np.uint8)
        for item in np.asarray(proposals).reshape(-1,4)[:512]:
            ends=item.reshape(2,2).astype(float);vector=ends[1]-ends[0]
            length=float(np.linalg.norm(vector))
            if length<4:
                continue
            direction=vector/length;center=ends.mean(axis=0)
            explained=False
            for line in long_lines:
                if abs(float(direction@line['direction']))<math.cos(math.radians(12)):
                    continue
                if abs(float((center-line['center'])@line['normal']))>line['radius']+1:
                    continue
                extent=(line['ends']-line['center'])@line['direction']
                along=float((center-line['center'])@line['direction'])
                if extent.min()-2<=along<=extent.max()+2:
                    explained=True
                    break
            if explained:
                continue
            x,y=np.rint(center).astype(int)
            radius=float(distance[np.clip(y,0,mask.shape[0]-1),np.clip(x,0,mask.shape[1]-1)])
            width=int(min(21,max(3,math.ceil(2*radius+2))))
            cv2.line(draw,tuple(item[:2]),tuple(item[2:]),1,width)
        return protected|((draw>0)&mask)

    def fuse(self,source_masks,source_weights=None):
        if source_weights is not None:
            raise ValueError('One equal geometric vote per origin; freshness is not confidence')
        sources={}
        for source,mask in sorted(source_masks.items()):
            if not isinstance(source,(int,np.integer)) or isinstance(source,bool) or source<0:
                raise ValueError('Non-negative integer origin ID required')
            value=np.asarray(mask)
            if value.shape!=self.shape or value.dtype!=np.bool_:
                raise ValueError('Expected one bool RL17 coarse map per origin')
            sources[int(source)]=value.copy()
        lines=[];per_source={}
        for source,mask in sources.items():
            per_source[source]=self._extract(mask,source)
            lines.extend(per_source[source])
        axis=self._observed_axis(lines)
        union=np.zeros(self.shape,bool);residual=np.zeros(self.shape,bool)
        models=[];source_stats={}
        for source,mask in sources.items():
            union|=mask
            own=mask.copy()
            own_models=[model for line in per_source[source] if (model:=self._band(mask,line,axis)) is not None]
            protected=self._protect(mask,own_models)
            for model in own_models:
                self._explain(own,model)
            own|=protected
            residual|=own
            models.extend(own_models)
            source_stats[str(source)]=dict(extracted_lines=len(per_source[source]),
                modeled_lines=len(own_models),input_cells=int(mask.sum()),
                residual_cells=int(own.sum()),protected_cells=int(protected.sum()))
        pixels=np.repeat(np.repeat(residual,3,0),3,1)[:self.height,:self.width].astype(np.uint8)
        self.last_lines=[]
        for cluster in self._clusters(models):
            by_source={}
            for model in cluster:
                by_source.setdefault(model['source'],[]).append(model)
            directions=[]
            for records in by_source.values():
                reference=records[0]['direction']
                directions.append(self._direction(np.mean([r['direction']*(1 if r['direction']@reference>=0 else -1) for r in records],axis=0)))
            reference=directions[0]
            direction=self._direction(np.mean([d*(1 if d@reference>=0 else -1) for d in directions],axis=0))
            normal=np.array([-direction[1],direction[0]])
            corners=[];source_positions=[]
            for source,records in by_source.items():
                source_positions.append(float(np.median([r['center']@normal for r in records])))
                for model in records:
                    for end in model['ends']:
                        corners.extend([end+model['normal']*model['lower'],end+model['normal']*model['upper']])
            corners=np.asarray(corners)
            along=corners@direction;perp=corners@normal
            lo,hi=float(along.min()),float(along.max())
            lower,upper=float(perp.min()),float(perp.max())
            polygon=np.array([direction*lo+normal*lower,direction*hi+normal*lower,
                              direction*hi+normal*upper,direction*lo+normal*upper])
            polygon=np.rint(polygon*3+1).astype(np.int32)
            cv2.fillConvexPoly(pixels,polygon,1,lineType=cv2.LINE_8)
            ends=np.array([direction*lo+normal*(lower+upper)/2,direction*hi+normal*(lower+upper)/2])
            self.last_lines.append(dict(sources=sorted(by_source),
                endpoints=np.rint(ends*3+1).astype(int).tolist(),width_pixels=3*(upper-lower),
                source_normal_spread=3*(max(source_positions)-min(source_positions)),
                participating_bands_enveloped=True))
        pixels=cv2.morphologyEx(pixels,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        count,labels,stats,_=cv2.connectedComponentsWithStats(pixels,connectivity=8)
        keep=np.zeros(count,np.uint8);keep[1:]=stats[1:,cv2.CC_STAT_AREA]>=27
        output=keep[labels].astype(bool)
        self.last_stats=dict(source_count=len(sources),per_source=source_stats,
            input_union_cells=int(union.sum()),residual_cells=int(residual.sum()),
            modeled_lines=len(models),fused_lines=len(self.last_lines),output_pixels=int(output.sum()),
            orthogonal_axis_radians=None if axis is None else axis[0],
            orthogonal_axis_concentration=None if axis is None else axis[1],
            snapped_lines=sum(bool(m['snapped']) for m in models),
            median_modeled_width_pixels=float(np.median([line['width_pixels'] for line in self.last_lines])) if self.last_lines else None,
            no_cross_source_residual_erasure=True,fixed_wall_width=False,
            source_equal_weight=True,navigation_map_modified=False)
        return output
