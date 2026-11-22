"""Structural mapping invariants with synthetic legal geometry; no flights."""
import json
import math

import cv2
import numpy as np
import pytest

from swarm_rescue.solutions.my_drone_red_repair17 import RepairMapper17
from swarm_rescue.solutions.red_structural28 import (
    Fusion28, StructuralHeading28, StructuralMapper28, coherent_axis, quarter_angle,
)


SIZE=(660,480)


def lines(angle):
    rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    return [np.array([[-80.,0.],[80.,0.]])@rotation.T,
            np.array([[0.,-60.],[0.,60.]])@rotation.T]


def mask(segments=()):
    result=np.zeros((160,220),np.uint8)
    for start,end,width in segments:
        cv2.line(result,start,end,1,width)
    return result.astype(bool)


def widths(values):
    delta=np.diff(np.r_[0,values.astype(np.int8),0])
    return (np.flatnonzero(delta==-1)-np.flatnonzero(delta==1)).tolist()


def test_axis_is_observed_and_rotation_equivariant():
    for angle in (0.,.31,.73,-.65):
        axis,confidence=coherent_axis(lines(angle))
        assert abs(quarter_angle(axis-angle))<1e-10 and confidence>.99


def test_one_parallel_family_cannot_establish_orthogonal_world_prior():
    one=lines(.37)[0]
    assert coherent_axis([one,one+10]) is None
    assert coherent_axis([one,np.array([[0.,0.],[80.,30.]])]) is None


def test_heading_reference_does_not_depend_on_initial_drone_heading():
    phi=.27
    for actual_heading in (-2.8,-1.,.6,2.9):
        estimator=StructuralHeading28()
        body=lines(phi-actual_heading)
        for index in range(40):
            compass=actual_heading+(.065 if index%2 else -.065)
            corrected=estimator.observe(body,compass)
        assert abs(quarter_angle(estimator.axis-phi))<1e-6
        assert abs(corrected-actual_heading)<1e-6


def test_heading_reference_rejects_unrelated_wall_directions():
    estimator=StructuralHeading28()
    for _ in range(20):estimator.observe(lines(.2),0.)
    before=estimator.axis
    assert estimator.observe(lines(.65),.04)==.04
    assert estimator.axis==before and estimator.stats['inconsistent_frame']==1


def test_fast_turn_reference_settles_without_becoming_permanent_rejection():
    estimator=StructuralHeading28()
    assert estimator.motion((0.,0.,.08))
    assert estimator.observe(lines(.2),.1)==.1
    assert not estimator.samples
    for _ in range(estimator.TURN_SETTLE_STEPS):estimator.motion((0.,0.,0.))
    assert estimator.settle==0
    estimator.observe(lines(.2),.1)
    assert len(estimator.samples)==1


def test_isolated_angular_noise_does_not_trigger_scan_blackout():
    estimator=StructuralHeading28()
    for value in (0.,.05,-.01,0.,0.,0.):
        assert not estimator.motion((0.,0.,value))
    assert estimator.settle==0
    for value in (.05,.05,.05):estimator.motion((0.,0.,value))
    assert estimator.settle==estimator.TURN_SETTLE_STEPS


def test_turn_gates_submission_lidar_but_keeps_legal_pose_updates(monkeypatch):
    calls=[]
    def update(self,*values):
        self.step+=1;calls.append(values)
    monkeypatch.setattr(RepairMapper17,'update',update)
    mapper=StructuralMapper28(SIZE)
    gps=np.array([1.,2.]);d=np.ones(20)*100;a=np.linspace(-1,1,20)
    mapper.update(gps,.4,np.array([1.,0.,.08]),d,a)
    assert calls[-1][0] is gps and calls[-1][3] is None
    assert mapper.step==1 and mapper.stats['structural_turn_frames_gated']==1
    for _ in range(8):mapper.update(gps,.4,np.zeros(3),d,a)
    assert calls[-1][3] is d


def test_missing_absolute_pose_keeps_original_outage_mapper_path(monkeypatch):
    calls=[]
    monkeypatch.setattr(RepairMapper17,'update',lambda self,*values:calls.append(values))
    mapper=StructuralMapper28(SIZE);d=np.ones(20)
    mapper.update(None,None,np.array([1.,0.,.08]),d,np.ones(20))
    assert calls[0][3] is d and calls[0][0] is None


def test_recovery_keeps_integrated_endpoint_for_pending_scan_anchoring(monkeypatch):
    mapper=StructuralMapper28(SIZE);mapper.position=np.zeros(2)
    mapper.heading=.8;mapper.missing=12;mapper.step=3
    def corrected(lines,compass):
        mapper.structure28.last_correction_applied=True
        return .2
    mapper.structure28.observe=corrected
    seen=[]
    monkeypatch.setattr(RepairMapper17,'update',lambda self,*values:seen.append((self.heading,values[1])))
    mapper.update(np.zeros(2),.3,np.zeros(3),np.ones(20)*100,np.linspace(-1,1,20))
    assert seen==[(.8,.2)]


def test_mapper_seal_stops_evidence_and_protects_copies(monkeypatch):
    mapper=StructuralMapper28(SIZE);mapper.hits[10:20,30:60]=4.;mapper.step=7
    expected=mapper.mask().copy();mapper.seal();mapper.hits[:]=0
    out=mapper.mask();out[:]=False
    monkeypatch.setattr(RepairMapper17,'update',lambda *a,**k:pytest.fail('Updated sealed evidence'))
    mapper.update(None,None,None,None,None)
    assert mapper.final_mapper_step==7 and mapper.step==7
    assert mapper.stats['after_cutoff_observations_not_mapped']==1
    np.testing.assert_array_equal(mapper.mask(),expected)


def test_other_source_unmodelled_evidence_cannot_be_erased_by_single_source_line():
    wrong=mask([((20,70),(200,70),9)])
    correct=mask();correct[74:77,90:100]=True
    source_pixels=np.repeat(np.repeat(correct,3,0),3,1)
    fuser=Fusion28(SIZE);out=fuser.fuse({0:wrong,4:correct})
    assert np.all(out[source_pixels])
    assert fuser.last_stats['no_cross_source_residual_erasure']


def test_parallel_passage_remains_separate():
    sources={i:mask([((20,45+i),(200,45+i),5),((20,65+i),(200,65+i),5)]) for i in range(2)}
    out=Fusion28(SIZE).fuse(sources)
    assert len(widths(out[:,300]))==2 and not out[166,300]


def test_real_small_and_large_box_interiors_are_preserved():
    small=mask();small[25:35,35:45]=True
    large=mask();large[70:130,70:150]=True
    out=Fusion28(SIZE).fuse({0:small,1:large})
    assert out[75:105,105:135].all()
    assert out[210:390,210:450].all()


def test_short_perpendicular_branch_is_preserved():
    item=mask([((20,70),(200,70),5),((100,70),(100,79),3)])
    out=Fusion28(SIZE).fuse({0:item})
    assert out[235,301] and out[211,301]


def test_sparse_tiny_band_spurs_are_removed_without_eroding_main_wall():
    item=mask();item[69:74,20:200]=True
    for x in (45,85,125,165):item[66:69,x]=True
    out=Fusion28(SIZE).fuse({0:item})
    assert out[211,100:580].all()
    for x in (45,85,125,165):assert not out[198:201,x*3:x*3+3].any()


def test_arbitrary_oriented_observation_is_not_forced_to_world_axes():
    item=mask([((20,40),(190,110),7)])
    fuser=Fusion28(SIZE);out=fuser.fuse({0:item})
    assert out[226,316] and not out[121,316]
    assert fuser.last_stats['orthogonal_axis_radians'] is None


def test_l_corner_and_rectangle_outline_remain_connected():
    item=mask([((40,30),(160,30),5),((160,30),(160,120),5),
               ((160,120),(40,120),5),((40,120),(40,30),5)])
    out=Fusion28(SIZE).fuse({0:item})
    _,labels=cv2.connectedComponents((~out).astype(np.uint8))
    assert labels[240,300]!=labels[5,5]
    assert not out[240,300]


def test_width_follows_observations_not_fixed_nine_pixels():
    thin=mask([((20,60),(200,60),3)])
    broad=mask([((20,60),(200,60),11)])
    a=Fusion28(SIZE).fuse({0:thin});b=Fusion28(SIZE).fuse({0:broad})
    assert widths(b[:,300])[0]>widths(a[:,300])[0]>9


def test_source_order_repeated_calls_and_immutable_inputs():
    sources={i:mask([((20,60+i),(200,60+i),5),((90,20),(90,130),5)]) for i in range(3)}
    copies={i:a.copy() for i,a in sources.items()}
    fuser=Fusion28(SIZE);first=fuser.fuse(sources)
    second=fuser.fuse(dict(reversed(list(sources.items()))))
    np.testing.assert_array_equal(first,second)
    for i in sources:np.testing.assert_array_equal(sources[i],copies[i])
    json.dumps(fuser.last_stats,allow_nan=False);json.dumps(fuser.last_lines,allow_nan=False)


def test_empty_maps_and_invalid_inputs():
    fuser=Fusion28(SIZE)
    assert not fuser.fuse({}).any()
    with pytest.raises(ValueError):fuser.fuse({0:np.zeros((2,3),bool)})
    with pytest.raises(ValueError):fuser.fuse({0:mask().astype(float)})
    with pytest.raises(ValueError):fuser.fuse({True:mask()})
    with pytest.raises(ValueError):fuser.fuse({0:mask()},source_weights={0:4})
