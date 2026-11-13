"""Small deterministic semifinal fixtures; no batch strategy evaluations."""
import copy
import json
import argparse

import pytest

from .generate import build,export
from .official_format import editor_document,read_spec,sources
from .structures import spawn_points
from .validate import validate,geometry_hash,structure_hash
from .semifinal import RULES,reference_manifest


@pytest.mark.parametrize('seed',range(1000,1012))
def test_layout_zone_matrix(seed):
    spec,stats,_ = build('semifinal',1700,1100,'hard',seed)
    assert spec['rules']==RULES
    assert stats['connected_components']==1
    assert len(spec['walls'])>=10
    assert stats['layout_algorithm']=='semifinal-partition-mix-v5'
    assert len(spec['zones'])==2
    assert stats['route_reachable_fraction']>=.98
    assert stats['route_p90_px']>=.9*(1700**2+1100**2)**.5
    assert stats['detour_p90']>=1.7
    assert len(spawn_points(spec['spawn'],10))==10


def test_reproducible_and_zone_identity():
    spec,_,_ = build('semifinal',1700,1100,'hard',1000)
    assert build('semifinal',1700,1100,'hard',1000)[0]==spec
    changed = copy.deepcopy(spec);changed['zones']=[]
    assert geometry_hash(changed)!=geometry_hash(spec)
    assert structure_hash(changed)==structure_hash(spec)
    assert len(reference_manifest())==10
    doc = editor_document(spec)
    assert sum(e['type'].startswith('no_') for e in doc['elements'])==2
    for name,source in sources(spec).items():
        compile(source,name,'exec')


def test_mixed_batch_has_distinct_structures_and_families():
    layouts=[build('semifinal',1660,1122,'hard',s) for s in range(1020,1030)]
    assert len({structure_hash(spec) for spec,_,_ in layouts})==10
    assert len({tuple(spec['spawn']) for spec,_,_ in layouts})>=8
    assert len({stats['partition_leaf_count'] for _,stats,_ in layouts})>=3


@pytest.mark.parametrize('damage',('outside','spawn','type','rules','blue_spawn'))
def test_invalid_zone_and_profile_rejected(damage):
    spec,_,_ = build('semifinal',1700,1100,'hard',1000)
    if damage=='outside':
        spec['zones'][0]['rect'][0]=-1
    elif damage=='spawn':
        x,y=spec['spawn'];spec['zones'][0]['rect']=[x-90,y-90,x+90,y+90]
    elif damage=='type':
        spec['zones'][0]['type']='KILL_ZONE'
    elif damage=='rules':
        spec['rules']['lidar_range']=300
    else:
        spec['spawn']=[55,170]  # Five-drone padding is not enough for 10 drones.
    with pytest.raises(ValueError):
        validate(spec)


def test_both_zones_native_export_and_index(tmp_path):
    from ..dataset_manager.index import index_map
    args=argparse.Namespace(output=tmp_path,type='semifinal',difficulty='hard',
        width=1700,height=1100,thickness=None,passage=None,count=1,start_index=1,
        seed=1000,save_official=True)
    export(args)
    data=tmp_path/'yh_map_fs_1.json'
    spec=read_spec(data)
    meta=json.loads((tmp_path/'.yh_records/yh_map_fs_1/metadata.json').read_text())
    assert meta['timestep_limit']==5400 and meta['blue_drones']==10
    assert meta['native_load_verified'] and spec['rules']==RULES
    assert index_map(data)['family']=='semifinal'


@pytest.mark.parametrize('walls',[
    [['wall',100,100,106,500],['wall',88,200,94,450]],
    [['wall',100,200,300,206],['wall',240,180,350,186]],
    [['wall',100,100,106,1093]],
])
def test_accidental_slits_rejected(walls):
    from .wall_spacing import check_spacing
    with pytest.raises(ValueError,match='wall slit'):
        check_spacing(dict(size=[1700,1100],walls=walls))


def test_joint_and_real_corridor_allowed():
    from .wall_spacing import check_spacing
    check_spacing(dict(size=[1700,1100],walls=[
        ['wall',100,100,300,106],['wall',100,106,106,300],
        ['wall',146,146,152,300]]))


