"""Small temporary fixtures and adversarial split/index checks for G2."""
import argparse
import copy
import json
import shutil

import pytest

from ..map_generator import generate
from ..map_generator.validate import digest
from .index import build_index, signature_variants
from .split import group_maps, make_split
from .artifacts import configuration, freeze, verify, file_hash
from .__main__ import reproduction


def row(number,family='rooms',seed=None):
    return dict(id=str(number),map_path=f'map_{number:03d}.json',family=family,
                difficulty='hard',seed=number*10 if seed is None else seed,size=[1200,900],
                layout_algorithm='fixture',generator_version=2,wall_thickness=6,passage_width=100,
                calibration_sha256=None,structure_sha256=f'structure-{number}',
                signatures=[format(1<<number,'x')])


def test_determinism_group_integrity_and_hard_view():
    rows = [row(i) for i in range(1,21)]
    # Related seeds with same parameters must stay together.
    rows[0]['seed'],rows[1]['seed'] = 1001,1002
    first = make_split(rows)
    assert make_split(list(reversed(rows)))==first
    for group in first['groups']:
        chosen = first['assignment'][group['id']]
        assert set(group['active'])<=set(first['bins'][chosen])
    assert first['bins']['hard']==first['bins']['test']
    primary = [i for s in ('train','validation','test','extrapolation') for i in first['bins'][s]]
    assert len(primary)==len(set(primary))==len(rows)
    assert all(first['bins'][s] for s in ('train','validation','test'))


def test_duplicate_alias_and_near_shape_merge():
    rows = [row(i) for i in range(1,7)]
    rows[1]['structure_sha256'] = rows[0]['structure_sha256']
    rows[2]['signatures'] = rows[3]['signatures']
    groups,duplicates,edges = group_maps(rows)
    assert duplicates[0]['kept']=='1' and duplicates[0]['excluded']==['2']
    assert any(set(g['members'])=={'3','4'} for g in groups)
    assert any(e['reason']=='near_geometry' for e in edges)


def test_same_seed_across_families_stays_together():
    rows = [row(1,'rooms',42),row(2,'maze',42)]
    groups,_,_ = group_maps(rows)
    assert len(groups)==1


def test_extrapolation_holds_out_entire_connected_group():
    rows = [row(i) for i in range(1,8)]+[row(8,'ring')]
    rows[0]['seed'] = rows[-1]['seed']
    result = make_split(rows,holdout_families=['ring'])
    assert set(result['bins']['extrapolation'])=={'1','8'}
    assert not set(result['bins']['extrapolation'])&set(result['bins']['train'])
    assert set(result['bins']['extrapolation'])<=set(result['bins']['hard'])


@pytest.mark.parametrize('config', [dict(ratios=(0,0,0)),dict(ratios=(-1,1,1)),
    dict(ratios=(float('nan'),1,1)),dict(seed_block=0),dict(near_threshold=.2),
    dict(holdout_families=['missing'])])
def test_invalid_policy_rejected(config):
    with pytest.raises(ValueError):
        make_split([row(i) for i in range(1,7)],**config)


def test_insufficient_groups_refused():
    with pytest.raises(ValueError,match='independent groups'):
        make_split([row(1,seed=1000),row(2,seed=1001),row(3,seed=1002)])


def test_signature_reflection_and_rotation():
    spec = dict(size=[800,800],walls=[['box',100,200,160,420]])
    mirrored = dict(size=[800,800],walls=[['box',640,200,700,420]])
    rotated = dict(size=[800,800],walls=[['box',200,640,420,700]])
    assert signature_variants(spec)==signature_variants(mirrored)==signature_variants(rotated)


@pytest.fixture(scope='module')
def small_fixture(tmp_path_factory):
    source = tmp_path_factory.mktemp('g2_maps')
    # G2 tests serialization/index logic, not native physics (already tested in G1).
    # Explicit mock receipts are confined to these three temporary fixtures.
    patch = pytest.MonkeyPatch()
    patch.setattr(generate,'verify_native',lambda *_:None)
    try:
        for i,family in enumerate(('rooms','maze','blocks'),1):
            args = argparse.Namespace(type=family,width=960,height=800,difficulty='medium',
                seed=i*100,count=1,thickness=None,passage=None,output=source,
                save_official=True,start_index=i)
            generate.export(args)
    finally:
        patch.undo()
    return source


@pytest.fixture
def maps(small_fixture,tmp_path):
    target = tmp_path/'maps'
    shutil.copytree(small_fixture,target)
    return target


def test_freeze_verify_and_no_overwrite(maps,tmp_path):
    rows = build_index(maps)
    assert len(rows)==3
    output = tmp_path/'dataset'
    info = freeze(rows,maps,output,configuration())
    assert info['split_counts']==dict(train=1,validation=1,test=1,extrapolation=0,hard=0)
    assert verify(output)==info
    before = (output/'dataset.json').read_bytes()
    with pytest.raises(FileExistsError):
        freeze(rows,maps,output,configuration())
    assert (output/'dataset.json').read_bytes()==before
    repeat = tmp_path/'repeat'
    freeze(list(reversed(rows)),maps,repeat,configuration())
    assert (repeat/'dataset.json').read_bytes()==before


@pytest.mark.parametrize('damaged',('map','png','metadata','generation'))
def test_changed_source_invalidates_frozen_dataset(maps,tmp_path,damaged):
    rows = build_index(maps)
    output = tmp_path/'dataset'
    freeze(rows,maps,output,configuration())
    target = {'map':maps/'map_01_yh.json','png':maps/'map_01_yh.png',
              'metadata':maps/'.yh_records/map_01_yh/metadata.json',
              'generation':maps/'.yh_records/map_01_yh/generation.json'}[damaged]
    target.write_bytes(target.read_bytes()+b' ')
    with pytest.raises(ValueError,match='Source file changed'):
        verify(output)


def test_manifest_tampering_rejected_even_with_updated_file_digest(maps,tmp_path):
    output = tmp_path/'dataset'
    freeze(build_index(maps),maps,output,configuration())
    train = json.loads((output/'train.json').read_text())
    train['maps'][0]['primary_split'] = 'test'
    (output/'train.json').write_text(json.dumps(train))
    envelope = json.loads((output/'dataset.json').read_text())
    envelope['files']['train.json'] = file_hash(output/'train.json')
    (output/'dataset.json').write_text(json.dumps(envelope))
    with pytest.raises(ValueError,match='Primary split'):
        verify(output)


def test_seed_reproduction_and_mismatch(maps,monkeypatch):
    data = maps/'map_01_yh.json'
    meta = json.loads((maps/'.yh_records/map_01_yh/metadata.json').read_text())
    assert reproduction(data)==meta['content_sha256']
    from . import __main__ as cli
    original = cli.build_map
    def changed(*args,**kwargs):
        spec,stats,mask = original(*args,**kwargs)
        spec['spawn'][0] += 1
        return spec,stats,mask
    monkeypatch.setattr(cli,'build_map',changed)
    with pytest.raises(ValueError,match='reproduction differs'):
        reproduction(data)


def test_incomplete_map_is_not_silently_skipped(maps):
    (maps/'map_04_yh.json').write_text('{}')
    with pytest.raises(ValueError,match='invalid/incomplete map'):
        build_index(maps)
