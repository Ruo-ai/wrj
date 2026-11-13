"""Small G1 checks. Family fixtures are generated in memory, not as a dataset."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from . import generate
from .generate import build, export
from .structures import FAMILIES
from .validate import digest, geometry_hash, structure_hash, validate
from .official_format import editor_document, check_sources, sources, read_spec
from .official_profile import read_profiles
from swarm_rescue.tools.map_wall_editor import load_elements, serialize_elements


@pytest.mark.parametrize('family', FAMILIES)
def test_each_family(family):
    size = (1700,1100) if family=='semifinal' else (960,800)
    spec, stats, mask = build(family,*size,'medium',42)
    assert stats['connected_components'] == 1
    assert stats['wall_count'] >= 1
    assert np.array_equal(validate(spec)[0], mask)
    if family in ('rooms','branches','maze','mixed'):
        assert stats['door_count'] >= stats['room_count']-1
    if family in ('bottleneck','narrow','junction'):
        assert spec['gates']


def test_repeat_and_seed_variation():
    params = ('maze',960,800,'hard')
    first = build(*params,17)[0]
    assert build(*params,17)[0] == first
    assert geometry_hash(build(*params,18)[0]) != geometry_hash(first)
    first['walls'].reverse()
    assert geometry_hash(first) == geometry_hash(build(*params,17)[0])


@pytest.mark.parametrize('damage', ('spawn','bounds','overlap','blocked_gate','width','island','center'))
def test_reject_invalid_geometry(damage):
    spec = build('bottleneck',960,800,'hard',8)[0]
    axis,mid,lo,hi = spec['gates'][0]
    if damage=='spawn':
        spec['spawn'] = [mid,lo-60]
    elif damage=='bounds':
        spec['walls'][0][1] = -1
    elif damage=='overlap':
        spec['walls'].append(copy.copy(spec['walls'][0]))
    elif damage=='blocked_gate':
        spec['walls'].append(['box',mid-2,lo,mid+2,hi])
    elif damage=='width':
        spec['passage_width'] += 10
    elif damage=='island':
        spec['gates'].clear()
        spec['walls'].append(['box',mid-2,lo,mid+2,hi])
    elif damage=='center':
        spec['center'] = [mid,lo-60]
    with pytest.raises(ValueError):
        validate(spec)


@pytest.mark.parametrize('family,passage', (('mixed',70),('narrow',38),('official',100)))
def test_publish_native_and_preserve(tmp_path,family,passage):
    import argparse
    args = argparse.Namespace(type=family,width=960,height=800,difficulty='medium',
                              seed=9,count=1,thickness=None if family=='official' else 18,passage=passage,output=tmp_path)
    export(args)
    folder, = tmp_path.iterdir()
    files = {p.name:p.read_bytes() for p in folder.iterdir()}
    spec = json.loads(files['generation.json'])
    meta = json.loads(files['metadata.json'])
    assert files.keys()=={'map.json','generation.json','metadata.json','preview.png'}
    assert json.loads(files['map.json'])==editor_document(spec)
    check_sources(spec)
    assert meta['geometry_sha256']==geometry_hash(spec) and meta['native_load_verified']
    assert meta['content_sha256']==digest(spec)
    if family=='mixed':
        assert any(w[0]=='box' for w in spec['walls'])
    with pytest.raises(FileExistsError):
        export(args)
    assert files == {p.name:p.read_bytes() for p in folder.iterdir()}


@pytest.mark.parametrize('family,size,difficulty', (
    ('maze',600,'hard'), ('narrow',600,'hard'), ('junction',600,'hard'),
    ('blocks',1800,'easy')))
def test_size_boundaries(family,size,difficulty):
    spec,_,_ = build(family,size,size,difficulty,42)
    validate(spec)


def test_reflection_deduplication():
    spec = build('mixed',960,800,'medium',42)[0]
    reflected = copy.deepcopy(spec)
    reflected['walls'] = [[kind,960-r,t,960-l,b] for kind,l,t,r,b in spec['walls']]
    reflected['spawn'] = [200,200]
    reflected['center'] = [300,300]
    assert structure_hash(spec)==structure_hash(reflected)
    assert geometry_hash(spec)!=geometry_hash(reflected)


def test_publish_failure_and_batch(tmp_path,monkeypatch):
    import argparse
    args = argparse.Namespace(type='mixed',width=960,height=800,difficulty='medium',
                              seed=19,count=2,thickness=None,passage=None,output=tmp_path)
    def fail(*_):
        raise RuntimeError('Injected native load failure')
    monkeypatch.setattr(generate,'verify_native',fail)
    with pytest.raises(RuntimeError,match='Injected'):
        export(args)
    assert not list(tmp_path.iterdir())
    calls = []
    # Test only batch publishing here. A separate test exercises the real probe.
    monkeypatch.setattr(generate,'verify_native',lambda *call: calls.append(call))
    export(args)
    assert len(calls)==2
    assert sorted(json.loads(p.read_text())['seed']
                  for p in tmp_path.glob('*/metadata.json'))==[19,20]
    assert not list(tmp_path.glob('.g1-*'))


def test_protected_output_and_count(tmp_path):
    import argparse
    args = argparse.Namespace(count=1,output=Path(__file__).resolve().parents[6]/'map_data')
    with pytest.raises(ValueError,match='separate'):
        export(args)
    args.output = tmp_path/'never_created'
    args.count = 0
    with pytest.raises(ValueError,match='positive'):
        export(args)
    assert not args.output.exists()


def test_parameter_overrides():
    spec,stats,_ = build('bottleneck',960,800,'medium',42,thickness=23,passage=61)
    assert stats['wall_thickness']==23
    assert all(r-l==23 for _,l,t,r,b in spec['walls'])
    assert all(hi-lo==61 for _,_,lo,hi in spec['gates'])


@pytest.mark.parametrize('difficulty',('easy','medium','hard'))
def test_official_style_geometry(difficulty):
    spec,stats,mask = build('official',1200,900,difficulty,42)
    assert stats['wall_thickness']==6
    assert spec['center_size']!=[48,48]
    assert stats['actual_internal_lines']>=.8*stats['target_internal_lines']
    assert stats['box_count']==sum(s[0]=='box' for s in spec['walls'])
    l,t,r,b = spec['start_area']
    assert not mask[t:b,l:r].any()
    assert build('official',1200,900,difficulty,42)[0]==spec


@pytest.mark.parametrize('thickness',(6,23))
def test_official_editor_roundtrip(thickness):
    spec = build('bottleneck',960,800,'medium',42,thickness=thickness,passage=61)[0]
    doc = editor_document(spec)
    assert serialize_elements(960,800,load_elements(doc,960,800))==doc
    code = sources(spec)
    for name,source in code.items():
        compile(source,name,'exec')
    if thickness==23:
        assert all(e['type']=='rect_wall' for e in doc['elements'][:-2])
        assert 'width=23' in code['walls_generated.py']
    else:
        assert 'wall_thickness=6' in code['walls_generated.py']


def test_official_reference_schema():
    refs = read_profiles()
    assert len(refs)==6
    assert all(p['line_thicknesses']==[6] for p in refs)
    assert sorted(p['internal_line_count'] for p in refs)==[2,13,25,49,50,56]


def test_editor_tampering_rejected(tmp_path):
    spec = build('mixed',960,800,'medium',42)[0]
    doc = editor_document(spec)
    (tmp_path/'generation.json').write_text(json.dumps(spec),encoding='utf-8')
    doc['elements'][4]['color'] = '#ff0000'
    (tmp_path/'map.json').write_text(json.dumps(doc),encoding='utf-8')
    with pytest.raises(ValueError,match='differs'):
        read_spec(tmp_path/'map.json')
