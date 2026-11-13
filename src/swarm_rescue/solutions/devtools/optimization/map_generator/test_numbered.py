"""Numbered official-path output; all generated test data stays in pytest temp dirs."""
import argparse
import json
from pathlib import Path
import shutil

import pytest

from . import generate, official_profile
from .naming import numbered_name, record_directory, publish_numbered
from .official_format import read_spec


def arguments(out,count=1,start=1):
    return argparse.Namespace(type='official',width=1200,height=900,difficulty='hard',
                              seed=42,count=count,thickness=None,passage=None,output=out,
                              save_official=True,start_index=start)


def test_numbered_native_publish(tmp_path):
    generate.export(arguments(tmp_path))
    assert {p.name for p in tmp_path.iterdir()}=={'yh_map_fs_1.json','yh_map_fs_1.png','.yh_records'}
    spec = read_spec(tmp_path/'yh_map_fs_1.json')
    meta = json.loads((record_directory(tmp_path/'yh_map_fs_1.json')/'metadata.json').read_text())
    assert meta['native_load_verified'] and spec['size']==[1200,900]
    # Re-load the public named file, not merely the staging dataset filename.
    generate.verify_native(tmp_path/'yh_map_fs_1.json',tmp_path)
    before = (tmp_path/'yh_map_fs_1.json').read_bytes()
    with pytest.raises(FileExistsError):
        generate.export(arguments(tmp_path))
    assert (tmp_path/'yh_map_fs_1.json').read_bytes()==before


def test_batch_numbering_and_preflight(tmp_path,monkeypatch):
    calls = []
    monkeypatch.setattr(generate,'verify_native',lambda *args:calls.append(args))
    args = arguments(tmp_path,count=2,start=20)
    # A conflict on the last planned name stops the batch before generating anything.
    occupied = tmp_path/'yh_map_fs_21.png'
    occupied.write_bytes(b'user image')
    with pytest.raises(FileExistsError):
        generate.export(args)
    assert not calls and not (tmp_path/'yh_map_fs_20.json').exists()
    assert occupied.read_bytes()==b'user image'
    args.start_index = 22
    generate.export(args)
    assert len(calls)==2
    assert (tmp_path/'yh_map_fs_22.json').exists() and (tmp_path/'yh_map_fs_23.png').exists()
    assert json.loads((tmp_path/'.yh_records/yh_map_fs_23/metadata.json').read_text())['seed']==43
    assert numbered_name(100)=='yh_map_fs_100'


def test_rollback_preserves_existing_files(tmp_path,monkeypatch):
    stage = tmp_path/'stage'
    stage.mkdir()
    for name in ('generation.json','metadata.json','map.json','preview.png'):
        (stage/name).write_bytes(b'test')
    real_open = Path.open
    def fail_last(path,*args,**kwargs):
        if path.name=='yh_map_fs_1.json' and args==('xb',):
            raise OSError('Injected publish failure')
        return real_open(path,*args,**kwargs)
    user = tmp_path/'map_01.json'
    user.write_bytes(b'official original')
    monkeypatch.setattr(Path,'open',fail_last)
    with pytest.raises(OSError,match='Injected'):
        publish_numbered(stage,tmp_path,'yh_map_fs_1')
    assert not (tmp_path/'yh_map_fs_1.png').exists()
    assert not (tmp_path/'.yh_records/yh_map_fs_1').exists()
    assert user.read_bytes()==b'official original'


def test_generated_maps_excluded_from_reference_bank(tmp_path,monkeypatch):
    data = tmp_path/'map_data'
    data.mkdir()
    for source in (official_profile.REPO/'map_data').glob('map_*.json'):
        if source.name in ('map_01.json','map_02.json','map_03.json','map_04.json','map_05.json','map_07.json'):
            shutil.copyfile(source,data/source.name)
    (data/'yh_map_fs_1.json').write_text('not an official reference')
    monkeypatch.setattr(official_profile,'REPO',tmp_path)
    official_profile.read_profiles.cache_clear()
    try:
        assert [p['source'] for p in official_profile.read_profiles()]==['map_01.json','map_02.json','map_03.json','map_04.json','map_05.json','map_07.json']
    finally:
        official_profile.read_profiles.cache_clear()

