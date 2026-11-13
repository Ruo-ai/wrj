"""Automatic append interface checks; never generate into the real map_data."""
import json

import pytest

from . import append_maps
from .errors import DuplicateStructureError, LayoutExhaustedError


def record(out,index,seed):
    folder = out/'.yh_records'/f'yh_map_fs_{index}'
    folder.mkdir(parents=True)
    (folder/'metadata.json').write_text(json.dumps({'seed':seed}))
    (out/f'yh_map_fs_{index}.json').write_text('{}')


def test_resume_skips_gaps_and_reserved_indices(tmp_path):
    record(tmp_path,1,1000)
    record(tmp_path,3,1005)
    (tmp_path/'yh_map_fs_8.png').write_bytes(b'existing')
    (tmp_path/'map_99.json').write_text('{}')
    assert append_maps.next_values(tmp_path)==(9,1006)


def test_repeated_commands_append_requested_count(tmp_path,monkeypatch):
    calls = []
    def fake_export(args):
        assert args.type=='semifinal' and args.difficulty=='hard'
        assert (args.width,args.height) in ((1700,1100),(1660,1122))
        calls.append((args.start_index,args.seed))
        record(args.output,args.start_index,args.seed)
    monkeypatch.setattr(append_maps,'export',fake_export)
    append_maps.generate_many(2,tmp_path)
    append_maps.generate_many(1,tmp_path)
    assert calls==[(1,1000),(2,1001),(3,1002)]


def test_retry_seed_without_consuming_map_number(tmp_path,monkeypatch):
    calls = []
    def fake_export(args):
        calls.append((args.start_index,args.seed))
        if args.seed==1000:
            raise DuplicateStructureError('duplicate')
        if args.seed==1001:
            raise LayoutExhaustedError('no room')
        record(args.output,args.start_index,args.seed)
    monkeypatch.setattr(append_maps,'export',fake_export)
    append_maps.generate_many(1,tmp_path)
    assert calls==[(1,1000),(1,1001),(1,1002)]
    assert append_maps.next_values(tmp_path)==(2,1003)


def test_native_or_io_failure_is_not_silently_retried(tmp_path,monkeypatch):
    calls = []
    def fail(args):
        calls.append(args)
        raise RuntimeError('Native probe failed')
    monkeypatch.setattr(append_maps,'export',fail)
    with pytest.raises(RuntimeError,match='Native'):
        append_maps.generate_many(2,tmp_path)
    assert len(calls)==1
    with append_maps.append_lock(tmp_path):
        pass  # The failed run released its lock.


def test_lock_and_invalid_count(tmp_path):
    with append_maps.append_lock(tmp_path):
        with pytest.raises(RuntimeError,match='Another'):
            with append_maps.append_lock(tmp_path):
                pass
    with pytest.raises(ValueError,match='positive'):
        append_maps.generate_many(0,tmp_path/'unused')
    assert not (tmp_path/'unused').exists()


def test_append_real_native_map(tmp_path):
    append_maps.generate_many(1,tmp_path)
    assert (tmp_path/'yh_map_fs_1.json').exists()
    meta = json.loads((tmp_path/'.yh_records/yh_map_fs_1/metadata.json').read_text())
    assert meta['seed']==1000 and meta['native_load_verified']

