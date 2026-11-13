"""Verify an exported G1 map in a fresh native process."""
import argparse
import json
from pathlib import Path
import tempfile

from .generate import verify_native
from .validate import digest, geometry_hash, structure_hash, validate
from .official_format import read_spec
from .naming import record_directory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path,help='Dataset folder or numbered map_data/map_01_yh.json')
    folder = p.parse_args().folder
    data = folder if folder.is_file() else folder/'map.json'
    spec = read_spec(data)
    meta = json.loads((record_directory(data)/'metadata.json').read_text(encoding='utf-8'))
    validate(spec)
    if (geometry_hash(spec)!=meta['geometry_sha256'] or
            structure_hash(spec)!=meta['structure_sha256'] or
            digest(spec)!=meta['content_sha256']):
        p.exit(1,'Geometry hash mismatch\n')
    if 'official_editor_sha256' in meta and digest(json.loads(data.read_text(encoding='utf-8')))!=meta['official_editor_sha256']:
        p.exit(1,'Official editor hash mismatch\n')
    with tempfile.TemporaryDirectory() as tmp:
        verify_native(data,Path(tmp))
    print('Verified',meta['native_class'])


if __name__=='__main__':
    main()
