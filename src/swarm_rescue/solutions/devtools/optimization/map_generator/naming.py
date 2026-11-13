"""Numbered JSON/PNG publication without replacing existing official or user files."""
from pathlib import Path
import re


def record_directory(data):
    data = Path(data)
    if re.fullmatch(r'(?:map_[0-9]+_yh|yh_map_fs_[0-9]+)\.json',data.name):
        return data.parent/'.yh_records'/data.stem
    return data.parent


def numbered_name(index):
    if type(index) is not int or index<1:
        raise ValueError('Start index must be a positive integer')
    return f'yh_map_fs_{index}'


def check_available(out,name):
    targets = (out/(name+'.json'),out/(name+'.png'),out/'.yh_records'/name)
    for target in targets:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'Already exists, choose another --start-index: {target}')


def publish_numbered(stage,out,name):
    check_available(out,name)
    records = out/'.yh_records'
    records.mkdir(exist_ok=True)
    record = records/name
    record.mkdir()  # Fails on a collision; no overwrite or shared record directory.
    created = []
    try:
        pairs = [(stage/'generation.json',record/'generation.json'),
                 (stage/'metadata.json',record/'metadata.json'),
                 (stage/'preview.png',out/(name+'.png')),
                 (stage/'map.json',out/(name+'.json'))]
        # JSON is published last. A failure rolls back only files opened by us.
        for source,target in pairs:
            with target.open('xb') as stream:
                created.append(target)
                stream.write(source.read_bytes())
    except BaseException:
        for target in reversed(created):
            target.unlink()
        record.rmdir()
        raise
