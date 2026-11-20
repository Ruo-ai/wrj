"""Restore explicitly selected cleanup-backup members without overwriting files."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def restore(backup, members, destination):
    backup, destination = Path(backup).resolve(), Path(destination).resolve()
    manifest = json.loads((backup / 'manifest.json').read_bytes())
    archive = backup / 'before_cleanup.zip'
    if digest(archive.read_bytes()) != manifest['archive_sha256']:
        raise ValueError('Backup archive SHA256 mismatch')
    if destination.exists():
        raise FileExistsError('Use a new destination directory; no files are overwritten')
    prepared = []
    with ZipFile(archive) as zipped:
        for name in dict.fromkeys(members):
            member = PurePosixPath(name)
            if member.is_absolute() or '..' in member.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe archive member')
            expected = manifest['files'][name]['sha256']
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Target leaves destination')
            data = zipped.read(name)
            if digest(data) != expected:
                raise ValueError('Member SHA256 mismatch: ' + name)
            prepared.append((name, target, data, expected))
    destination.mkdir(parents=True, exist_ok=False)
    result = []
    for name, target, data, expected in prepared:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
        actual = digest(target.read_bytes())
        if actual != expected:
            raise ValueError('Restored bytes differ: ' + name)
        result.append(dict(member=name, destination=str(target), sha256=actual))
    with (destination / 'restore_receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup', required=True, type=Path)
    parser.add_argument('--member', required=True, action='append')
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(restore(args.backup, args.member, args.destination),
                     ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
