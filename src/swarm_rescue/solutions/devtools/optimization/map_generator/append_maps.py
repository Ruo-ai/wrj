"""Generate N additional maps with automatic numbering and fresh seeds."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import subprocess

from .generate import export
from .errors import LayoutExhaustedError, DuplicateStructureError


REPO = Path(__file__).resolve().parents[6]


def next_values(out):
    highest,seed = 0,1000
    if out.exists():
        paths = list(out.iterdir())
        records = out/'.yh_records'
        if records.exists():
            paths.extend(records.iterdir())
        for path in paths:
            match = re.fullmatch(r'yh_map_fs_([0-9]+)(?:\.(?:json|png))?',path.name)
            if match:
                highest = max(highest,int(match[1]))
        for path in list(records.glob('map_*_yh/metadata.json'))+list(records.glob('yh_map_fs_*/metadata.json')):
            value = json.loads(path.read_text(encoding='utf-8'))['seed']
            if type(value) is not int:
                raise ValueError(f'Invalid recorded seed: {path}')
            seed = max(seed,value+1)
    return highest+1,seed


@contextmanager
def append_lock(out):
    """OS lock is released on process exit; an empty lock file can safely remain."""
    records = out/'.yh_records'
    records.mkdir(parents=True,exist_ok=True)
    with (records/'append.lock').open('a+b') as stream:
        if stream.tell()==0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as error:
                raise RuntimeError('Another automatic generation command is running') from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            try:
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError('Another automatic generation command is running') from error
            try:
                yield
            finally:
                fcntl.flock(stream,fcntl.LOCK_UN)


def generate_many(count,out=None):
    if type(count) is not int or count<1:
        raise ValueError('Count must be a positive integer')
    out = (REPO/'map_data' if out is None else Path(out)).resolve()
    with append_lock(out):
        index,seed = next_values(out)
        for completed in range(count):
            for attempt in range(8):
                width,height = ((1700,1100),(1660,1122))[seed%2]
                args = argparse.Namespace(output=out,type='semifinal',difficulty='hard',
                    width=width,height=height,thickness=None,passage=None,count=1,
                    start_index=index,seed=seed,save_official=True)
                print(f'[{completed+1}/{count}] yh_map_fs_{index}, seed={seed}',flush=True)
                try:
                    export(args)
                except (LayoutExhaustedError,DuplicateStructureError) as error:
                    print(f'Skipping seed {seed}: {error}',flush=True)
                    seed += 1
                    continue
                index += 1
                seed += 1
                break
            else:
                raise RuntimeError(f'No valid map after 8 seeds; {completed}/{count} maps saved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('count',type=int,help='Number of NEW maps to generate')
    args = parser.parse_args()
    if args.count<1:
        parser.error('count must be positive')
    try:
        generate_many(args.count)
    except (ValueError,OSError,RuntimeError,subprocess.TimeoutExpired) as error:
        parser.exit(1,str(error)+'\n')


if __name__=='__main__':
    main()
