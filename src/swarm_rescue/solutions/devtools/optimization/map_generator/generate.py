"""Generate validated maps for offline five-drone experiments."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from .structures import FAMILIES, DIFFICULTY, VERSION, geometry, spawn_points
from .validate import CLEARANCE, digest, geometry_hash, structure_hash, validate, wall_mask
from .official_format import FORMAT, editor_document, render_preview, publish_sources
from .naming import numbered_name, check_available, publish_numbered
from .errors import LayoutExhaustedError, DuplicateStructureError


def build(family, width, height, difficulty, seed, thickness=None, passage=None):
    if family not in FAMILIES or difficulty not in DIFFICULTY:
        raise ValueError('Unknown family or difficulty')
    if not all(600<=v<=1800 and v%2==0 for v in (width,height)):
        raise ValueError('Width and height must be even, in 600..1800')
    if thickness is not None and not 6<=thickness<=32:
        raise ValueError('Thickness must be in 6..32')
    if passage is not None and not 38<=passage<=140:
        raise ValueError('Passage must be in 38..140')
    if family=='semifinal':
        from .semifinal import build_semifinal
        return build_semifinal(width,height,difficulty,seed,thickness,passage)
    if family=='official':
        from .official_layout import build_official
        return build_official(width,height,difficulty,seed,thickness,passage)
    reason = ''
    for attempt in range(32):
        rng = random.Random(f'g1:{VERSION}:{seed}:{attempt}')
        try:
            walls,gates,rooms,ends = geometry(family,width,height,difficulty,rng.getrandbits(64),
                                              thickness,passage)
            spec = dict(version=VERSION,size=[width,height],walls=sorted(walls),gates=gates,
                        passage_width=passage or DIFFICULTY[difficulty][1])
            mask = wall_mask(spec)
            distances = cv2.distanceTransform(1-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            # Sample positions with sufficient room for the entire formation.
            ys,xs = np.mgrid[48:height-48:8,48:width-48:8]
            safe = np.ones(xs.shape,dtype=bool)
            for dx,dy in spawn_points([0,0]):
                safe &= distances[ys+dy,xs+dx]>CLEARANCE
            candidates = np.column_stack((ys[safe],xs[safe]))
            if family in ('narrow','bottleneck'):
                candidates = candidates[candidates[:,1]<gates[0][1]-70]
            if not len(candidates):
                raise ValueError('No room for formation')
            y,x = map(int,rng.choice(candidates))
            spec['spawn'] = [x,y]
            center_candidates = np.argwhere(distances[::8,::8]>40)*8
            far = np.hypot(center_candidates[:,0]-y,center_candidates[:,1]-x)>120
            center_candidates = center_candidates[far]
            if not len(center_candidates):
                raise ValueError('No legal disposal center')
            cy,cx = map(int,rng.choice(center_candidates))
            spec['center'] = [cx,cy]
            mask,stats = validate(spec)
        except ValueError as error:
            reason = str(error)
            continue
        stats.update(room_count=rooms, topology_dead_ends=ends, attempt=attempt,
                     wall_thickness=thickness or DIFFICULTY[difficulty][0],
                     passage_width=spec['passage_width'])
        return spec,stats,mask
    raise LayoutExhaustedError(f'Cannot generate {family} with these parameters after 32 attempts: {reason}')


def preview(spec,mask,path):
    render_preview(spec,path)


def verify_native(data, scratch):
    # Reuse resolved venv paths. Windows locale-encoded .pth files can fail
    # during UTF-8 startup, while engine YAML files require UTF-8 when read.
    bootstrap = ("import runpy,sys; "
                 f"sys.path[:] = {sys.path!r}; sys.prefix={sys.prefix!r}; "
                 f"sys.exec_prefix={sys.exec_prefix!r}; "
                 f"runpy.run_module({(__package__+'.native_map')!r},run_name='__main__')")
    exe = [sys.executable,'-B','-S','-X','utf8','-c',bootstrap]
    env = dict(os.environ,SWARM_RESCUE_RESULT_PATH=str(scratch/'probe_artifacts'))
    result = subprocess.run([*exe,str(data.resolve())],cwd=Path(__file__).resolve().parents[6],
                            env=env,timeout=90,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Native probe failed: '+result.stdout.decode('utf-8',errors='replace')[-3500:])


def export(args):
    if args.count<1:
        raise ValueError('Count must be positive')
    out = args.output.resolve()
    repo = Path(__file__).resolve().parents[6]
    numbered = getattr(args,'save_official',False)
    start_index = getattr(args,'start_index',1)
    if numbered:
        numbered_name(start_index)
    if (out==repo or any(out.is_relative_to(repo/p) for p in
                         ('src','config','tests','.git','.venv','平台评测结果')) or
            (out.is_relative_to(repo/'map_data') and not (numbered and out==repo/'map_data'))):
        raise ValueError('Choose a separate dataset output directory')
    if numbered:
        for index in range(args.count):
            check_available(out,numbered_name(start_index+index))
    out.mkdir(parents=True,exist_ok=True)
    seen = set()
    if numbered:
        for record in (out/'.yh_records').glob('*/metadata.json'):
            seen.add(json.loads(record.read_text(encoding='utf-8'))['structure_sha256'])
    for index in range(args.count):
        seed = args.seed+index
        spec,stats,mask = build(args.type,args.width,args.height,args.difficulty,seed,
                                args.thickness,args.passage)
        geometry_digest = geometry_hash(spec)
        # A structural hash also catches identical walls with only spawn/center changed.
        structure = structure_hash(spec)
        name = numbered_name(start_index+index) if numbered else 'map_'+structure
        dest = out/(name+'.json') if numbered else out/name
        if dest.exists():
            raise FileExistsError(f'Geometry already present: {dest}')
        if numbered and structure in seen:
            raise DuplicateStructureError(f'Duplicate structure at seed {seed}; choose a new --seed')
        with tempfile.TemporaryDirectory(prefix='.g1-',dir=out) as temp:
            stage = Path(temp)/'map'
            stage.mkdir()
            data = stage/'map.json'
            document = editor_document(spec)
            data.write_text(json.dumps(document,indent=2)+'\n',encoding='utf-8')
            (stage/'generation.json').write_text(json.dumps(spec,indent=2,sort_keys=True)+'\n',encoding='utf-8')
            preview(spec,mask,stage/'preview.png')
            code = publish_sources(spec)
            verify_native(data,Path(temp))
            metadata = dict(generator_version=VERSION, seed=seed, family=args.type,
                            difficulty=args.difficulty,size=spec['size'],geometry_sha256=geometry_digest,
                            content_sha256=digest(spec),
                            export_format=FORMAT,official_editor_sha256=digest(document),
                            native_source_directory=code.relative_to(repo).as_posix(),
                            structure_sha256=structure,native_class='Generated'+geometry_digest,
                            native_load_verified=True, timestep_limit=spec.get('rules',{}).get('timestep_limit',5000),walltime_limit=180,
                            count_drones=5,bombs_per_drone=2,
                            blue_drones=spec.get('rules',{}).get('blue_drones',5),
                            parameters=dict(type=args.type,width=args.width,height=args.height,
                                            difficulty=args.difficulty,seed=seed,
                                            thickness=args.thickness,passage=args.passage), **stats)
            (stage/'metadata.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n',encoding='utf-8')
            if numbered:
                publish_numbered(stage,out,name)
            else:
                stage.rename(dest)
        seen.add(structure)
        print(dest,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--type',choices=FAMILIES,default='semifinal')
    p.add_argument('--difficulty',choices=DIFFICULTY,default='medium')
    p.add_argument('--width',type=int,default=1700)
    p.add_argument('--height',type=int,default=1100)
    p.add_argument('--thickness',type=int)
    p.add_argument('--passage',type=int)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--count',type=int,default=1)
    p.add_argument('--output',type=Path)
    p.add_argument('--save-official',action='store_true',
                   help='Save yh_map_fs_N.json and .png, default output is repository map_data')
    p.add_argument('--start-index',type=int,default=1,help='First numbered map, default 1')
    args = p.parse_args()
    if args.count<1:
        p.error('--count must be positive')
    if args.start_index<1:
        p.error('--start-index must be positive')
    if args.output is None:
        if args.save_official:
            args.output = Path(__file__).resolve().parents[6]/'map_data'
        else:
            p.error('--output is required unless --save-official is used')
    try:
        export(args)
    except (ValueError,OSError,RuntimeError,subprocess.TimeoutExpired) as error:
        p.exit(1,str(error)+'\n')


if __name__=='__main__':
    main()
