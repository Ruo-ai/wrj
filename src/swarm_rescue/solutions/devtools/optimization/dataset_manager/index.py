"""Read-only indexing of generated maps and their provenance."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

from ..map_generator.official_format import read_spec, check_sources
from ..map_generator.naming import record_directory
from ..map_generator.validate import digest, geometry_hash, structure_hash, validate


REPO = Path(__file__).resolve().parents[6]
SIGNATURE_VERSION = 'interior-64-dilate1-dihedral-v1'


def path_name(path):
    path = Path(path).resolve()
    return path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else path.as_posix()


def resolve_path(name):
    path = Path(name)
    return path if path.is_absolute() else REPO/path


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature_variants(spec):
    """Coarse tolerant interior geometry, excluding border and movable markers."""
    w,h = spec['size']
    mask = np.zeros((h,w),np.uint8)
    for _,l,t,r,b in spec['walls']:
        mask[t:b,l:r] = 1
    scaled = cv2.resize(mask.astype(np.float32),(64,64),interpolation=cv2.INTER_AREA)
    coarse = cv2.dilate((scaled>.08).astype(np.uint8),np.ones((3,3),np.uint8))
    result = set()
    for k in range(4):
        rotated = np.rot90(coarse,k)
        for image in (rotated,np.fliplr(rotated)):
            result.add(np.packbits(image.ravel()).tobytes().hex())
    return sorted(result)


def discover(source):
    source = Path(source).resolve()
    if not source.is_dir():
        raise ValueError(f'Map source directory does not exist: {source}')
    found = [p for p in source.glob('map_*_yh.json') if re.fullmatch(r'map_[0-9]+_yh\.json',p.name)]
    found.extend(p for p in source.glob('yh_map_fs_*.json') if re.fullmatch(r'yh_map_fs_[0-9]+\.json',p.name))
    found.extend(p for p in source.rglob('map.json')
                 if not any(part.startswith('.') for part in p.relative_to(source).parts))
    return sorted(set(found),key=lambda p:p.relative_to(source).as_posix())


def index_map(data):
    data = Path(data).resolve()
    record = record_directory(data)
    meta_path = record/'metadata.json'
    spec_path = record/'generation.json'
    png = data.with_suffix('.png') if data.name!='map.json' else data.parent/'preview.png'
    opening_hashes = {path_name(p):file_hash(p) for p in (data,png,meta_path,spec_path)}
    spec = read_spec(data)
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    document = json.loads(data.read_text(encoding='utf-8'))
    _,stats = validate(spec)
    expected = {'geometry_sha256':geometry_hash(spec),'structure_sha256':structure_hash(spec),
                'content_sha256':digest(spec),'official_editor_sha256':digest(document)}
    for key,value in expected.items():
        if meta.get(key)!=value:
            raise ValueError(f'{data}: {key} mismatch')
    for key,value in stats.items():
        if meta.get(key)!=value:
            raise ValueError(f'{data}: recomputed {key} disagrees with metadata')
    if meta.get('native_load_verified') is not True:
        raise ValueError(f'{data}: lacks successful G1 native-load receipt')
    params = meta['parameters']
    if (meta.get('seed')!=params.get('seed') or type(meta['seed']) is not int or
            meta['family']!=params.get('type') or meta['difficulty']!=params.get('difficulty') or
            spec['size']!=[params.get('width'),params.get('height')] or
            meta.get('generator_version')!=spec['version']):
        raise ValueError(f'{data}: generation parameters disagree with metadata')
    native = check_sources(spec)  # Compare text only; no generated Python execution.
    if meta.get('native_source_directory')!=path_name(native):
        raise ValueError(f'{data}: native source location mismatch')
    with Image.open(png) as image:
        if image.size!=tuple(spec['size']):
            raise ValueError(f'{png}: preview dimensions mismatch')
        image.verify()
    paths = [data,png,meta_path,spec_path,*[native/n for n in ('__init__.py','map_generated.py','walls_generated.py')]]
    receipts = {path_name(p):file_hash(p) for p in paths}
    if any(receipts[name]!=value for name,value in opening_hashes.items()):
        raise ValueError(f'{data}: source changed while being indexed')
    row = dict(id=digest([path_name(data),expected['content_sha256']]),name=data.stem,
               map_path=path_name(data),preview_path=path_name(png),
               files=receipts,**expected,
               seed=meta['seed'],family=meta['family'],difficulty=meta['difficulty'],
               parameters=params,size=spec['size'],generator_version=meta['generator_version'],
               layout_algorithm=meta.get('layout_algorithm','specialist-v2'),
               export_format=meta['export_format'],
               calibration_source=meta.get('calibration_source'),
               calibration_sha256=meta.get('calibration_sha256'),
               wall_thickness=meta['wall_thickness'],passage_width=meta['passage_width'],
               room_count=meta['room_count'],topology_dead_ends=meta['topology_dead_ends'],
               wall_fraction=round(stats['wall_area_px']/(spec['size'][0]*spec['size'][1]),6),
               signature_version=SIGNATURE_VERSION,signatures=signature_variants(spec),
               **stats)
    return row


def build_index(source):
    paths = discover(source)
    if not paths:
        raise ValueError('No generated maps found; official map_NN.json references are excluded')
    rows = []
    for path in paths:
        try:
            rows.append(index_map(path))
        except (ValueError,KeyError,OSError,TypeError) as error:
            raise ValueError(f'Index refused invalid/incomplete map {path}: {error}') from error
    return rows


def distribution(rows):
    def counts(key):
        return dict(sorted(Counter(str(r.get(key)) for r in rows).items()))
    return dict(count=len(rows),families=counts('family'),difficulties=counts('difficulty'),
                calibration_sources=counts('calibration_source'),algorithms=counts('layout_algorithm'),
                sizes=dict(sorted(Counter('x'.join(map(str,r['size'])) for r in rows).items())),
                wall_fraction_range=[min((r['wall_fraction'] for r in rows),default=0),
                                     max((r['wall_fraction'] for r in rows),default=0)],
                room_count_range=[min((r['room_count'] for r in rows if r['room_count'] is not None),default=0),
                                  max((r['room_count'] for r in rows if r['room_count'] is not None),default=0)])
