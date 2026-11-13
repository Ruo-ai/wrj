"""Immutable, hash-bound manifests referencing existing map files."""
import json
from pathlib import Path
import tempfile

from ..map_generator.validate import digest
from .index import REPO, distribution, file_hash, path_name, resolve_path
from .split import make_split


SCHEMA = 1
DEFAULT_ROOT = REPO/'output/optimization/datasets'
SPLITS = ('train','validation','test','extrapolation','hard')


def configuration(ratios=(70,15,15),seed=2026,seed_block=10,near_threshold=.80,holdout_families=()):
    return dict(ratios=list(ratios),seed=seed,seed_block=seed_block,near_threshold=near_threshold,
                holdout_families=sorted(set(holdout_families)))


def next_output():
    index = 1
    while (DEFAULT_ROOT/f'dataset_{index:03d}').exists():
        index += 1
    return DEFAULT_ROOT/f'dataset_{index:03d}'


def latest_output():
    found = sorted(p for p in DEFAULT_ROOT.glob('dataset_*') if p.name.removeprefix('dataset_').isdigit()
                   and p.is_dir() and (p/'dataset.json').is_file())
    if not found:
        raise ValueError('No frozen dataset found; run prepare_dataset.cmd first')
    return max(found,key=lambda p:int(p.name.removeprefix('dataset_')))


def assert_inputs_unchanged(rows):
    for row in rows:
        for name,expected in row['files'].items():
            path = resolve_path(name)
            if not path.is_file() or file_hash(path)!=expected:
                raise ValueError(f'Source file changed or disappeared: {path}')


def summary(rows,result,config):
    by_id = {r['id']:r for r in rows}
    primary_count = sum(len(result['bins'][s]) for s in SPLITS[:4])
    ordinary = sum(len(result['bins'][s]) for s in SPLITS[:3])
    return dict(indexed_count=len(rows),unique_count=primary_count,
                excluded_duplicate_count=len(rows)-primary_count,
                independent_groups=len(result['groups']),
                split_counts={s:len(result['bins'][s]) for s in SPLITS},
                achieved_ordinary_ratios={s:len(result['bins'][s])/ordinary if ordinary else 0 for s in SPLITS[:3]},
                requested_ratios=config['ratios'],target_counts=result['target'],
                distributions={s:distribution([by_id[i] for i in result['bins'][s]]) for s in SPLITS},
                overall_distribution=distribution(rows),warnings=result['warnings'],
                extrapolation_ready=bool(result['bins']['extrapolation']),
                hard_definition='difficulty=hard AND primary split is test or extrapolation')


def markdown_report(info,dataset_id):
    lines = ['# G2 数据集划分报告','',f'数据集 ID：`{dataset_id}`','',
             f"索引 {info['indexed_count']} 张，保留 {info['unique_count']} 张，重复别名 {info['excluded_duplicate_count']} 张。",
             f"独立分组 {info['independent_groups']} 个。按组划分，比例允许偏离目标。",'',
             '|集合|数量|','|---|---:|']
    lines += [f'|{s}|{n}|' for s,n in info['split_counts'].items()]
    lines += ['', 'hard 是 test/extrapolation 内的困难标签子集，不是可再用于训练的额外图池。',
              '原图留在原位置；清单保存文件哈希。新增地图不改变本快照，改动已收录文件会使校验失败。',
              '', '## 限制与使用边界','']
    lines += ['- '+message for message in info['warnings']]
    lines += ['', 'test、extrapolation、hard 用于冻结后的验收，不能用来挑参数。',
              '本阶段没有对打分数或录像，也没有自动启动 G3。',
              '完整分布见 summary.json；分组原因见 groups.json；重复关系见 duplicates.json。','']
    return '\n'.join(lines)


def freeze(rows,source,output,config):
    output = Path(output).resolve()
    if output==REPO or any(output.is_relative_to(REPO/p) for p in
                           ('src','map_data','config','tests','.git','.venv','平台评测结果')):
        raise ValueError('Choose a new dataset directory under output')
    if output.exists():
        raise FileExistsError(f'Dataset already exists; immutable snapshots are never overwritten: {output}')
    rows = sorted(rows,key=lambda r:r['map_path'])
    result = make_split(rows,**config)
    info = summary(rows,result,config)
    snapshot = dict(schema=SCHEMA,source=path_name(source),maps=rows)
    dataset_id = digest(dict(schema=SCHEMA,snapshot=snapshot,configuration=config))
    by_id = {r['id']:r for r in rows}
    membership = {i:split for split in SPLITS[:4] for i in result['bins'][split]}
    files = {'index.json':snapshot,'summary.json':info,'duplicates.json':result['duplicates'],
             'groups.json':dict(groups=result['groups'],assignment=result['assignment'],edges=result['edges'])}
    for split in SPLITS:
        files[split+'.json'] = dict(schema=SCHEMA,dataset_id=dataset_id,split=split,
            usage='tuning' if split in ('train','validation') else 'locked_evaluation',
            maps=[dict(id=i,map_path=by_id[i]['map_path'],primary_split=membership[i],
                       content_sha256=by_id[i]['content_sha256'],geometry_sha256=by_id[i]['geometry_sha256'])
                  for i in result['bins'][split]])
    assert_inputs_unchanged(rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.g2-',dir=output.parent) as tmp:
        stage = Path(tmp)/'dataset'
        stage.mkdir()
        for name,data in files.items():
            (stage/name).write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
        (stage/'report.md').write_text(markdown_report(info,dataset_id),encoding='utf-8')
        code = {path_name(p):file_hash(p) for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')}
        envelope = dict(schema=SCHEMA,dataset_id=dataset_id,configuration=config,
                        files={p.name:file_hash(p) for p in sorted(stage.iterdir())},manager_code=code)
        (stage/'dataset.json').write_text(json.dumps(envelope,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
        # Validate the exported manifests and current source files before publication.
        verify(stage)
        if output.exists():
            raise FileExistsError(f'Dataset destination was created by another process: {output}')
        stage.rename(output)
    return info


def verify(folder):
    folder = Path(folder).resolve()
    envelope = json.loads((folder/'dataset.json').read_text(encoding='utf-8'))
    if envelope['schema']!=SCHEMA:
        raise ValueError('Unsupported dataset schema')
    required = {'index.json','summary.json','duplicates.json','groups.json','report.md',*[s+'.json' for s in SPLITS]}
    if set(envelope['files'])!=required:
        raise ValueError('Dataset file manifest is incomplete')
    for name,expected in envelope['files'].items():
        if file_hash(folder/name)!=expected:
            raise ValueError(f'Dataset artifact changed: {name}')
    snapshot = json.loads((folder/'index.json').read_text(encoding='utf-8'))
    config = envelope['configuration']
    expected_id = digest(dict(schema=SCHEMA,snapshot=snapshot,configuration=config))
    if envelope['dataset_id']!=expected_id:
        raise ValueError('Dataset identity mismatch')
    rows = snapshot['maps']
    assert_inputs_unchanged(rows)
    # Recompute grouping and assignment, rather than trusting copied split lists.
    result = make_split(rows,**config)
    primary = {i:s for s in SPLITS[:4] for i in result['bins'][s]}
    for split in SPLITS:
        saved = json.loads((folder/(split+'.json')).read_text(encoding='utf-8'))
        if (saved['dataset_id']!=expected_id or saved['split']!=split or
                saved['usage']!=('tuning' if split in ('train','validation') else 'locked_evaluation') or
                [r['id'] for r in saved['maps']]!=result['bins'][split]):
            raise ValueError(f'Invalid split membership: {split}')
        by_id = {r['id']:r for r in rows}
        for r in saved['maps']:
            original = by_id[r['id']]
            if any(r[k]!=original[k] for k in ('map_path','content_sha256','geometry_sha256')):
                raise ValueError(f'Map reference changed in {split}')
            if r['primary_split']!=primary[r['id']]:
                raise ValueError(f'Primary split changed in {split}')
    saved_groups = json.loads((folder/'groups.json').read_text(encoding='utf-8'))
    if saved_groups!=dict(groups=result['groups'],assignment=result['assignment'],edges=result['edges']):
        raise ValueError('Grouping evidence mismatch')
    if json.loads((folder/'duplicates.json').read_text(encoding='utf-8'))!=result['duplicates']:
        raise ValueError('Duplicate evidence mismatch')
    info = summary(rows,result,config)
    if json.loads((folder/'summary.json').read_text(encoding='utf-8'))!=info:
        raise ValueError('Summary disagrees with split memberships')
    return info
