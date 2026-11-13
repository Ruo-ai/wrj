"""G2 commands: inspect, build, verify, reproduce. No simulation rounds."""
import argparse
import json
from pathlib import Path

from ..map_generator.generate import build as build_map, export
from ..map_generator.validate import digest
from .index import REPO, build_index, distribution, index_map
from .split import group_maps
from .artifacts import configuration, freeze, next_output, latest_output, verify


def reproduction(data,output=None):
    row = index_map(data)
    params = row['parameters']
    spec,_,_ = build_map(params['type'],params['width'],params['height'],params['difficulty'],
                        params['seed'],params.get('thickness'),params.get('passage'))
    if digest(spec)!=row['content_sha256']:
        raise ValueError('Seed reproduction differs: restore the recorded generator/reference versions first')
    if output is not None:
        export(argparse.Namespace(**params,count=1,output=Path(output),save_official=False,start_index=1))
    return row['content_sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    inspect = commands.add_parser('inspect',help='Read-only index validation and grouping summary')
    inspect.add_argument('--source',type=Path,default=REPO/'map_data')
    build = commands.add_parser('build',help='Publish a new immutable grouped dataset manifest')
    build.add_argument('--source',type=Path,default=REPO/'map_data')
    build.add_argument('--output',type=Path)
    build.add_argument('--ratios',nargs=3,type=float,default=[70,15,15])
    build.add_argument('--seed',type=int,default=2026)
    build.add_argument('--seed-block',type=int,default=10)
    build.add_argument('--near-threshold',type=float,default=.80)
    build.add_argument('--holdout-family',action='append',default=[])
    check = commands.add_parser('verify',help='Verify a snapshot; defaults to latest numbered dataset')
    check.add_argument('folder',type=Path,nargs='?')
    reproduce = commands.add_parser('reproduce',help='Check seed reproduction; optional export into a new directory')
    reproduce.add_argument('map',type=Path)
    reproduce.add_argument('--output',type=Path)
    args = parser.parse_args()
    try:
        if args.command=='inspect':
            rows = build_index(args.source)
            groups,duplicates,edges = group_maps(rows)
            info = dict(distribution(rows),independent_groups=len(groups),
                        group_sizes=sorted(len(g['active']) for g in groups),
                        duplicate_aliases=sum(len(d['excluded']) for d in duplicates),
                        near_geometry_merges=sum(e['reason']=='near_geometry' for e in edges),
                        source_integrity='passed',simulation_run=False)
        elif args.command=='build':
            rows = build_index(args.source)
            config = configuration(args.ratios,args.seed,args.seed_block,args.near_threshold,args.holdout_family)
            target = args.output or next_output()
            info = freeze(rows,args.source,target,config)
            print('Saved:',target.resolve())
            info = {k:info[k] for k in ('indexed_count','unique_count','independent_groups',
                    'split_counts','extrapolation_ready','warnings')}
        elif args.command=='verify':
            info = verify(args.folder or latest_output())
            print('Verified dataset and source files')
            info = {k:info[k] for k in ('indexed_count','unique_count','split_counts','extrapolation_ready')}
        else:
            info = dict(reproduced_content_sha256=reproduction(args.map,args.output))
        print(json.dumps(info,ensure_ascii=True,sort_keys=True,indent=2))
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as error:
        parser.exit(1,str(error)+'\n')


if __name__=='__main__':
    main()
