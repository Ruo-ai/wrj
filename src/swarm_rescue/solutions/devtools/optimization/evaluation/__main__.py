"""G3 evaluator: default advanced2, training split, three rounds, serial 4 GB workers."""
import argparse
from pathlib import Path
import sys


def main():
    from .runner import run
    from ...evaluate_verified_revision import CONTROLLERS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path)
    from .scenarios import OFFICIAL
    parser.add_argument('--official-maps',nargs='+',choices=OFFICIAL)
    parser.add_argument('--map-names',nargs='+',help='Explicit official + existing training map names')
    parser.add_argument('--split',choices=['train','validation','test','extrapolation','hard'],default='train')
    parser.add_argument('--controllers',nargs='+',choices=CONTROLLERS,default=['advanced2'])
    from .identities import BLUE_CONTROLLERS
    parser.add_argument('--blue-variant',choices=BLUE_CONTROLLERS,default='adapted')
    parser.add_argument('--prepare-only',action='store_true',help='Freeze the explicit plan and source snapshot; do not start simulations')
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--timeout',type=float,default=240)
    parser.add_argument('--memory-bytes',type=int,default=4_000_000_000)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--new',action='store_true')
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='Only 8 steps, one train map, one round; never a valid score')
    args = parser.parse_args()
    if args.official_maps and args.dataset:
        parser.error('--official-maps and --dataset are mutually exclusive')
    if args.map_names and args.official_maps:
        parser.error('--map-names and --official-maps are mutually exclusive')
    if args.rounds<1 or args.timeout<=0 or args.memory_bytes<256_000_000 or (args.limit is not None and args.limit<1):
        parser.error('Invalid count, timeout, or memory limit')
    if len(set(args.controllers))!=len(args.controllers):
        parser.error('Controller selectors must be unique')
    if args.smoke:
        args.rounds,args.limit = 1,1
        args.split = 'train'
    try:
        if args.prepare_only:
            from .runner import create_or_resume
            output,plan=create_or_resume(args)
            print(f"Prepared {len(plan['schedule'])} red + up to {len(plan['schedule'])} blue phases: {output}")
        else:
            run(args)
    except KeyboardInterrupt:
        parser.exit(130,'Stopped; completed receipts retained. Repeat the same command to resume.\n')
    except (ValueError,OSError,KeyError,RuntimeError) as error:
        parser.exit(1,str(error)+'\n')


if __name__=='__main__':
    main()
