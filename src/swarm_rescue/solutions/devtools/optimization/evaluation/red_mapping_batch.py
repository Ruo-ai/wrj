"""Four-map A0/A3 mapping screen against fixed B0, explicitly user launched."""
import argparse
from pathlib import Path
from .joint_batch import arguments
from . import runner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=runner.REPO/'output/sf13/map_p1')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    task=arguments(args.output.resolve())
    task.controllers=['advanced2','advanced2_mapping']
    if args.prepare_only:
        runner.create_or_resume(task)
        print('Prepared',task.output)
    else:
        runner.run(task)


if __name__=='__main__': main()
