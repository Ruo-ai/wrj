"""FS0 baseline freeze and read-only verification. Never starts a simulation."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import zipfile

from .runner import REPO, read, write, file_hash, source_receipt, validate_sources, environment
from .identities import catalog, load_blue
from .scenarios import official_rows, engine_profile, scenario
from ..dataset_manager.artifacts import verify


def git(*args):
    return subprocess.check_output(['git', '-c', 'core.quotepath=false', *args],
                                   cwd=REPO, encoding='utf-8').strip()


def audit():
    from swarm_rescue.solutions.my_drone_eval import drone_class_for_mode
    from swarm_rescue.solutions import my_drone_eval_blue as blue
    from swarm_rescue.simulation.reporting.place_score_manager import PlaceScoreConfig
    dataset=REPO/'output/optimization/datasets/dataset_001'
    dataset_info=verify(dataset)
    assert dataset_info['split_counts']['train']==70
    assert dataset_info['split_counts']['validation']==15
    assert dataset_info['split_counts']['test']==15
    actual=dict(place=drone_class_for_mode('place').__name__,
                rescue=drone_class_for_mode('rescue').__name__,
                local_blue=load_blue('baseline').__name__,
                blue_entry=blue.drone_class_for_mode('rescue').__name__,
                blue_alias=blue.MyDroneEval.__name__)
    expected=dict(place='MyDroneRedAdvanced2',rescue='MyDroneRescueExample',
                  local_blue='MyDroneBlueAdapted',blue_entry='MyDroneBlueAdapted',
                  blue_alias='MyDroneBlueAdapted')
    if actual!=expected:
        raise ValueError('Entrypoints changed; review before freezing: '+str(actual))
    import yaml
    team=yaml.safe_load((REPO/'src/swarm_rescue/solutions/team_info.yml').read_text(encoding='utf-8'))
    if type(team['team_number']) is not int or team['team_number']!=742:
        raise ValueError('Expected integer team_number 742')
    maps=official_rows(['Map06','Map07t4','Map08','Map09'])
    score=asdict(PlaceScoreConfig())
    if [score[k] for k in ('w_exploration','w_wall','w_bomb','w_time')]!=[.5,.2,.2,.1]:
        raise ValueError('Official score weights changed')
    return dict(head=git('rev-parse','HEAD'),branch=git('branch','--show-current'),
        entries=actual,candidates=catalog(),engine_profile=engine_profile(),score_configuration=score,
        environment=environment(),dataset=dict(path=str(dataset),summary=dataset_info,
        manifest_sha256=file_hash(dataset/'dataset.json'),
        files={p.name:file_hash(p) for p in sorted(dataset.iterdir()) if p.is_file()}),
        scenarios={r['name']:scenario(REPO/r['map_path'],r['official_map']) for r in maps},
        resources=dict(cpu_count=1,memory_bytes=4_000_000_000,max_steps=5400,max_walltime=180,outer_timeout=240),
        team_number=742,package_name='team742_preliminary.zip',
        existing_packages={p.name:file_hash(p) for p in sorted(REPO.glob('*.zip'))},
        source_sha256=source_receipt(maps))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=REPO/'output/semifinal_fs0_20261113/baseline')
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(REPO/'output'):
        parser.error('FS0 artifacts must be inside repository output/')
    current=audit()
    if output.exists():
        frozen=read(output/'identity.json')
        validate_sources(frozen['source_sha256'])
        for key in ('entries','dataset','existing_packages','head','candidates'):
            if current[key]!=frozen[key]:
                raise ValueError('Frozen identity changed: '+key)
        if file_hash(output/'source_snapshot.zip')!=frozen['source_snapshot_sha256']:
            raise ValueError('Frozen archive changed')
        print('FS0 verified; 0 simulations; A0/N0/B0 unchanged:', output)
        return
    output.mkdir(parents=True)
    (output/'git_status.txt').write_text(git('status','--short')+'\n',encoding='utf-8')
    (output/'tracked_changes.patch').write_text(git('diff','--binary')+'\n',encoding='utf-8')
    with zipfile.ZipFile(output/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in current['source_sha256']:
            z.write(REPO/name,name)
    current.update(created_utc=datetime.now(timezone.utc).isoformat(),
                   source_snapshot_sha256=file_hash(output/'source_snapshot.zip'))
    write(output/'identity.json',current)
    print('FS0 frozen; 0 simulations; formal rescue remains example:',output)


if __name__=='__main__':
    main()
