"""Build a red-only RL17-R archive without changing the workspace entry or old ZIP.

The reference archive supplies the flat layout, team metadata and requirements.
Controller files are byte-identical to the R20 frozen sources. Verification
loads solutions modules directly from the ZIP and forbids workspace fallback.
"""
import argparse
import ast
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

REPO=Path(__file__).resolve().parents[4]
SOLUTIONS=REPO/'src/swarm_rescue/solutions'
PREFIX='swarm_rescue.solutions.'
ENTRY=b'''"""Team 742 red-stage submission: RL17-R."""
from swarm_rescue.simulation.reporting.team_mode import TeamMode
from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17


def drone_class_for_mode(mode: str):
    if TeamMode.from_string(mode) == TeamMode.PLACE:
        return MyDroneRedRepair17
    raise ValueError("This submission is for the red PLACE stage only.")
'''


def sha(data):return hashlib.sha256(data).hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)


def closure():
    todo=['my_drone_red_repair17'];files={};external=set()
    while todo:
        name=todo.pop();key=name.replace('.','/')+'.py'
        if key in files:continue
        source=(SOLUTIONS/key).read_bytes();tree=ast.parse(source,filename=key);files[key]=source
        for node in ast.walk(tree):
            modules=[]
            if isinstance(node,ast.Import):modules=[a.name for a in node.names]
            if isinstance(node,ast.ImportFrom):
                if node.level:raise ValueError('Unreviewed relative import: '+key)
                modules=[node.module or '']
            for module in modules:
                if module.startswith(PREFIX):todo.append(module[len(PREFIX):])
                elif module:external.add(module)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('exec','eval','__import__'):
                raise ValueError('Unreviewed dynamic execution/import: '+key)
    files['__init__.py']=(SOLUTIONS/'__init__.py').read_bytes()
    return files,sorted(external)


class ArchiveModules(importlib.abc.MetaPathFinder,importlib.abc.Loader):
    def __init__(self,path):
        self.path=path;self.loaded={}
        with zipfile.ZipFile(path) as archive:self.files={name:archive.read(name) for name in archive.namelist()}
    def find_spec(self,fullname,path=None,target=None):
        if not fullname.startswith(PREFIX):return None
        key=fullname[len(PREFIX):].replace('.','/')+'.py'
        if key not in self.files:raise ModuleNotFoundError('Missing ZIP dependency; workspace fallback forbidden: '+fullname)
        return importlib.util.spec_from_loader(fullname,self,origin=str(self.path)+'!/'+key)
    def create_module(self,spec):return None
    def exec_module(self,module):
        key=module.__name__[len(PREFIX):].replace('.','/')+'.py'
        module.__file__=str(self.path)+'!/'+key
        exec(compile(self.files[key],module.__file__,'exec'),module.__dict__)
        self.loaded[key]=sha(self.files[key])


def verify(path):
    finder=ArchiveModules(path);sys.meta_path.insert(0,finder)
    from swarm_rescue.solutions.my_drone_eval import drone_class_for_mode
    from swarm_rescue.simulation.reporting.team_mode import TeamMode
    from swarm_rescue.simulation.utils.misc_data import MiscData
    cls=drone_class_for_mode('place')
    assert cls.__module__==PREFIX+'my_drone_red_repair17' and cls.__name__=='MyDroneRedRepair17'
    instances=[cls(identifier=i,misc_data=MiscData(size_area=(1600,1200),number_drones=5,
        max_timestep_limit=5400,max_walltime_limit=180,team_mode=TeamMode.PLACE)) for i in range(5)]
    assert all(type(d._submission_mapper).__name__=='RepairMapper17' for d in instances)
    try:drone_class_for_mode('rescue')
    except ValueError:blue_rejected=True
    else:blue_rejected=False
    assert blue_rejected
    for key in finder.files:
        if key.endswith('.py') and key!='__init__.py':
            assert key in finder.loaded,('Unused/unverified bundled module',key)
    result=dict(controller=cls.__module__+'.'+cls.__name__,mapper=type(instances[0]._submission_mapper).__name__,
        constructor_instances=len(instances),red_only=blue_rejected,loaded_from_zip=finder.loaded,
        workspace_solution_fallback_allowed=False,full_flights=0)
    write(path.parent/'import_verification.json',result)
    print('ZIP entry, dependency closure and five red instances verified; no flights.')


def build(out):
    out.mkdir(parents=True,exist_ok=False)
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=REPO).decode('utf-8').split('\0')
    protected={name:sha((REPO/name).read_bytes()) for name in names if name and (REPO/name).is_file()}
    write(out/'protected_before.json',protected)
    reference=REPO/'team742_preliminary.zip';old_hash=sha(reference.read_bytes());files,external=closure()
    frozen=json.loads((REPO/'output/sf13/r20a_20261117_150615/plan.json').read_text('utf-8'))['source_sha256']
    for name,data in files.items():
        key='src/swarm_rescue/solutions/'+name
        if name!='__init__.py':assert frozen[key]==sha(data),('Source differs from verified R20',name)
    source_hashes={name:sha(data) for name,data in files.items()}
    with zipfile.ZipFile(reference) as old:
        reference_names=old.namelist()
        for name in ('team_info.yml','req.txt'):files[name]=old.read(name)
    import yaml
    assert yaml.safe_load(files['team_info.yml'])['team_number']==742
    assert yaml.safe_load(files['team_info.yml'])==yaml.safe_load((SOLUTIONS/'team_info.yml').read_bytes())
    assert files['req.txt']==(REPO/'req.txt').read_bytes()
    files['my_drone_eval.py']=ENTRY
    assert not any('/' in name or 'blue' in name or 'rescue_example' in name for name in files)
    archive_path=out/'team742_preliminary.zip'
    with zipfile.ZipFile(archive_path,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,data in sorted(files.items()):archive.writestr(name,data)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert {name:archive.read(name) for name in archive.namelist()}==files
    from swarm_rescue.submission_check import validate_submission
    check=validate_submission(archive_path)
    write(out/'official_check.json',dict(ok=check.ok,errors=check.errors,warnings=check.warnings))
    if not check.ok:raise RuntimeError(check.errors)
    validation=subprocess.run([sys.executable,'-B','-m','swarm_rescue.solutions.devtools.package_red_rl17r_20261118',
        '--verify',str(archive_path)],cwd=REPO,capture_output=True)
    (out/'import_check.stdout.log').write_bytes(validation.stdout);(out/'import_check.stderr.log').write_bytes(validation.stderr)
    if validation.returncode:raise RuntimeError('Archive import verification failed; see retained logs')
    changed=[name for name,h in protected.items() if not (REPO/name).is_file() or sha((REPO/name).read_bytes())!=h]
    assert not changed,changed
    assert sha(reference.read_bytes())==old_hash
    manifest=dict(scope='Red PLACE only; RL17-R current empirical candidate, not established universally best.',
        created_client_date='2026-11-18',head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        reference_zip=str(reference),reference_sha256=old_hash,reference_entries=reference_names,
        zip=str(archive_path),zip_sha256=sha(archive_path.read_bytes()),zip_bytes=archive_path.stat().st_size,
        entries={name:dict(bytes=len(data),sha256=sha(data)) for name,data in sorted(files.items())},
        source_sha256=source_hashes,external_imports=external,official_check=dict(ok=check.ok,errors=check.errors,warnings=check.warnings),
        protected_files=len(protected),protected_changes=changed,workspace_entry_changed=False,full_flights=0)
    write(out/'package_manifest.json',manifest)
    with (out/'SHA256.txt').open('x',encoding='ascii') as stream:stream.write(manifest['zip_sha256']+'  '+archive_path.name+'\n')
    print('READY',archive_path,'bytes',manifest['zip_bytes'],'files',len(files),'sha256',manifest['zip_sha256'])
    for warning in check.warnings:print('Official warning:',warning)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);parser.add_argument('--verify',type=Path)
    args=parser.parse_args()
    if args.verify:return verify(args.verify.resolve())
    if args.output:return build(args.output.resolve())
    parser.error('--output or --verify required')


if __name__=='__main__':main()
