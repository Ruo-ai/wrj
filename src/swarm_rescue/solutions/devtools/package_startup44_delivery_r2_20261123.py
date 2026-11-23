"""Package the frozen Startup44 Delivery R2; validation only, no simulations."""
import argparse
import ast
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[4]
SOL = ROOT / 'src/swarm_rescue/solutions'
PREFIX = 'swarm_rescue.solutions.'
TARGET = 'my_drone_red_startup44_delivery'
ENTRY = b'''"""Team 742 red-stage submission: Startup44 Delivery R2."""
from swarm_rescue.simulation.reporting.team_mode import TeamMode
from swarm_rescue.solutions.my_drone_red_startup44_delivery import MyDroneRedStartup44Delivery


def drone_class_for_mode(mode: str):
    if TeamMode.from_string(mode) == TeamMode.PLACE:
        return MyDroneRedStartup44Delivery
    raise ValueError("This submission is for the red PLACE stage only.")
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def closure():
    files = {}; external = set(); todo = [TARGET]
    while todo:
        name = todo.pop(); key = name.replace('.', '/') + '.py'
        if key in files:
            continue
        data = (SOL / key).read_bytes()
        tree = ast.parse(data, filename=key, feature_version=(3, 11))
        files[key] = data
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError('Relative import needs review: ' + key)
                modules = [node.module or '']
            for module in modules:
                if module.startswith(PREFIX):
                    todo.append(module[len(PREFIX):])
                elif module:
                    external.add(module)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ('eval', 'exec', '__import__', 'open'):
                    raise ValueError('Dynamic execution or runtime file dependency: ' + key)
                if isinstance(node.func, ast.Attribute) and node.func.attr in ('import_module', 'read_text', 'read_bytes', 'load', 'loadtxt'):
                    raise ValueError('Dynamic import or runtime file dependency: ' + key)
    files['__init__.py'] = (SOL / '__init__.py').read_bytes()
    return files, sorted(external)


def verify(stage, report):
    """Import only the actual unpacked submission, with no solution fallback."""
    installed = stage / 'src/swarm_rescue/solutions'
    import swarm_rescue
    assert not any(k.startswith(PREFIX) for k in sys.modules)
    package = 'swarm_rescue.solutions'
    spec = importlib.util.spec_from_file_location(package, installed / '__init__.py',
                                                 submodule_search_locations=[str(installed)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    setattr(swarm_rescue, 'solutions', module)
    spec.loader.exec_module(module)

    class NoFallback(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith(PREFIX):
                file = installed / (fullname[len(PREFIX):].replace('.', '/') + '.py')
                if not file.is_file():
                    raise ModuleNotFoundError('Missing bundled dependency; fallback forbidden: ' + fullname)
            return None

    sys.meta_path.insert(0, NoFallback())
    entry = importlib.import_module(PREFIX + 'my_drone_eval')
    cls = entry.drone_class_for_mode('place')
    assert cls.__name__ == 'MyDroneRedStartup44Delivery' and cls.__module__ == PREFIX + TARGET
    from swarm_rescue.simulation.reporting.team_mode import TeamMode
    from swarm_rescue.simulation.utils.misc_data import MiscData
    instances = [cls(identifier=i, misc_data=MiscData(size_area=(1600, 1200), number_drones=5,
                    max_timestep_limit=5400, max_walltime_limit=180, team_mode=TeamMode.PLACE))
                 for i in range(5)]
    assert all(hasattr(d, '_dl44') and hasattr(d, '_st44_plan') for d in instances)
    try:
        entry.drone_class_for_mode('rescue')
    except ValueError:
        red_only = True
    else:
        raise AssertionError('Unexpected blue entry')
    for file in sorted(installed.glob('*.py')):
        if file.stem != '__init__':
            importlib.import_module(PREFIX + file.stem)
    loaded = {}
    for name, mod in list(sys.modules.items()):
        if name == package or name.startswith(PREFIX):
            path = Path(mod.__file__).resolve()
            assert path.is_relative_to(installed.resolve()), (name, str(path))
            loaded[path.name] = sha(path.read_bytes())
    assert len(loaded) == len(list(installed.glob('*.py')))
    result = dict(controller=cls.__module__ + '.' + cls.__name__, red_only=red_only,
                  constructor_instances=len(instances), inherited_classes=[c.__name__ for c in cls.__mro__],
                  loaded_from_unpacked_zip=loaded, workspace_solution_fallback_allowed=False,
                  control_calls=0, simulations=0, python=sys.version)
    write(report, result)
    print('Unpacked ZIP: exact R2 entry, all Python modules, five constructors OK; no simulations.')


def check(target, log):
    process = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/check_submission.py'), str(target)],
                             cwd=ROOT, capture_output=True)
    log.write_bytes(process.stdout + process.stderr)
    return dict(exit_code=process.returncode, log=str(log))


def build(out):
    out.mkdir(parents=True, exist_ok=False)
    original = ROOT / 'team742_preliminary.zip'
    old = original.read_bytes(); old_hash = sha(old)
    backup = out / 'previous_rl17r' / original.name
    backup.parent.mkdir()
    backup.write_bytes(old)
    assert sha(backup.read_bytes()) == old_hash
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    protected = {n: sha((ROOT / n).read_bytes()) for n in names if n and (ROOT / n).is_file()}
    write(out / 'protected_before.json', protected)
    (out / 'git_status_before.txt').write_bytes(subprocess.check_output(['git', 'status', '--short'], cwd=ROOT))
    source, external = closure()
    frozen_evidence = []
    for batch in ('startup44_delivery_r2_20261123', 'startup44_deep_bomb_random3_20261123'):
        plan_path = ROOT / 'output/sf13' / batch / 'plan.json'
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        for name, data in source.items():
            assert plan['source_sha256']['src/swarm_rescue/solutions/' + name] == sha(data), (batch, name)
        frozen_evidence.append(dict(plan=str(plan_path), sha256=sha(plan_path.read_bytes()), matched=len(source)))
    source_hashes = {name: sha(data) for name, data in source.items()}
    files = dict(source)
    with zipfile.ZipFile(original) as archive:
        files.update({n: archive.read(n) for n in ('team_info.yml', 'req.txt')})
    import yaml
    info = yaml.safe_load(files['team_info.yml'])
    assert type(info['team_number']) is int and info['team_number'] == 742
    assert info == yaml.safe_load((SOL / 'team_info.yml').read_bytes())
    assert files['req.txt'] == (ROOT / 'req.txt').read_bytes()
    files['my_drone_eval.py'] = ENTRY
    assert not any('/' in n or '45' in n or 'deep_bomb' in n or 'opportunistic' in n or 'blue' in n for n in files)
    for name, data in files.items():
        if name.endswith('.py'):
            ast.parse(data, filename=name, feature_version=(3, 11))
    archive_path = out / original.name
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            entry = zipfile.ZipInfo(name, (2026, 11, 23, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data)
    stage = out / 'installed_workspace'
    installed = stage / 'src/swarm_rescue/solutions'
    installed.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(n.casefold() for n in archive.namelist()))
        assert {n: archive.read(n) for n in archive.namelist()} == files
        for name in archive.namelist():
            destination = stage / name if name == 'req.txt' else installed / name
            destination.write_bytes(archive.read(name))
    workspace_check = check(ROOT, out / 'workspace_static_check.log')
    installed_check = check(stage, out / 'installed_static_check.log')
    zip_check = check(archive_path, out / 'zip_static_check.log')
    assert installed_check['exit_code'] == zip_check['exit_code'] == 0, 'Final package checks failed'
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYGLET_SHADOW_WINDOW='false', PYTHONIOENCODING='utf-8')
    process = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--verify', str(stage),
                              '--report', str(out / 'import_verification.json')],
                             cwd=out, env=env, capture_output=True)
    (out / 'import_stdout.log').write_bytes(process.stdout)
    (out / 'import_stderr.log').write_bytes(process.stderr)
    assert process.returncode == 0, 'Isolated import/constructor validation failed'
    changed = [n for n, h in protected.items() if not (ROOT / n).is_file() or sha((ROOT / n).read_bytes()) != h]
    assert not changed, changed
    assert sha(original.read_bytes()) == old_hash
    manifest = dict(version='Startup44 Delivery R2', scope='red PLACE only', client_date='2026-11-23',
                    head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    archive=str(archive_path), sha256=sha(archive_path.read_bytes()), bytes=archive_path.stat().st_size,
                    files={n: dict(bytes=len(b), sha256=sha(b)) for n, b in sorted(files.items())},
                    frozen_source_checks=frozen_evidence, runtime_source_sha256=source_hashes, external_imports=external,
                    previous_archive_sha256=old_hash, previous_archive_backup=str(backup),
                    workspace_check=workspace_check, installed_check=installed_check, zip_check=zip_check,
                    isolated_import_exit_code=process.returncode, no_algorithm_changes=True,
                    workspace_entry_changed=False, protected_files=len(protected), protected_changes=changed,
                    simulations=0, published=False)
    write(out / 'package_manifest.json', manifest)
    (out / 'SHA256.txt').write_text(manifest['sha256'] + '  ' + original.name + '\n', encoding='ascii')
    print(json.dumps({k: manifest[k] for k in ('version', 'archive', 'sha256', 'bytes', 'workspace_check', 'installed_check', 'zip_check')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify.resolve(), args.report.resolve())
    elif args.output:
        build(args.output.resolve())
    else:
        parser.error('--output or --verify required')
