"""Read-only historical RN7 repair replay; never launches a simulation.

Only saved legal free/hit/clearance fields enter route(). Historical truth is
not loaded. Successful replay proves endpoint semantics, not physical crossing.
"""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import textwrap

import numpy as np

from swarm_rescue.solutions.my_drone_red_advanced5_narrow import NarrowLocalMap
from swarm_rescue.solutions.my_drone_red_rn7 import LocalPassage, PassageMotion
from swarm_rescue.solutions.my_drone_red_exploration22 import LocalPassage22, PassageMotion22


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'output/sf13/r21d_20261118_095350_321/runs/yh_map_fs_72_r02/rl17_repair/red/attempt_001'
EXPECTED_HASHES = {
    'gate21_diagnostics.json': '09fa746f88294e160a465b992313d94fd4216aae3812b6474b2280c41842bdd6',
    'gate21_diagnostics.npz': '7106500f13fdc35eef4911d08e4d59b197a54fec2dd920b911df60ce3d1dd901',
}
CASES = ((3, 296), (2, 554), (1, 1133))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases(source=SOURCE):
    source = Path(source)
    for name, expected in EXPECTED_HASHES.items():
        if digest(source / name) != expected:
            raise ValueError('Historical input hash differs: ' + name)
    document = json.loads((source / 'gate21_diagnostics.json').read_text(encoding='utf-8'))
    cases = []
    with np.load(source / 'gate21_diagnostics.npz', allow_pickle=False) as saved:
        for owner, step in CASES:
            matches = [p for p in document['probes'] if p['owner'] == owner and p['step'] == step
                       and p['kwargs'].get('target') is not None]
            if len(matches) != 1:
                raise ValueError('Expected exactly one repair probe for ' + str((owner, step)))
            probe = matches[0]
            fields = {name: saved[probe['arrays'] + '_' + name].copy()
                      for name in ('free', 'hits', 'clearance', 'squeeze')}
            cases.append((probe, fields))
    return cases


def restore(snapshot, cls):
    """Restore only the legal route inputs, preserving the recorded resolution."""
    probe, fields = snapshot
    fine = cls.__new__(cls)
    height, width = probe['shape']
    if probe['cell_size'] != cls.CELL:
        raise ValueError('Cell size differs from historical input')
    NarrowLocalMap.__init__(fine, (width * cls.CELL, height * cls.CELL))
    fine.free = fields['free'].copy()
    fine.hits = fields['hits'].copy()
    fine._clearance = fields['clearance'].copy()
    fine._squeeze = fields['squeeze'].copy()
    return fine


def replay_case(snapshot):
    probe, _ = snapshot
    original = np.asarray(probe['kwargs']['target'], float)
    if not np.array_equal(original, probe['preferred']):
        raise ValueError('Historical repair did not share target/preferred values')
    old = restore(snapshot, LocalPassage)
    old_target = original.copy()
    old_path = old.route(np.array(probe['position']), old_target, None,
                         target=old_target, limit=probe['kwargs']['limit'])
    fixed = restore(snapshot, LocalPassage22)
    target = original.copy()
    fixed_path = fixed.route(np.array(probe['position']), target, None,
                             target=target, limit=probe['kwargs']['limit'])
    match = bool(np.array_equal(np.asarray(old_path), np.asarray(probe['selected_path'])))
    safe = bool(fixed_path) and all(fixed.line_clear(a, b) for a, b in zip(fixed_path, fixed_path[1:]))
    endpoint_matches = bool(fixed_path) and fixed.cell(fixed_path[-1]) == fixed.cell(original)
    unchanged = bool(np.array_equal(target, original))
    return dict(owner=probe['owner'], tick=probe['step'], probe=probe['index'],
                target_before=original.tolist(), old_target_after=old_target.tolist(),
                old_path=[p.tolist() for p in old_path], fixed_path=[p.tolist() for p in fixed_path],
                old_matches_recorded=match, fixed_reason=fixed.last_reason,
                fixed_endpoint_matches_target_cell=endpoint_matches,
                fixed_endpoint_error_px=float(np.linalg.norm(fixed_path[-1] - original)) if fixed_path else None,
                fixed_all_segments_pass_original_clearance=safe, target_unchanged=unchanged,
                passed=match and safe and endpoint_matches and unchanged)


def executor_scope_check():
    old = ast.parse(textwrap.dedent(inspect.getsource(PassageMotion.command)))
    new = ast.parse(textwrap.dedent(inspect.getsource(PassageMotion22.command)))
    replaced = 0
    for node in ast.walk(new):
        if isinstance(node, ast.Name) and node.id == 'LocalPassage22':
            node.id = 'LocalPassage'
            replaced += 1
    return replaced == 2 and ast.dump(old) == ast.dump(new)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    cases = [replay_case(c) for c in load_cases(args.source)]
    scope_ok = executor_scope_check()
    if args.output is None:
        parent = ROOT / 'output/sf13'
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix='rl22_replay_', dir=parent))
    else:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    solution = ROOT / 'src/swarm_rescue/solutions'
    files = ['my_drone_red_exploration22.py', 'my_drone_red_rn7.py', 'my_drone_red_advanced5_narrow.py',
             'my_drone_red_repair17.py', 'my_drone_red_lines11.py']
    result = dict(schema='rl22-historical-repair-replay-v1', source=str(args.source.resolve()),
                  source_sha256={name: digest(args.source / name) for name in EXPECTED_HASHES},
                  code_sha256={name: digest(solution / name) for name in files},
                  executor_identical_except_two_constructors=scope_ok,
                  cases=cases, simulations_started=0,
                  passed=scope_ok and all(c['passed'] for c in cases),
                  limitations=['Saved legal route inputs only; no new flight, crossing or score result.',
                               'Map72 is historical defect evidence, excluded from the official batch.'])
    (output / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), passed=result['passed'], cases=len(cases), simulations_started=0),
                     ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
