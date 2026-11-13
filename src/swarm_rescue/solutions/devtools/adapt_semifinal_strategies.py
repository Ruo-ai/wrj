"""One-time audited semifinal range migration; never touches the engine."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[4]
SOL=ROOT/'src/swarm_rescue/solutions'
OUT=ROOT/'output/semifinal_strategy_adaptation_20261110'
OUT.mkdir(exist_ok=True)
files=list(SOL.glob('my_drone_red*.py'))
archive=OUT/'before_adaptation.zip'
if not archive.exists():
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(ROOT))
    (OUT/'before_sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
for name in ('my_drone_red_advanced2.py','my_drone_red_advanced5_narrow.py'):
    p=SOL/name;s=p.read_text(encoding='utf-8')
    if '_SF_LIDAR_RANGE' in s:continue
    anchor='from dataclasses import dataclass\n'
    addition='''
from swarm_rescue.simulation.utils.constants import MAX_RANGE_LIDAR_SENSOR as _SF_LIDAR_RANGE
# Current engine lidar Gaussian sigma=2.5 px. Four sigma separates range returns
# conservatively; repeated evidence remains necessary, not a hard noise guarantee.
_SF_LIDAR_SIGMA = 2.5
_SF_HIT_MARGIN = 4.0 * _SF_LIDAR_SIGMA
_SF_HIT_LIMIT = _SF_LIDAR_RANGE - _SF_HIT_MARGIN

'''
    s=s.replace(anchor,anchor+addition)
    s=s.replace('max_range=300','max_range=_SF_LIDAR_RANGE')
    s=s.replace('max_range - 7','max_range - _SF_HIT_MARGIN')
    s=s.replace('d < 290','d < _SF_HIT_LIMIT').replace('d<290','d<_SF_HIT_LIMIT')
    s=s.replace('< 294','< _SF_HIT_LIMIT').replace('<294','<_SF_HIT_LIMIT')
    s=s.replace('np.arange(0, 281, 12)','np.arange(0, _SF_LIDAR_RANGE, 12)')
    s=s.replace('steps[None, :] < d[:, None] - 26','steps[None, :] < np.minimum(d[:, None], _SF_LIDAR_RANGE) - 26')
    s=s.replace('np.clip(d,0,294)','np.clip(d,0,_SF_LIDAR_RANGE)')
    # Narrow fan previously converted NaN/inf to invalid integer polygon vertices.
    s=s.replace('raw=np.clip(d,0,_SF_LIDAR_RANGE); safe=raw.copy()',
        'raw=np.clip(np.where(valid,d,0.),0,_SF_LIDAR_RANGE); safe=raw.copy()')
    s=s.replace('min_clearance=300.','min_clearance=float(_SF_LIDAR_RANGE)')
    # Existing filter already dead-reckons when GPS and compass are both disabled.
    # Track this condition for diagnostics; retain validated fusion coefficients.
    s=s.replace('        self.heading = None\n','        self.heading = None\n        self.absolute_missing_steps = 0\n',1)
    s=s.replace('        if self.position is None:\n',
        '        self.absolute_missing_steps = 0 if valid_gps and valid_compass else self.absolute_missing_steps + 1\n        if self.position is None:\n',1)
    p.write_text(s,encoding='utf-8')
