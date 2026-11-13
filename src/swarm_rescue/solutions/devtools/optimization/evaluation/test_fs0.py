"""FS0 identity, score validity and atlas regression checks."""
from pathlib import Path
import pytest

from . import identities, runner, atlas
from .test_evaluation import pair, frozen_plan


def test_reference_registry_and_unimplemented_candidates():
    from swarm_rescue.solutions.my_drone_eval import drone_class_for_mode
    assert identities.load_blue('baseline') is identities.load_blue('adapted')
    assert drone_class_for_mode('rescue').__name__=='MyDroneRescueExample'
    for candidate in ('N1','B1'):
        assert identities.catalog()[candidate]['selector'] is None
        with pytest.raises(ValueError):
            runner.ev.controller_identity(candidate)


def test_frozen_plan_pins_both_controller_identities(frozen_plan):
    args,_=frozen_plan
    _,plan=runner.create_or_resume(args)
    assert plan['configuration']['blue_controller_class']=='MyDroneBlueAdapted'
    assert len(plan['configuration']['controller_identities'])==2
    assert not (args.output/'runs').exists()  # preparation cannot start workers
    args.blue_variant='enhanced'
    with pytest.raises(ValueError,match='mismatch'):
        runner.create_or_resume(args)


def test_atlas_retains_raw_invalid_score_and_all_images(tmp_path):
    item=pair(tmp_path)
    red=Path(item['red_run']);blue=Path(item['blue_run'])
    (red/'artifacts').mkdir(parents=True);blue.mkdir()
    for name in ('sample_exploration_diff.png','sample_wall_destruction.png'):
        (red/'artifacts'/name).write_bytes(b'fixture')
    layout=tmp_path/'layout.png';layout.write_bytes(b'fixture')
    runner.write(red/'request.json',dict(seed=11))
    runner.write(blue/'request.json',dict(seed=22))
    (blue/'run.log').write_text('Exception ignored from cffi callback',encoding='utf-8')
    original=repr(item)
    text='\n'.join(atlas.round_section(item,tmp_path,layout,historical=True))
    assert text.count('![')==3 and text.count('.\\replay_map.cmd')==2
    assert '红种子：11；蓝种子：22' in text and '有效完整成绩：False' in text
    assert '|100|83|未知/未产生|' in text
    assert '运输成功率' in text and '历史数据' in text
    assert repr(item)==original  # auditing never repairs old raw results


def test_actual_baseline_rules_and_dataset():
    from .fs0 import audit
    result=audit()
    assert result['team_number']==742 and result['resources']['max_steps']==5400
    assert result['dataset']['summary']['unique_count']==100
    assert result['score_configuration']['w_exploration']==.5
