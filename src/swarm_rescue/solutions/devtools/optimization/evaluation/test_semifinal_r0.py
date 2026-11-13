"""R0 conditions and replay rejection checks, no formal strategy scores."""
from pathlib import Path
import pytest
from . import scenarios
from ...evaluate_red import evaluation_config,MAPS


def test_official_catalog():
    rows=scenarios.official_rows(['Map06','Map07t4','Map08','Map09'])
    assert len(rows)==4
    expected=[['NO_COM_ZONE'],[],['NO_GPS_ZONE'],['NO_COM_ZONE','NO_GPS_ZONE']]
    for row,zones in zip(rows,expected):
        assert scenarios.scenario(scenarios.REPO/row['map_path'],row['official_map'])['zones']==zones
    assert scenarios.engine_profile()['lidar']==230


def test_zones_in_both_phase_configs():
    for count in (5,10):
        cfg=evaluation_config(MAPS[0],drones=count,zones=['NO_GPS_ZONE'])['evaluation_plan'][0]
        assert cfg['number_drones']==count and cfg['zones_config']==['NO_GPS_ZONE']


def test_recording_rejects_wrong_region_or_count(monkeypatch):
    from swarm_rescue.simulation.reporting import state_replayer
    from types import SimpleNamespace
    metadata=dict(map_name='Map09',number_drones=10,zones_config=['NO_GPS_ZONE'],team_mode='rescue')
    monkeypatch.setattr(state_replayer,'StateReplayer',lambda _:SimpleNamespace(metadata=metadata))
    request=dict(map='Map09',drones=10,zones=['NO_GPS_ZONE'],phase='blue')
    assert scenarios.verify_recording(Path('unused'),request)==metadata
    for change in (dict(drones=5),dict(zones=[]),dict(phase='red')):
        with pytest.raises(ValueError):scenarios.verify_recording(Path('unused'),dict(request,**change))
