"""Lightweight atlas regressions; no simulator, result mutation, or rendering."""
from copy import deepcopy

import numpy as np
import pytest

from swarm_rescue.solutions.devtools import red_p2l17_review as review


def trajectory_data():
    trace = np.zeros((2, 1), dtype=[('x', float), ('y', float), ('active', bool)])
    trace['active'] = True
    return {'trajectories': trace, 'w': 1700, 'h': 1100}


@pytest.mark.parametrize('raises', [False, True])
def test_map06_observation_audit_restores_regions(monkeypatch, raises):
    registry = review.evidence.REGIONS
    before = deepcopy(registry)
    assert 'Map06' not in registry
    called = []

    def audit(row, data):
        assert registry['Map06'] == {}
        # Call the actual imported region reader, including its module globals.
        assert review.evidence.regions(data, row['map']) == {}
        called.append(row['map'])
        if raises:
            raise ValueError('synthetic audit failure')

    monkeypatch.setattr(review.evidence, 'audit', audit)
    if raises:
        with pytest.raises(ValueError, match='synthetic audit failure'):
            review.observation_audit({'map': 'Map06'}, trajectory_data())
    else:
        review.observation_audit({'map': 'Map06'}, trajectory_data())
    assert called == ['Map06']
    assert review.evidence.REGIONS is registry
    assert registry == before
    assert 'Map06' not in registry


@pytest.mark.parametrize('raises', [False, True])
def test_existing_roi_preserved(monkeypatch, raises):
    registry = review.evidence.REGIONS
    before = deepcopy(registry)
    original_regions = registry['Map08']
    called = []

    def audit(row, data):
        assert registry['Map08'] is original_regions
        assert set(review.evidence.regions(data, row['map'])) == set(original_regions)
        called.append(row['map'])
        if raises:
            raise ValueError('synthetic existing-map failure')

    monkeypatch.setattr(review.evidence, 'audit', audit)
    if raises:
        with pytest.raises(ValueError, match='synthetic existing-map failure'):
            review.observation_audit({'map': 'Map08'}, trajectory_data())
    else:
        review.observation_audit({'map': 'Map08'}, trajectory_data())
    assert called == ['Map08']
    assert review.evidence.REGIONS is registry
    assert registry == before
    assert registry['Map08'] is original_regions


def test_no_outage_is_not_a_zero_error_measurement():
    row = {'full_outage_position': review.summary([]),
           'full_outage_heading_degrees': review.summary([])}
    text = review.outage_description(row)
    assert '无缺测样本' in text and '不适用' in text
    assert 'None' not in text and '0.0000' not in text
    assert row['full_outage_position']['mean'] is None
    assert row['full_outage_heading_degrees']['mean'] is None


def test_outage_measurements_keep_four_decimal_precision():
    row = {'full_outage_position': review.summary([1.23456, 1.23456]),
           'full_outage_heading_degrees': review.summary([7.89123])}
    text = review.outage_description(row)
    assert '1.2346' in text and '7.8912' in text
    assert 'px' in text and '°' in text
    assert '不适用' not in text
