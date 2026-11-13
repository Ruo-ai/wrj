import pytest as _retired_pytest
_retired_pytest.importorskip('swarm_rescue.solutions.my_drone_red_wall_margin', reason='Historical strategy retired by user; source archived')
import numpy as np
import pytest

from swarm_rescue.solutions.my_drone_red_wall_margin import WallMarginField


@pytest.mark.parametrize('placed,claims', [([], []), ([(0, 0)], []), ([], [(0, 0)]),
                                           ([], [(0, 0), (0, 0)])])
def test_coverage_scenarios(placed, claims):
    walls = np.zeros((31, 31), np.uint8)
    walls[:, 15] = 1
    field = WallMarginField(walls, 4, (124, 124), placed, claims, 20, .5)
    gross = float(field.standalone[field.disk((0, 0))].sum())
    expected = 0 if placed else (.325 * gross if claims else gross)
    assert gross > 0
    assert field.value((0, 0)) == pytest.approx(expected)
    assert field.sampled_value((0, 0)) == pytest.approx(expected, abs=1e-4)

