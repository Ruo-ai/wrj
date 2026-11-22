"""Evidence-only navigation cleanup; no simulator or map truth required."""
import numpy as np

from swarm_rescue.solutions.my_drone_red_advanced2 import _M19_SurveyGrid
from swarm_rescue.solutions.red_navigation32 import EvidenceNavigation32
from swarm_rescue.solutions.my_drone_red_mapping import MappingSnapshots


def grid():
    return EvidenceNavigation32((400, 400), lambda *a, **k: None)


def scan(g, step, x, distances=(180.,), angles=(0.,), dynamic=None):
    g.step = step
    g.observe(np.array([float(x), 0.]), 0., np.array(distances), np.array(angles), dynamic)


def occupied(g, point=(70., 0.)):
    cell = g.cell(point)
    g.obstacle[cell] = 6.
    return cell


def test_three_separated_recent_views_clear_residual_after_parent_updates():
    g = grid(); cell = occupied(g)
    for s, x in ((6, -40), (12, -20)):
        scan(g, s, x)
        assert g.obstacle[cell] >= 2
    scan(g, 18, 0)
    assert g.obstacle[cell] == 0 and g.free[cell]
    assert g.stats['cleared_occupied_cells'] == 1
    assert g.cleanup_events32[0]['previous_evidence'] == 4.5


def test_stationary_or_nearby_scans_do_not_accelerate_parent_decay():
    g = grid(); b = _M19_SurveyGrid((400,400), lambda *a,**k:None)
    cell = occupied(g); occupied(b)
    for s, x in ((6,0),(12,1),(18,0),(24,3)):
        scan(g,s,x); scan(b,s,x)
    np.testing.assert_array_equal(g.obstacle, b.obstacle)
    assert not g.stats['cleared_occupied_cells']


def test_duplicate_step_cannot_add_independent_evidence():
    g = grid(); cell=occupied(g)
    for x in (-40,-20,0): scan(g,6,x)
    assert g.obstacle[cell] == 4.5 and g._views32[cell] == 1


def test_duplicate_frame_endpoint_still_resets_prior_free_evidence():
    g=grid();cell=occupied(g)
    scan(g,6,-40);scan(g,12,-20)
    scan(g,12,-20,distances=(90.,))
    assert g._views32[cell] == 0 and g._last_hit32[cell] == 12


def test_current_endpoint_wins_over_crossing_free_ray():
    g = grid(); cell=occupied(g)
    scan(g,6,-40); scan(g,12,-20)
    scan(g,18,0,distances=(180.,70.),angles=(0.,0.))
    assert g.obstacle[cell] >= 2 and g._views32[cell] == 0
    assert g._last_hit32[cell] == 18


def test_recent_hit_restarts_evidence_and_has_holdoff():
    g=grid();cell=occupied(g)
    scan(g,6,-40,distances=(110.,))
    scan(g,12,-20);scan(g,18,0)
    assert g._views32[cell] == 0 and g.obstacle[cell] >= 2
    for s,x in ((24,-40),(30,-20),(36,0)):scan(g,s,x)
    assert g.obstacle[cell] == 0


def test_persistent_single_cell_short_wall_is_preserved():
    g=grid();cell=occupied(g)
    for s,x in ((6,-40),(12,-20),(18,0),(24,20)):
        scan(g,s,x,distances=(70.-x,))
    assert g.obstacle[cell] == 6 and not g.stats['cleared_occupied_cells']


def test_dynamic_rays_provide_no_additional_cleanup_evidence():
    g=grid();cell=occupied(g)
    for s,x in ((6,-40),(12,-20),(18,0)):scan(g,s,x,dynamic=np.array([True]))
    assert g.obstacle[cell] == 4.5 and g._views32[cell] == 0


def test_no_cleanup_or_known_space_behind_dynamic_occluder():
    g=grid();cell=occupied(g)
    for s,x in ((6,-40),(12,-20),(18,0)):
        scan(g,s,x,distances=(40.,),dynamic=np.array([True]))
    assert g.obstacle[cell] == 6 and not g.free[cell]


def test_unknown_cells_are_not_promoted_by_cleanup():
    g=grid();cell=g.cell((70.,100.))
    for s,x in ((6,-40),(12,-20),(18,0)):scan(g,s,x)
    assert not g.free[cell] and g.obstacle[cell] == 0 and g._views32[cell] == 0


def test_old_views_expire_instead_of_accumulating_forever():
    g=grid();cell=occupied(g)
    scan(g,6,-40);scan(g,12,-20);scan(g,150,0)
    assert g.obstacle[cell] == 4.5 and g._views32[cell] == 1


def test_original_endpoint_margin_protects_near_wall_band():
    g=grid();cell=occupied(g,(90.,0.))
    for s,x in ((6,-40),(12,-20),(18,0)):
        scan(g,s,x,distances=(100.-x,))
    assert g.obstacle[cell] == 6 and not g.stats['cleared_occupied_cells']


def test_original_planner_and_safety_inflation_are_inherited_unchanged():
    assert EvidenceNavigation32.plan is _M19_SurveyGrid.plan
    assert EvidenceNavigation32.line_clear is _M19_SurveyGrid.line_clear
    g=grid();g.free[:]=True;g.obstacle[g.cell((10.,10.))]=6
    assert not g.line_clear(np.array([-70.,10.]),np.array([70.,10.]))


def test_cleanup_does_not_touch_submitted_maps_or_radio_payloads():
    g=grid();cell=occupied(g)
    class Mapper:
        def __init__(self):self.value=np.ones(g.shape,bool)
        def mask(self):return self.value.copy()
    mapper=Mapper();store=MappingSnapshots(g.shape,5,mapper)
    store.publish(0,1,g.obstacle>=2)
    snapshot=store.snapshots[0];mapped=store.mapping_masks[0].copy();nav=store.masks[0].copy()
    for s,x in ((6,-40),(12,-20),(18,0)):scan(g,s,x)
    assert g.obstacle[cell] == 0 and store.snapshots[0] == snapshot
    np.testing.assert_array_equal(store.mapping_masks[0],mapped)
    np.testing.assert_array_equal(store.masks[0],nav)
    assert mapper.value.all() and not np.shares_memory(g.obstacle,store.masks[0])
