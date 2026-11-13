import numpy as np
from swarm_rescue.solutions.my_drone_blue_search import SemanticSearchGrid
from swarm_rescue.solutions.my_drone_blue_transport import TransportGrid


def test_lidar_free_space_is_not_automatically_semantic_search_coverage():
    g=SemanticSearchGrid((640,480));p=g.point((15,15))
    g.observe(p,0.,np.array([230.]),np.array([0.]),np.array([False]))
    assert g.free[g.cell(p+np.array([160.,0.]))]
    assert not g.searched[g.cell(p+np.array([160.,0.]))]
    assert g.searched[g.cell(p+np.array([60.,0.]))]


def test_occlusion_and_disabled_semantics_do_not_mark_hidden_area():
    g=SemanticSearchGrid((640,480));p=g.point((15,15))
    g.observe(p,0.,np.array([60.]),np.array([0.]),np.array([False]))
    assert not g.searched[g.cell(p+np.array([80.,0.]))]
    g.semantic_enabled=False;before=g.searched.copy()
    g.observe(p,0.,np.array([230.]),np.array([0.]),np.array([False]))
    np.testing.assert_array_equal(g.searched,before)


def test_search_coverage_requires_actual_received_message_and_is_idempotent():
    a=SemanticSearchGrid((320,256));b=SemanticSearchGrid((320,256))
    a.searched[3:6,4:9]=True
    assert not b.search_coverage().any()
    b.accept(1,10,a.snapshot());before=b.search_coverage()
    assert before.sum()==15
    a.searched[:]=False;b.accept(1,9,a.snapshot())
    np.testing.assert_array_equal(b.search_coverage(),before)


def test_collection_and_payload_routes_equal_compute_baseline():
    a=TransportGrid((640,480));b=SemanticSearchGrid((640,480))
    for g in (a,b):g.free[:]=True;g.evidence[:20,20]=8
    p,t=a.point((10,4)),a.point((10,35))
    np.testing.assert_array_equal(a.route(p,t),b.route(p,t))


def test_search_path_never_uses_unknown_or_disconnected_cells():
    g=SemanticSearchGrid((640,480));g.free[5:15,4:14]=True
    g.searched[5:10,4:14]=True
    path=g.route(g.point((7,7)),None,0,10)
    allowed,_=g.layers()
    assert path and all(allowed[g.cell(p)] for p in path)
