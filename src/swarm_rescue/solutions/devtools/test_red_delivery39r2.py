"""The estimator's relock must transform return paths with the same correction."""
from types import SimpleNamespace
import numpy as np
from swarm_rescue.solutions.my_drone_red_delivery39r2 import CorrectedWalkGraph39


def test_relock_rebases_old_path_and_rebuilds_spatial_index_without_touching_mapper():
    mapper=SimpleNamespace(recovery38=None)
    graph=CorrectedWalkGraph39(mapper)
    graph.points=[np.array([0.,0.]),np.array([30.,0.]),np.array([60.,0.])]
    graph.edges=[{1:30.},{0:30.,2:30.},{1:30.}]
    graph.birth_steps=[1,10,20]
    graph.pose_history={10:np.array([30.,0.]),20:np.array([60.,0.])}
    correction={10:(np.array([30.,20.]),0.),20:(np.array([60.,40.]),0.)}
    graph.apply_recovery(correction,21)
    np.testing.assert_array_equal(graph.points[0],[0,0])
    np.testing.assert_array_equal(graph.points[1],[30,20])
    np.testing.assert_array_equal(graph.points[2],[60,40])
    assert 2 in graph.near(np.array([60,40]),5)
    assert graph.edges[0][1]>30 and mapper.recovery38 is None
