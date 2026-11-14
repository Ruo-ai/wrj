import numpy as np
from swarm_rescue.solutions.my_drone_red_entrance9 import EntranceHint,needs_precision
from swarm_rescue.solutions.red_pixel_faces import PixelFaceMapper
from swarm_rescue.solutions.red_pixel_faces_incremental import IncrementalPixelMapper
from swarm_rescue.solutions.devtools.test_advanced5_narrow import corridor


def test_entrance_hint_keeps_intended_direction_with_finite_expiry():
    h=EntranceHint();h.remember([0,-100],10)
    assert np.isclose(h.direction(np.zeros(2),0.,100),-np.pi/2)
    assert h.direction(np.zeros(2),0.,161) is None
    assert h.direction(np.array([0.,-99.]),0.,100) is None


def test_precise_controller_is_not_requested_for_wide_open_corridor():
    d,a=corridor(100)
    assert not needs_precision(d,a,0.)
    d,a=corridor(46)
    assert needs_precision(d,a,0.)


def test_incremental_outage_mask_matches_p8_bit_for_bit_including_recovery():
    a=PixelFaceMapper((300,200));b=IncrementalPixelMapper((300,200))
    d,angles=corridor(80,end=90)
    for tick in range(100):
        good=tick<12 or 56<=tick<68 or tick>90
        gps=np.array([tick*.4,0.]) if good else None;compass=.01 if good else None
        args=(gps,compass,np.array([.4,.02,.001]),d,angles)
        a.update(*args);b.update(*args)
        if tick%7==0 or tick in (55,56,90,91,99):
            np.testing.assert_array_equal(a.mask(),b.mask())
            np.testing.assert_array_equal(a.hits,b.hits)
            np.testing.assert_array_equal(a.support,b.support)
            np.testing.assert_array_equal(a.free_seen,b.free_seen)
    assert b.stats['provisional_frame_integrations']<40
