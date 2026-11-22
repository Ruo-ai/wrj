"""Geometric invariants for the source-balanced submission fusion (no flights)."""
import unittest

import cv2
import numpy as np

from swarm_rescue.solutions.red_fusion27 import Fusion27


class Fusion27Tests(unittest.TestCase):
    SIZE = (660, 480)

    def mask(self, lines=()):
        result = np.zeros((160, 220), np.uint8)
        for start, end, width in lines:
            cv2.line(result, start, end, 1, width)
        return result.astype(bool)

    @staticmethod
    def widths(column):
        diff = np.diff(np.r_[0, column.astype(np.int8), 0])
        return (np.flatnonzero(diff == -1) - np.flatnonzero(diff == 1)).tolist()

    def test_offset_duplicate_bands_are_one_thin_wall(self):
        masks = {i:self.mask([((30, 70+i*2), (180, 70+i*2), 7)]) for i in range(3)}
        out = Fusion27(self.SIZE).fuse(masks)
        self.assertEqual(self.widths(out[:, 300]), [9])

    def test_lone_observer_wall_is_kept(self):
        mask = self.mask([((30, 70), (180, 70), 7)])
        f = Fusion27(self.SIZE)
        out = f.fuse({2:mask, 0:np.zeros_like(mask)})
        self.assertEqual(self.widths(out[:, 300]), [9])
        self.assertTrue(out[211, 300])
        self.assertGreater(f.last_stats['singleton_lines'], 0)

    def test_near_parallel_walls_preserve_passage(self):
        masks = {i:self.mask([((30, 60+i*2), (180, 60+i*2), 7),
                             ((30, 76+i*2), (180, 76+i*2), 7)]) for i in range(2)}
        out = Fusion27(self.SIZE).fuse(masks)
        self.assertEqual(self.widths(out[:, 300]), [9, 9])
        self.assertFalse(out[208, 300])

    def test_oblique_wall_does_not_snap_to_axis(self):
        masks = {i:self.mask([((30, 50+i*2), (180, 110+i*2), 7)]) for i in range(2)}
        f = Fusion27(self.SIZE)
        out = f.fuse(masks)
        self.assertTrue(out[244, 316])
        self.assertFalse(out[151, 316])
        self.assertTrue(any(abs(line['endpoints'][1][1]-line['endpoints'][0][1]) > 100 for line in f.last_lines))

    def test_junction_arms_survive(self):
        mask = self.mask([((30, 80), (180, 80), 7), ((100, 30), (100, 80), 7)])
        out = Fusion27(self.SIZE).fuse({0:mask})
        self.assertTrue(out[241, 121])
        self.assertTrue(out[241, 481])
        self.assertTrue(out[121, 301])
        self.assertTrue(out[241, 301])

    def test_l_corner_stays_connected(self):
        mask = self.mask([((40, 40), (120, 40), 7), ((120, 40), (120, 120), 7)])
        out = Fusion27(self.SIZE).fuse({0:mask})
        _, labels = cv2.connectedComponents(out.astype(np.uint8))
        self.assertGreater(labels[121, 181], 0)
        self.assertEqual(labels[121, 181], labels[301, 361])

    def test_short_unmodelled_wall_is_retained(self):
        mask = self.mask([((40, 50), (47, 50), 3)])
        f = Fusion27(self.SIZE)
        out = f.fuse({0:mask})
        self.assertEqual(f.last_stats['extracted_lines'], 0)
        self.assertTrue(out[151, 130])
        self.assertGreaterEqual(int(out.sum()), int(mask.sum())*9)

    def test_model_does_not_erase_neighbor_short_wall(self):
        mask = self.mask([((30, 60), (180, 60), 9), ((80, 76), (87, 76), 3)])
        out = Fusion27(self.SIZE).fuse({0:mask})
        self.assertTrue(out[229, 250])
        self.assertTrue(out[181, 250])
        self.assertFalse(out[205, 250])

    def test_model_does_not_erase_neighbor_box(self):
        wall = self.mask([((30, 60), (180, 60), 9)])
        box = self.mask(); box[76:87, 100:111] = True
        out = Fusion27(self.SIZE).fuse({0:wall, 1:box})
        self.assertTrue(out[228:261, 300:333].all())
        self.assertFalse(out[210, 310])

    def test_small_solid_box_is_not_skeletonized(self):
        mask = self.mask()
        mask[40:51, 60:71] = True
        out = Fusion27(self.SIZE).fuse({0:mask})
        self.assertEqual(int(out.sum()), 11*11*9)

    def test_large_solid_box_interior_is_retained(self):
        mask = self.mask()
        mask[40:100, 60:140] = True
        out = Fusion27(self.SIZE).fuse({0:mask})
        self.assertEqual(int(out.sum()), 60*80*9)

    def test_hollow_box_has_closed_outline(self):
        mask = self.mask([((50, 40), (150, 40), 7), ((150, 40), (150, 120), 7),
                          ((150, 120), (50, 120), 7), ((50, 120), (50, 40), 7)])
        out = Fusion27(self.SIZE).fuse({0:mask})
        self.assertTrue(out[121, 301])
        self.assertTrue(out[241, 151])
        self.assertFalse(out[241, 301])
        _, labels = cv2.connectedComponents((~out).astype(np.uint8))
        self.assertNotEqual(labels[241, 301], labels[10, 10])

    def test_packet_order_cannot_change_fusion(self):
        masks = {i:self.mask([((30, 70+i), (180, 70+i), 7)]) for i in range(3)}
        f = Fusion27(self.SIZE)
        first = f.fuse(masks)
        second = f.fuse(dict(reversed(list(masks.items()))))
        np.testing.assert_array_equal(first, second)

    def test_received_masks_are_immutable(self):
        mask = self.mask([((30, 70), (180, 70), 7)])
        before = mask.copy()
        Fusion27(self.SIZE).fuse({0:mask})
        np.testing.assert_array_equal(mask, before)

    def test_empty_map_and_small_unsupported_island(self):
        f = Fusion27(self.SIZE)
        self.assertFalse(f.fuse({}).any())
        mask = self.mask(); mask[80, 80] = True
        self.assertFalse(f.fuse({0:mask}).any())

    def test_malformed_input_cannot_be_treated_as_evidence(self):
        f = Fusion27(self.SIZE)
        with self.assertRaises(ValueError): f.fuse({0:np.zeros((2,2),bool)})
        with self.assertRaises(ValueError): f.fuse({0:self.mask().astype(float)})
        with self.assertRaises(ValueError): f.fuse({0:self.mask()}, {0:100})


if __name__ == '__main__':
    unittest.main()
