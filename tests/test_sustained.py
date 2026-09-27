"""Exercise layer limits, average qualification, and sustained voice identity."""

import unittest

import numpy as np

from tapeheads import selection
from tapeheads import sustained


class SustainedTest(unittest.TestCase):
    """Check behavior with controlled attention maps."""

    def setUp(self):
        self.edges = np.arange(101) * 100
        self.valid = np.ones(100, dtype=bool)
        self.config = selection.SelectionConfig(
            average_window_seconds=0, minimum_block_seconds=0.12
        )

    def test_layer_limit_and_empty_heads(self):
        rows = np.full((5, 100), 0.001)
        rows[:4, 30:40] = 0.09
        selector = sustained.LayerSelector(self.config, self.edges, 1000, 5)
        diagnostics, active = selector.select(rows, 100, self.valid)
        self.assertEqual(len(active), 3)
        self.assertFalse(diagnostics[4]['selected_blocks'])

    def test_multiple_blocks_can_belong_to_one_head(self):
        rows = np.full((1, 100), 0.001)
        for start in (20, 40, 60):
            rows[0, start : start + 5] = 0.06
        selector = sustained.LayerSelector(self.config, self.edges, 1000, 1)
        _, active = selector.select(rows, 100, self.valid)
        self.assertEqual(len(active), 3)

    def test_overlap_preserves_voice_and_playback_extent(self):
        rows = np.full((1, 100), 0.001)
        rows[0, 30:40] = 0.09
        selector = sustained.LayerSelector(self.config, self.edges, 1000, 1)
        _, first = selector.select(rows, 100, self.valid)
        rows[0, 30:32] = 0.001
        _, second = selector.select(rows, 200, self.valid)
        self.assertEqual(first[0]['voice_id'], second[0]['voice_id'])
        self.assertEqual(second[0]['playback_start'], 3000)
        self.assertEqual(second[0]['source_start'], 3200)
        rows[:] = 0.001
        rows[0, 60:70] = 0.09
        _, third = selector.select(rows, 300, self.valid)
        self.assertNotEqual(second[0]['voice_id'], third[0]['voice_id'])

    def test_uniform_attention_is_silent_including_near(self):
        selector = sustained.LayerSelector(self.config, self.edges, 1000, 1)
        diagnostics, active = selector.select(
            np.full((1, 100), 0.01), 5000, self.valid
        )
        self.assertFalse(active)
        self.assertFalse(diagnostics[0]['near_attention']['active'])

    def test_diagonal_qualifies_quiet_monitor(self):
        rows = np.zeros((1, 100))
        rows[0, 49:52] = 1 / 3
        selector = sustained.LayerSelector(self.config, self.edges, 1000, 1)
        diagnostics, active = selector.select(rows, 5000, self.valid)
        self.assertFalse(active)
        self.assertTrue(diagnostics[0]['near_attention']['active'])

    def test_average_allows_weak_interior_frame(self):
        row = np.zeros(100)
        row[30:40] = 0.1
        row[35] = 0
        blocks = sustained.average_blocks(
            row, self.valid, 0.02, self.edges, 1000, 0.3, 0.12
        )
        self.assertEqual(len(blocks), 1)
        self.assertLess(blocks[0]['key_start'], 35)
        self.assertGreater(blocks[0]['key_end'], 35)
        self.assertGreater(blocks[0]['mean_attention'], 0.02)


if __name__ == '__main__':
    unittest.main()
