"""Check shared recursion, depth transitions, and synthetic annotations."""

import unittest

import numpy as np

from tapeheads import annotations
from tapeheads import deep
from tapeheads import selection
from tapeheads import sustained


def layer(start, end, source):
    """Build a single-voice schedule for an unambiguous clock test."""
    return {
        'selection': {'max_layer_blocks': 6},
        'near_monitor': {'relative_gain': 0.15, 'playback': []},
        'heads': [
            {
                'index': 0,
                'voices': [
                    {
                        'voice_id': 0,
                        'playback': [
                            {
                                'output_start': start,
                                'output_end': end,
                                'source_start': source,
                                'source_end': source + end - start,
                                'fade_in_samples': 0,
                                'fade_out_samples': 0,
                            }
                        ],
                    }
                ],
            }
        ],
    }


class DeepTest(unittest.TestCase):
    """Exercise behavior independently of pretrained model weights."""

    def test_second_bus_reads_first_bus_at_attended_time(self):
        source = np.zeros(1000, dtype=np.float32)
        source[200:300] = 0.2
        first, _ = deep.render_bus(layer(0, 100, 200), source, None)
        second, statistics = deep.render_bus(layer(400, 500, 0), source, first)
        expected = first[:100] * (0.8 / 6.15) * statistics['gain']
        np.testing.assert_allclose(second[400:500], expected, atol=1e-7)
        self.assertGreater(float(second[450, 0]), 0)
        self.assertFalse(second[:400].any())

    def test_progression_reaches_eighth_layer_and_crossfades(self):
        buses = [
            np.full((800, 2), index / 10, dtype=np.float32)
            for index in range(8)
        ]
        stages = deep.progression(800, 8)
        mix = deep.progressive_mix(buses, stages, 100)
        self.assertEqual(stages[-1]['depth'], 8)
        self.assertAlmostEqual(float(mix[100, 0]), 0)
        self.assertAlmostEqual(float(mix[107, 0]), 0.1)
        self.assertAlmostEqual(float(mix[-1, 0]), 0.7)

    def test_signal_labels_clip_to_excerpt(self):
        result = annotations.create(np.zeros(4000), 1000, 'chirp_sequence')
        self.assertEqual(len(result['sections']), 2)
        self.assertEqual(result['sections'][-1]['end'], 4000)
        self.assertEqual(result['sections'][-1]['confidence'], 'exact')

    def test_six_blocks_are_not_truncated_to_old_shortlist(self):
        row = np.zeros((1, 100))
        for start in (10, 25, 40, 55, 70, 85):
            row[0, start : start + 5] = 1 / 30
        config = selection.SelectionConfig(
            max_layer_blocks=6,
            average_window_seconds=0,
            minimum_block_seconds=0.1,
        )
        selector = sustained.LayerSelector(
            config, np.arange(101) * 100, 1000, 1
        )
        _, active = selector.select(row, 0, np.ones(100, dtype=bool))
        self.assertEqual(len(active), 6)


if __name__ == '__main__':
    unittest.main()
