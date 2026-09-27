"""Regressions for stale support, duplicate playback, and recursive gating."""

import dataclasses
import unittest

import numpy as np

from tapeheads import responsive_audio
from tapeheads import selection
from tapeheads import sustained


class CurrentSupportTest(unittest.TestCase):
    """Require continued playback to qualify against the current attention."""

    def setUp(self):
        self.edges = np.arange(101) * 100
        self.valid = np.ones(100, dtype=bool)
        self.config = selection.SelectionConfig(
            tracking_policy='current_support',
            max_layer_blocks=1,
            average_window_seconds=0,
            minimum_block_seconds=0.12,
            switch_ratio=1.05,
        )

    def test_shrinking_support_discards_historical_bounds(self):
        selector = sustained.LayerSelector(self.config, self.edges, 1000, 1)
        rows = np.full((1, 100), 0.001)
        rows[0, 30:40] = 0.09
        _, first = selector.select(rows, 100, self.valid)
        rows[0, 30:32] = 0.001
        _, second = selector.select(rows, 200, self.valid)
        self.assertEqual(first[0]['voice_id'], second[0]['voice_id'])
        self.assertEqual(second[0]['playback_start'], 3200)
        self.assertEqual(second[0]['playback_end'], 4000)

    def test_stronger_remote_block_replaces_supported_old_voice(self):
        selector = sustained.LayerSelector(self.config, self.edges, 1000, 2)
        rows = np.full((2, 100), 0.001)
        rows[0, 30:40] = 0.09
        _, first = selector.select(rows, 100, self.valid)
        rows[1, 60:70] = 0.18
        _, second = selector.select(rows, 200, self.valid)
        self.assertEqual(second[0]['head'], 1)
        self.assertNotEqual(first[0]['voice_id'], second[0]['voice_id'])

    def test_weak_continuation_does_not_keep_a_voice_alive(self):
        selector = sustained.LayerSelector(self.config, self.edges, 1000, 1)
        rows = np.full((1, 100), 0.001)
        rows[0, 30:40] = 0.09
        selector.select(rows, 100, self.valid)
        rows[0, 30:40] = 0.015
        _, active = selector.select(rows, 200, self.valid)
        self.assertFalse(active)

    def test_heads_reading_the_same_block_share_one_slot(self):
        config = dataclasses.replace(self.config, max_layer_blocks=3)
        selector = sustained.LayerSelector(config, self.edges, 1000, 2)
        rows = np.full((2, 100), 0.001)
        rows[:, 30:40] = 0.09
        _, active = selector.select(rows, 100, self.valid)
        self.assertEqual(len(active), 1)


class RouteAudioTest(unittest.TestCase):
    """Check coincident gain control and continuous audio across query cuts."""

    def test_intermediate_frame_and_all_child_heads_are_audible(self):
        def layer(segments):
            return {
                'selection': {'guard_samples': 100, 'max_layer_blocks': 3},
                'near_monitor': {'relative_gain': 0.15, 'playback': []},
                'heads': [
                    {
                        'index': index,
                        'voices': [{'voice_id': index, 'playback': [segment]}],
                    }
                    for index, segment in enumerate(segments)
                ],
            }

        first = layer(
            [
                {'output_start': 300, 'output_end': 400, 'source_start': 600},
                {'output_start': 300, 'output_end': 400, 'source_start': 800},
            ]
        )
        second = layer(
            [{'output_start': 0, 'output_end': 100, 'source_start': 300}]
        )
        routes = responsive_audio.compose_routes([first, second], 1000)
        self.assertEqual(
            {route.source_offset for route in routes}, {300, 600, 800}
        )
        self.assertTrue(all(route.output_start == 0 for route in routes))
        self.assertTrue(all(route.output_end == 100 for route in routes))
        self.assertTrue(
            all(route.left_gain + route.right_gain > 0 for route in routes)
        )
        right_child = next(
            route for route in routes if route.source_offset == 800
        )
        self.assertGreater(right_child.right_gain, 0.01)

    def test_identical_paths_do_not_double_the_audio(self):
        source = np.sin(np.arange(2000) * 0.1).astype(np.float32)
        route = responsive_audio.SourceRoute(0, 1000, 100, 0.2, 0.2)
        single, _ = responsive_audio.mix_routes([route], source, 1000)
        duplicate, stats = responsive_audio.mix_routes(
            [route, route], source, 1000
        )
        np.testing.assert_array_equal(single, duplicate)
        self.assertEqual(stats['coincident_source_samples'], 1000)

    def test_contiguous_query_cuts_do_not_create_amplitude_gates(self):
        source = np.ones(2000, dtype=np.float32)
        whole = [responsive_audio.SourceRoute(0, 1000, 100, 0.2, 0.2)]
        split = [
            responsive_audio.SourceRoute(start, start + 40, 100, 0.2, 0.2)
            for start in range(0, 1000, 40)
        ]
        expected, _ = responsive_audio.mix_routes(whole, source, 1000)
        actual, stats = responsive_audio.mix_routes(split, source, 1000)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(stats['splice_corrections'], 1)

    def test_splice_correction_does_not_extend_into_unselected_silence(self):
        source = np.ones(1000, dtype=np.float32)
        routes = [responsive_audio.SourceRoute(0, 100, 200, 0.2, 0.2)]
        actual, _ = responsive_audio.mix_routes(routes, source, 1000)
        self.assertFalse(actual[100:].any())


if __name__ == '__main__':
    unittest.main()
