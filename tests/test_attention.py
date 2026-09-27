"""Regression tests for source exclusion, scoring, and tape scheduling."""

import unittest

import numpy as np

from tapeheads import clap_adapter
from tapeheads import playback
from tapeheads import selection


class SelectionTest(unittest.TestCase):
    """Verify attention behavior using deliberately constructed distributions."""

    def setUp(self):
        self.edges = np.arange(21) * 20
        self.config = selection.SelectionConfig(guard_seconds=0.1)

    def test_vector_reduction_matches_scalar_reference(self):
        random_generator = np.random.default_rng(7)
        for unused_trial in range(20):
            row = random_generator.random(100)
            valid = random_generator.random(100) > 0.2
            edges = np.r_[0, np.cumsum(random_generator.integers(1, 10, 100))]
            ranked = selection.blocks(row, valid, 0.5, edges, 1000)
            expected_keys = np.flatnonzero(valid & (row > 0.5))
            actual_keys = []
            for block in ranked:
                start, end = block['key_start'], block['key_end']
                actual_keys.extend(range(start, end))
                volume = sum(
                    (row[k] - 0.5) * (edges[k + 1] - edges[k]) / 1000
                    for k in range(start, end)
                )
                self.assertAlmostEqual(block['volume'], volume)
                self.assertAlmostEqual(block['mass'], sum(row[start:end]))
                self.assertEqual(block['peak'], max(row[start:end]))
            np.testing.assert_array_equal(sorted(actual_keys), expected_keys)

    def test_diagonal_is_excluded_and_second_region_survives(self):
        row = np.zeros(20)
        row[10] = 0.6
        row[0:2] = 0.2
        selector = selection.HeadSelector(self.config, self.edges, 1000)
        chosen = selector.select(row, 210)
        self.assertEqual(chosen['selected']['key_start'], 0)
        self.assertEqual(chosen['selected']['key_end'], 2)
        self.assertEqual(chosen['selected']['original_rank'], 2)

    def test_integrated_excess_beats_a_higher_peak(self):
        row = np.zeros(20)
        row[0] = 0.25
        row[15:18] = 0.2
        selector = selection.HeadSelector(self.config, self.edges, 1000)
        chosen = selector.select(row, 160)
        self.assertEqual(chosen['selected']['key_start'], 15)
        self.assertAlmostEqual(chosen['selected']['volume'], 0.006)

    def test_uniform_attention_stays_silent(self):
        row = np.full(20, 1 / 20)
        selector = selection.HeadSelector(self.config, self.edges, 1000)
        self.assertIsNone(selector.select(row, 210)['selected'])

    def test_guard_splits_a_block_and_overrides_hold(self):
        row = np.zeros(20)
        row[4:16] = 0.2
        selector = selection.HeadSelector(self.config, self.edges, 1000)
        first = selector.select(row, 0)['selected']
        self.assertIsNotNone(first)
        chosen = selector.select(row, 210)
        for block in chosen['candidates']:
            self.assertTrue(
                block['source_end'] <= 110 or block['source_start'] > 310
            )
        if chosen['selected']:
            block = chosen['selected']
            self.assertTrue(
                block['source_end'] <= 110 or block['source_start'] > 310
            )

    def test_context_mask_controls_threshold_not_song_length(self):
        row = np.zeros(20)
        row[:4] = 0.25
        visible = np.zeros(20, dtype=bool)
        visible[:4] = True
        selector = selection.HeadSelector(self.config, self.edges, 1000)
        self.assertIsNone(selector.select(row, 350, visible)['selected'])

    def test_disjoint_winner_waits_for_dwell(self):
        config = selection.SelectionConfig(guard_seconds=0, dwell_frames=3)
        selector = selection.HeadSelector(config, self.edges, 1000)
        first = np.zeros(20)
        first[:2] = 0.25
        selector.select(first, 200)
        second = np.zeros(20)
        second[:2] = 0.12
        second[15:18] = 0.24
        self.assertEqual(
            selector.select(second, 200)['selected']['key_start'], 0
        )
        self.assertEqual(
            selector.select(second, 200)['selected']['key_start'], 0
        )
        self.assertEqual(
            selector.select(second, 200)['selected']['key_start'], 15
        )


class PlaybackTest(unittest.TestCase):
    """Ensure every audible sample has a safe source offset."""

    def test_loop_never_enters_guard(self):
        head = playback.TapeHead(100)
        selected = {'source_start': 200, 'source_end': 260}
        head.append(0, 300, selected)
        segments = playback.finalize_segments(head.segments, 1000, 100)
        self.assertGreater(len(segments), 1)
        for segment in segments:
            output = np.arange(segment['output_start'], segment['output_end'])
            source = np.arange(segment['source_start'], segment['source_end'])
            self.assertTrue(np.all(np.abs(source - output) > 100))
            self.assertTrue(np.all((source >= 200) & (source < 260)))

    def test_continuation_does_not_restart_at_each_query(self):
        head = playback.TapeHead(100)
        selected = {'source_start': 1000, 'source_end': 2000}
        head.append(0, 20, selected)
        head.append(20, 40, selected)
        self.assertEqual(len(head.segments), 1)
        self.assertEqual(head.segments[0]['source_end'], 1040)

    def test_silence_breaks_continuation(self):
        head = playback.TapeHead(100)
        selected = {'source_start': 1000, 'source_end': 2000}
        head.append(0, 20, selected)
        head.append(20, 40, None)
        head.append(40, 60, selected)
        self.assertEqual(len(head.segments), 2)
        self.assertEqual(head.segments[1]['source_start'], 1000)


class ClapProjectionTest(unittest.TestCase):
    """Check stripe mapping, probability conservation, and native visibility."""

    def test_uniform_native_windows_stay_uniform_over_eight_times(self):
        native = np.full((64, 4, 64, 64), 1 / 64, dtype=np.float32)
        projected, valid = clap_adapter.project_windows(native)
        np.testing.assert_allclose(projected.sum(-1), 1)
        np.testing.assert_array_equal(valid.sum(-1), 8)
        np.testing.assert_allclose(projected[:, 0, :8], 1 / 8)
        np.testing.assert_array_equal(projected[:, 0, 8:], 0)
        np.testing.assert_allclose(projected[:, 64, 64:72], 1 / 8)

    def test_identity_attention_stays_at_the_same_time(self):
        native = np.broadcast_to(np.eye(64, dtype=np.float32), (64, 4, 64, 64))
        projected, unused_valid = clap_adapter.project_windows(native)
        np.testing.assert_allclose(projected[0], np.eye(256))


if __name__ == '__main__':
    unittest.main()
