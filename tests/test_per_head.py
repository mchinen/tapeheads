"""Check independent voice budgets and support-renewed playback spans."""

import unittest

import numpy as np

from tapeheads import per_head
from tapeheads import playback
from tapeheads import selection
from tapeheads import shared_audio


class PerHeadTest(unittest.TestCase):
    """Verify persistence without stale support or a hidden layer-wide cap."""

    def setUp(self):
        self.config = selection.SelectionConfig(
            tracking_policy='per_head_support',
            entry_alpha=1.5,
            exit_alpha=1.3,
            average_window_seconds=0,
            minimum_block_seconds=0.2,
            switch_ratio=1.15,
        )
        self.edges = np.arange(501) * 100
        self.valid = np.ones(500, dtype=bool)

    def test_three_blocks_are_available_to_each_head(self):
        selector = per_head.PerHeadSelector(self.config, self.edges, 1000, 2)
        rows = np.full((2, 500), 0.0001)
        for start in (100, 200, 300):
            rows[:, start : start + 10] = 0.02
        diagnostics, voices = selector.select(rows, 0, self.valid)
        self.assertEqual(len(voices), 6)
        self.assertEqual(
            [len(item['selected_blocks']) for item in diagnostics], [3, 3]
        )

    def test_five_second_acquisition_extends_only_under_fresh_support(self):
        selector = per_head.PerHeadSelector(self.config, self.edges, 1000, 1)
        rows = np.full((1, 500), 0.0001)
        rows[0, 100:200] = 0.005
        _, first = selector.select(rows, 0, self.valid)
        self.assertEqual(
            first[0]['source_end'] - first[0]['source_start'], 5000
        )
        _, second = selector.select(rows, 100, self.valid)
        self.assertEqual(second[0]['voice_id'], first[0]['voice_id'])
        self.assertEqual(second[0]['source_end'], first[0]['source_end'] + 100)
        rows[:] = 0.0001
        _, last = selector.select(rows, 200, self.valid)
        self.assertFalse(last)

    def test_only_three_heads_play_with_three_regions_each(self):
        selector = per_head.PerHeadSelector(self.config, self.edges, 1000, 5)
        rows = np.full((5, 500), 0.0001)
        for head in range(5):
            for start in (100, 200, 300):
                rows[head, start : start + 10] = 0.01 + head * 0.002
        diagnostics, voices = selector.select(rows, 0, self.valid)
        self.assertEqual({voice['head'] for voice in voices}, {2, 3, 4})
        self.assertEqual(len(voices), 9)
        self.assertFalse(diagnostics[0]['selected_blocks'])

    def test_supported_playback_remains_contiguous_past_five_seconds(self):
        selector = per_head.PerHeadSelector(self.config, self.edges, 1000, 1)
        rows = np.full((1, 500), 0.0001)
        rows[0, 200:400] = 0.006
        tape = playback.TapeHead(100)
        identifiers = set()
        for query in range(0, 8000, 100):
            _, voices = selector.select(rows, query, self.valid)
            identifiers.add(voices[0]['voice_id'])
            tape.append(
                query,
                query + 100,
                {
                    'source_start': voices[0]['source_start'],
                    'source_end': voices[0]['source_end'],
                },
            )
        self.assertEqual(len(identifiers), 1)
        self.assertEqual(len(tape.segments), 1)
        self.assertEqual(tape.segments[0]['output_end'], 8000)
        rows[:] = 0.0001
        _, voices = selector.select(rows, 8000, self.valid)
        self.assertFalse(voices)

    def test_shared_recursion_preserves_opposite_side_children(self):
        def layer(head, start, end, source_start):
            heads = [{'index': index, 'voices': []} for index in range(2)]
            heads[head]['voices'] = [
                {
                    'voice_id': 0,
                    'playback': [
                        {
                            'output_start': start,
                            'output_end': end,
                            'source_start': source_start,
                            'source_end': source_start + end - start,
                        }
                    ],
                }
            ]
            return {
                'heads': heads,
                'selection': {'voice_budget': 1},
                'near_monitor': {'relative_gain': 0.15, 'playback': []},
            }

        source = np.zeros(1000, dtype=np.float32)
        source[700:800] = 0.2
        audio, _ = shared_audio.render_depth(
            [layer(1, 300, 400, 700), layer(0, 0, 100, 300)], source, 1000
        )
        self.assertGreater(float(audio[50, 1]), 0.01)
        self.assertFalse(audio[100:].any())


if __name__ == '__main__':
    unittest.main()
