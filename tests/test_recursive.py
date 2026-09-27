"""Test recursive clock composition against explicit hand-built tape paths."""

import tempfile
import unittest

import numpy as np
import soundfile

from tapeheads import recursive


def segment(output_start, output_end, source_start, fade=0):
    """Build a 1x tape run with optional symmetric fades."""
    return {
        'output_start': output_start,
        'output_end': output_end,
        'source_start': source_start,
        'source_end': source_start + output_end - output_start,
        'fade_in_samples': fade,
        'fade_out_samples': fade,
    }


def layer(segments):
    """Build a minimal layer with one voice per supplied head."""
    return {
        'selection': {'guard_samples': 10, 'max_layer_blocks': 3},
        'near_monitor': {'relative_gain': 0.15, 'playback': []},
        'heads': [
            {'index': index, 'voices': [{'voice_id': index, 'playback': runs}]}
            for index, runs in enumerate(segments)
        ],
    }


class RecursiveTest(unittest.TestCase):
    """Follow intermediate time, preserve overlaps, and attenuate near returns."""

    def test_child_is_read_at_intermediate_time(self):
        first = layer([[segment(200, 250, 700)]])
        second = layer([[segment(0, 100, 180)]])
        events = recursive.compose(first, second)
        children = [event for event in events if event['branch'] == 'recursive']
        self.assertEqual(len(children), 1)
        child = children[0]
        self.assertEqual((child['output_start'], child['output_end']), (20, 70))
        self.assertEqual(
            (child['source_start'], child['source_end']), (700, 750)
        )
        self.assertEqual(child['intermediate_start'], 200)

    def test_all_nine_selected_paths_are_preserved(self):
        first = layer(
            [[segment(200, 300, 500 + index * 100)] for index in range(3)]
        )
        second = layer([[segment(0, 100, 200)] for _ in range(3)])
        events = recursive.compose(first, second)
        self.assertEqual(
            sum(event['branch'] == 'recursive' for event in events), 9
        )
        self.assertEqual(
            sum(event['branch'] == 'direct' for event in events), 3
        )

    def test_current_output_return_is_quieter(self):
        second = layer([[segment(0, 100, 200)]])
        distant = recursive.compose(layer([[segment(200, 300, 500)]]), second)
        nearby = recursive.compose(layer([[segment(200, 300, 0)]]), second)
        remote = next(
            event for event in distant if event['branch'] == 'recursive'
        )
        local = next(
            event for event in nearby if event['branch'] == 'recursive'
        )
        self.assertTrue(local['near_output'])
        self.assertAlmostEqual(local['gain'], remote['gain'] * 0.15)

    def test_parent_head_two_can_follow_child_head_five(self):
        first = layer([[], [], [], [], [segment(200, 300, 700)]])
        second = layer([[], [segment(0, 100, 200)]])
        events = recursive.compose(first, second)
        children = [event for event in events if event['branch'] == 'recursive']
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0]['parent_head'], 1)
        self.assertEqual(children[0]['child_head'], 4)
        self.assertEqual(children[0]['source_start'], 700)

    def test_empty_child_keeps_only_direct_branch(self):
        events = recursive.compose(layer([[]]), layer([[segment(0, 100, 200)]]))
        self.assertEqual([event['branch'] for event in events], ['direct'])

    def test_audio_uses_both_fades_and_recorded_gain(self):
        first = layer([[segment(200, 300, 500, fade=5)]])
        second = layer([[segment(0, 100, 200, fade=5)]])
        events = recursive.compose(first, second)
        trace = {
            'source': {'sample_rate_hz': 1000},
            'layers': [first, second],
            'events': events,
        }
        with tempfile.TemporaryDirectory() as directory:
            recursive.render_audio(
                trace, np.ones(1000, dtype=np.float32), directory
            )
            audio, _ = soundfile.read(f'{directory}/recursive.wav')
        event = next(
            event for event in events if event['branch'] == 'recursive'
        )
        self.assertEqual(audio[0, 0], 0)
        self.assertAlmostEqual(audio[2, 0], event['gain'] * 0.25, places=7)
        self.assertAlmostEqual(audio[50, 0], event['gain'], places=7)
        self.assertEqual(audio[100, 0], 0)


if __name__ == '__main__':
    unittest.main()
