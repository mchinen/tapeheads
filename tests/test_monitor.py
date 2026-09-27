"""Verify the quiet monitor's gain and independent source timeline."""

import pathlib
import tempfile
import unittest

import numpy as np
import soundfile

from tapeheads import playback


class MonitorTest(unittest.TestCase):
    """Keep current-time audio separate from remote stems."""

    def test_monitor_gain_is_applied_once(self):
        trace = {
            'source': {'sample_rate_hz': 1000},
            'selection': {'max_layer_blocks': 3},
            'heads': [{'index': 0, 'playback': []}],
            'near_monitor': {
                'relative_gain': 0.15,
                'playback': [
                    {
                        'output_start': 100,
                        'output_end': 200,
                        'source_start': 100,
                        'source_end': 200,
                        'fade_in_samples': 0,
                        'fade_out_samples': 0,
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            metadata = playback.render_audio(
                trace, np.ones(1000, dtype=np.float32), directory
            )
            nearby, _ = soundfile.read(pathlib.Path(directory) / 'nearby.wav')
            mix, _ = soundfile.read(pathlib.Path(directory) / 'mix.wav')
            head, _ = soundfile.read(pathlib.Path(directory) / 'head_00.wav')
        np.testing.assert_allclose(nearby[100:200], 0.15)
        np.testing.assert_allclose(
            mix[100:200], metadata['head_gain'] * 0.15 / np.sqrt(2)
        )
        self.assertFalse(np.any(head))
        self.assertFalse(np.any(nearby[:100]))
        self.assertFalse(np.any(nearby[200:]))


if __name__ == '__main__':
    unittest.main()
