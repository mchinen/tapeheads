"""Check the repeated test sequence's timing and actual signal content."""

import unittest

import numpy as np

from tools import generate_chirp_sequence as sequence


class ChirpSequenceTest(unittest.TestCase):
    """Verify repetitions, exact silence, and the two DTMF frequencies."""

    @classmethod
    def setUpClass(cls):
        cls.audio, cls.events = sequence.build_sequence()

    def test_chirps_repeat_exactly_at_expected_times(self):
        chirps = [event for event in self.events if event['kind'] == 'chirp']
        self.assertEqual(
            [event['start_seconds'] for event in chirps], [1, 10, 19, 28]
        )
        first = self.audio[chirps[0]['start_sample'] : chirps[0]['end_sample']]
        for event in chirps[1:]:
            np.testing.assert_array_equal(
                first, self.audio[event['start_sample'] : event['end_sample']]
            )
        self.assertEqual(len(self.audio), 32 * sequence.SAMPLE_RATE)

    def test_silence_is_zero_and_all_samples_are_finite(self):
        self.assertTrue(np.isfinite(self.audio).all())
        self.assertLess(float(np.abs(self.audio).max()), 1)
        for event in self.events:
            if event['kind'] == 'silence':
                self.assertFalse(
                    np.any(
                        self.audio[event['start_sample'] : event['end_sample']]
                    )
                )

    def test_dtmf_has_the_correct_frequency_pairs(self):
        for event in self.events:
            if event['kind'] != 'dtmf':
                continue
            tone = self.audio[event['start_sample'] : event['end_sample']]
            spectrum = np.abs(np.fft.rfft(tone * np.hanning(len(tone))))
            frequencies = np.fft.rfftfreq(len(tone), 1 / sequence.SAMPLE_RATE)
            for left, right, expected in (
                (650, 1000, event['parameters']['frequencies_hz'][0]),
                (1150, 1700, event['parameters']['frequencies_hz'][1]),
            ):
                selected = (frequencies >= left) & (frequencies <= right)
                peak = frequencies[selected][np.argmax(spectrum[selected])]
                self.assertLess(abs(peak - expected), 3)

    def test_noise_is_seeded_but_not_repeated(self):
        second, unused_events = sequence.build_sequence()
        np.testing.assert_array_equal(self.audio, second)
        noises = [
            self.audio[event['start_sample'] : event['end_sample']]
            for event in self.events
            if event['kind'] == 'gaussian_noise'
        ]
        self.assertFalse(np.array_equal(noises[0], noises[1]))
        for noise in noises:
            self.assertLess(abs(float(noise.mean())), 0.002)
            self.assertAlmostEqual(float(noise.std()), 0.1, delta=0.002)


if __name__ == '__main__':
    unittest.main()
