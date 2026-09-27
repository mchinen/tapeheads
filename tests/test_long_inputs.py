"""Cover long attention rows and resampled annotation boundaries."""

import unittest
from unittest import mock

import numpy as np
import torch

from tapeheads import annotations
from tapeheads import multilayer


class LongInputTest(unittest.TestCase):
    """Check numerical and clock boundaries found in collection studies."""

    def test_peaked_long_attention_rows_keep_unit_mass(self):
        query = torch.ones(1, 16384, 1)
        key = torch.full_like(query, -15)
        key[:, 0] = 0
        _, weights, _ = next(multilayer.make_rows(query, key, 1)())
        self.assertLess(abs(float(weights.sum()) - 1), 2e-6)
        expected = 1 / (1 + 16383 * np.exp(-15))
        self.assertAlmostEqual(float(weights[0, 0]), expected, places=7)

    def test_nature_labels_cover_final_resampled_sample(self):
        manifest = {
            'sections': [
                {'start_seconds': 0, 'end_seconds': 1, 'label': 'BIRDS'}
            ]
        }
        with (
            mock.patch.object(
                annotations.io, 'read_json', return_value=manifest
            ),
            mock.patch.object(annotations.io, 'sha256', return_value='fixture'),
        ):
            result = annotations.create(
                np.zeros(16001), 16000, 'nature_montage'
            )
        self.assertEqual(result['sections'][-1]['end'], 16001)


if __name__ == '__main__':
    unittest.main()
