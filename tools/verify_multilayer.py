"""Compare single-pass attention with independent native layer extraction."""

import argparse

import numpy as np
import torch

from tapeheads import encoders
from tapeheads import io
from tapeheads import multilayer


def main() -> None:
    """Gate long-form inference with checks across all requested layers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--encoder', choices=('wav2vec2', 'hubert', 'muq'), required=True
    )
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    audio = io.decode(
        io.ROOT / 'data/audio/chirp_sequence/mono_48000.wav',
        duration_seconds=4,
        sample_rate=encoders.ENCODERS[arguments.encoder].sample_rate,
    )
    views = multilayer.create_views(arguments.encoder, audio)
    errors = []
    for index, view in enumerate(views):
        reference = encoders.create_view(
            arguments.encoder, audio, verify=True, layer_index=index
        )
        maximum = 0.0
        assert np.array_equal(view.edges, reference.edges)
        assert np.array_equal(view.queries, reference.queries)
        for (query, actual, valid), (other_query, expected, other_valid) in zip(
            view.rows(), reference.rows()
        ):
            assert query == other_query and np.array_equal(valid, other_valid)
            maximum = max(maximum, float(np.abs(actual - expected).max()))
        assert maximum < 1e-5, (index, maximum)
        errors.append(maximum)
        print(index, maximum, flush=True)
    io.write_json(
        io.ROOT / f'data/single-pass-{arguments.encoder}-validation.json',
        {
            'generator': 'python -m tools.verify_multilayer',
            'encoder': arguments.encoder,
            'maximum_errors': errors,
            'absolute_tolerance': 1e-5,
            'duration_seconds': 4,
        },
    )


if __name__ == '__main__':
    main()
