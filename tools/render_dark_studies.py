"""Generate dark recursive studies across installed temporal encoders."""

import argparse
import json

from tapeheads import io
from tapeheads import recursive
from tapeheads import render
from tapeheads import selection
from tapeheads import validation


def main() -> None:
    """Run reproducible longer-span comparisons on chirps and full Elvis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--encoder', choices=('wav2vec2', 'hubert', 'muq'), required=True
    )
    arguments = parser.parse_args()
    config = selection.SelectionConfig(
        entry_alpha=1.5,
        exit_alpha=1.1,
        average_window_seconds=0.35,
        minimum_block_seconds=0.3,
    )
    report = {
        'generator': 'python -m tools.render_dark_studies',
        'encoder': arguments.encoder,
        'studies': {},
    }
    for name, path in (
        ('chirp_sequence', 'data/audio/chirp_sequence/mono_48000.wav'),
        ('blue_suede_shoes_full', 'data/audio/blue_suede_shoes/source.flac'),
    ):
        output = io.ROOT / 'outputs' / f'{name}_{arguments.encoder}_dark'
        trace_path = output / 'trace.json'
        if not trace_path.exists():
            recursive.analyze(
                io.ROOT / path, output, encoder=arguments.encoder, config=config
            )
        render.render(trace_path)
        report['studies'][name] = validation.validate(trace_path)
        io.write_json(
            io.ROOT / 'data' / f'dark-{arguments.encoder}-validation.json',
            report,
        )
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
