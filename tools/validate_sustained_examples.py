"""Validate sustained examples and record reproducible continuity comparisons."""

import json

import numpy as np

from tapeheads import io
from tapeheads.validation import validate


def main() -> None:
    """Write verification results and run-length statistics for saved traces."""
    names = (
        'chirp_sequence_wav2vec2',
        'blue_suede_shoes_full_wav2vec2',
    )
    report = {
        'generator': 'python -m tools.validate_sustained_examples',
        'statistic': 'uninterrupted 1x source-to-output runs, including loops as separate runs',
        'comparisons': {},
        'validation': {},
    }
    for name in names:
        comparison = {}
        for suffix in ('', '_v2'):
            path = io.ROOT / 'outputs' / (name + suffix) / 'trace.json'
            trace = json.loads(path.read_text())
            sample_rate = trace['source']['sample_rate_hz']
            lengths = np.array(
                [
                    (segment['output_end'] - segment['output_start'])
                    / sample_rate
                    for head in trace['heads']
                    for segment in head['playback']
                ]
            )
            comparison[trace['format']] = {
                'segments': len(lengths),
                'median_seconds': float(np.median(lengths)),
                'p90_seconds': float(np.quantile(lengths, 0.9)),
                'maximum_seconds': float(lengths.max()),
            }
            if suffix:
                report['validation'][name + suffix] = validate(path)
        report['comparisons'][name] = comparison
    for name in ('chirp_sequence_clap_v2', 'chirp_sequence_muq_v2'):
        report['validation'][name] = validate(
            io.ROOT / 'outputs' / name / 'trace.json'
        )
    io.write_json(io.ROOT / 'data' / 'sustained-validation.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
