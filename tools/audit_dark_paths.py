"""Audit dark-view bracket alignment, cross-head paths, and run lengths."""

import bisect
import json
from typing import Any

import numpy as np

from tapeheads import io


def selected(
    layer: dict[str, Any], head: int, voice: int, sample: int
) -> dict[str, Any] | None:
    """Resolve a voice at its own layer's query clock."""
    index = bisect.bisect_right(layer['frames']['query_samples'], sample) - 1
    if index < 0:
        return None
    return next(
        (
            block
            for block in layer['heads'][head]['selections'][index][
                'selected_blocks'
            ]
            if block['voice_id'] == voice
        ),
        None,
    )


def audit(trace: dict[str, Any]) -> dict[str, Any]:
    """Check visible endpoints against the same brackets used by the film."""
    checked = cross_head = 0
    for event in trace['events']:
        if (
            event['branch'] != 'recursive'
            or event['near_output']
            or event['parent_head'] < 0
            or event['child_head'] < 0
        ):
            continue
        cross_head += event['parent_head'] != event['child_head']
        for sample in (event['output_start'], event['output_end'] - 1):
            middle = (
                event['intermediate_start'] + sample - event['output_start']
            )
            source = event['source_start'] + sample - event['output_start']
            parent = selected(
                trace['layers'][1],
                event['parent_head'],
                event['parent_voice'],
                sample,
            )
            child = selected(
                trace['layers'][0],
                event['child_head'],
                event['child_voice'],
                middle,
            )
            assert parent and child
            assert parent['playback_start'] <= middle < parent['playback_end']
            assert child['playback_start'] <= source < child['playback_end']
            checked += 2
    sample_rate = trace['source']['sample_rate_hz']
    statistics = []
    for layer in trace['layers']:
        durations = [
            (segment['output_end'] - segment['output_start']) / sample_rate
            for head in layer['heads']
            for segment in head['playback']
        ]
        statistics.append(
            {
                'runs': len(durations),
                'median_seconds': float(np.median(durations))
                if durations
                else 0,
                'runs_at_most_50_ms': sum(value <= 0.05 for value in durations),
            }
        )
    return {
        'bracket_endpoints_checked': checked,
        'cross_head_recursive_events': int(cross_head),
        'layers': statistics,
    }


def main() -> None:
    """Save numerical audits for all six dark studies."""
    report = {'generator': 'python -m tools.audit_dark_paths', 'studies': {}}
    for encoder in ('wav2vec2', 'muq', 'hubert'):
        for source in ('chirp_sequence', 'blue_suede_shoes_full'):
            name = f'{source}_{encoder}_dark'
            trace = json.loads(
                (io.ROOT / 'outputs' / name / 'trace.json').read_text()
            )
            report['studies'][name] = audit(trace)
    io.write_json(io.ROOT / 'data/dark-path-audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
