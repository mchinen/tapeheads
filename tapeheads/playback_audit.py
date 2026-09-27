"""Measure whether scheduled audio remains inside current attention support."""

import bisect

from tapeheads import trace_types


def support_statistics(trace: trace_types.JsonObject) -> dict[str, int | float]:
    """Compare source runs with selected support at every query interval."""
    scheduled = 0
    covered = 0
    unsupported = 0
    for head in trace['heads']:
        voices = {
            voice['voice_id']: voice['playback'] for voice in head['voices']
        }
        starts = {
            identifier: [segment['output_start'] for segment in segments]
            for identifier, segments in voices.items()
        }
        scheduled += sum(
            segment['output_end'] - segment['output_start']
            for segments in voices.values()
            for segment in segments
        )
        for item in head['selections']:
            for block in item['selected_blocks']:
                identifier = block['voice_id']
                segments = voices[identifier]
                index = max(
                    0,
                    bisect.bisect_right(
                        starts[identifier], item['query_sample']
                    )
                    - 1,
                )
                while index < len(segments):
                    segment = segments[index]
                    if segment['output_start'] >= item['output_end']:
                        break
                    start = max(item['query_sample'], segment['output_start'])
                    end = min(item['output_end'], segment['output_end'])
                    index += 1
                    if start >= end:
                        continue
                    source_start = (
                        segment['source_start']
                        + start
                        - segment['output_start']
                    )
                    length = end - start
                    supported = max(
                        0,
                        min(source_start + length, block['source_end'])
                        - max(source_start, block['source_start']),
                    )
                    unsupported += length - supported
                    covered += length
    return {
        'scheduled_samples': scheduled,
        'unselected_samples': scheduled - covered,
        'unsupported_samples': unsupported,
        'unsupported_fraction': unsupported / max(1, scheduled),
    }
