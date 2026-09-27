"""Validate a trace and its audio independently of the extraction adapter."""

import json
import pathlib
import subprocess

import jsonschema
import numpy as np
import soundfile

from tapeheads import io
from tapeheads import playback_audit
from tapeheads import trace_types


def validate(trace_path: trace_types.PathLike) -> trace_types.JsonObject:
    """Check schema, file integrity, timing, and the per-sample source guard."""
    trace_path = pathlib.Path(trace_path)
    trace = io.read_json(trace_path)
    if trace.get('format') == 'tapeheads/4':
        from tapeheads import deep_validation

        return deep_validation.validate(trace_path, trace)
    if trace.get('format') == 'tapeheads/3':
        from tapeheads import recursive_validation

        return recursive_validation.validate(trace_path, trace)
    schema = json.loads((io.ROOT / 'schemas' / 'trace.schema.json').read_text())
    jsonschema.validate(trace, schema)
    source = trace['source']
    assert io.sha256(trace_path.parent / source['path']) == source['sha256']
    count = source['samples']
    sample_rate = source['sample_rate_hz']
    guard = trace['selection']['guard_samples']
    frames = trace['frames']
    edges = np.array(frames['edges_samples'])
    queries = frames['query_samples']
    assert len(edges) == len(queries) + 1
    assert np.all(np.diff(edges) > 0)
    assert 0 <= edges[0] < edges[-1] <= count
    assert np.all(np.diff(queries) > 0)
    total_segments = 0
    events = []
    voice_budget = trace['selection'].get(
        'voice_budget',
        trace['selection'].get('max_layer_blocks', len(trace['heads'])),
    )
    if trace['format'] == 'tapeheads/2':
        for query_index in range(len(queries)):
            active = [
                block
                for head in trace['heads']
                for block in head['selections'][query_index]['selected_blocks']
            ]
            assert (
                len({block['head'] for block in active})
                <= trace['selection'].get(
                    'max_active_heads', len(trace['heads'])
                )
                if trace['selection'].get('tracking_policy')
                == 'per_head_support'
                else True
            )
            assert len(active) <= voice_budget
            assert len({block['voice_id'] for block in active}) == len(active)
    if trace['format'] == 'tapeheads/2':
        previous = {}
        seen = set()
        for query_index in range(len(queries)):
            current = {}
            for head in trace['heads']:
                item = head['selections'][query_index]
                assert item['selected'] == next(
                    iter(item['selected_blocks']), None
                )
                if (
                    trace['selection'].get('tracking_policy')
                    == 'per_head_support'
                ):
                    assert (
                        len(item['selected_blocks'])
                        <= trace['selection']['max_head_blocks']
                    )
                for block in item['selected_blocks']:
                    identifier = block['voice_id']
                    assert block['head'] == head['index']
                    if identifier in previous:
                        previous_segment = previous[identifier]
                        assert previous_segment['head'] == block['head']
                        assert previous_segment['key_start'] < block['key_end']
                        assert block['key_start'] < previous_segment['key_end']
                        assert block['playback_start'] == (
                            block['source_start']
                            if trace['selection'].get('tracking_policy')
                            in ('current_support', 'per_head_support')
                            else min(
                                previous_segment['playback_start'],
                                block['source_start'],
                            )
                        )
                        assert block['playback_end'] == (
                            block['source_end']
                            if trace['selection'].get('tracking_policy')
                            in ('current_support', 'per_head_support')
                            else max(
                                previous_segment['playback_end'],
                                block['source_end'],
                            )
                        )
                    else:
                        assert identifier not in seen
                        assert block['playback_start'] == block['source_start']
                        assert block['playback_end'] == block['source_end']
                    current[identifier] = block
                    seen.add(identifier)
            previous = current
        for head in trace['heads']:
            flattened = [
                {**segment, 'voice_id': voice['voice_id']}
                for voice in head['voices']
                for segment in voice['playback']
            ]
            assert (
                sorted(flattened, key=lambda item: item['output_start'])
                == head['playback']
            )
    for index, head in enumerate(trace['heads']):
        assert head['index'] == index
        assert len(head['selections']) == len(queries)
        previous_end = 0
        for query_index, item in enumerate(head['selections']):
            assert item['query_index'] == query_index
            query = item['query_sample']
            assert query == queries[query_index]
            for block in item['candidates'] + item.get(
                'selected_blocks',
                [item['selected']] if item['selected'] else [],
            ):
                if 'mean_attention' in block:
                    assert block['mean_attention'] > block['threshold']
                if 'playback_start' in block:
                    assert 0 <= block['playback_start'] <= block['source_start']
                    assert block['source_end'] <= block['playback_end'] <= count
                assert 0 <= block['key_start'] < block['key_end'] < len(edges)
                assert block['source_start'] == edges[block['key_start']]
                assert block['source_end'] == edges[block['key_end']]
                assert (
                    block['source_end'] <= query - guard
                    or block['source_start'] > query + guard
                )
        for voice in head.get('voices', [{'playback': head['playback']}]):
            previous_end = 0
            for segment in voice['playback']:
                start, end = segment['output_start'], segment['output_end']
                left, right = segment['source_start'], segment['source_end']
                assert previous_end <= start < end <= count
                assert 0 <= left < right <= count
                assert end - start == right - left
                # Source and output advance at 1x, so the offset is constant.
                assert abs(left - start) > guard
                assert (
                    segment['fade_in_samples'] + segment['fade_out_samples']
                    <= end - start
                )
                previous_end = end
                total_segments += 1
                events.extend([(start, 1), (end, -1)])
    maximum_active = 0
    if trace['selection'].get('tracking_policy') in (
        'current_support',
        'per_head_support',
    ):
        support = playback_audit.support_statistics(trace)
        assert support['unselected_samples'] == 0
        assert support['unsupported_samples'] == 0
    active = 0
    for _, delta in sorted(events):
        active += delta
        maximum_active = max(maximum_active, active)
    if 'near_monitor' in trace:
        assert maximum_active <= voice_budget
        monitor = trace['near_monitor']
        assert 0 <= monitor['relative_gain'] < 1
        expected = []
        for query_index, query in enumerate(queries):
            items = [head['selections'][query_index] for head in trace['heads']]
            if any(item['near_attention']['active'] for item in items):
                end = items[0]['output_end']
                if expected and expected[-1][1] == query:
                    expected[-1][1] = end
                else:
                    expected.append([query, end])
        assert expected == [
            [item['output_start'], item['output_end']]
            for item in monitor['playback']
        ]
        previous_end = 0
        for segment in monitor['playback']:
            assert (
                previous_end
                <= segment['output_start']
                < segment['output_end']
                <= count
            )
            assert segment['output_start'] == segment['source_start']
            assert segment['output_end'] == segment['source_end']
            previous_end = segment['output_end']
    extra = (
        [trace['audio']['near_stem']] if trace['audio'].get('near_stem') else []
    )
    for filename in filter(
        None,
        (
            extra
            + trace['audio']['stems']
            + [
                trace['audio']['mix'],
                source['path'],
            ]
        ),
    ):
        data, actual_rate = soundfile.read(trace_path.parent / filename)
        assert actual_rate == sample_rate and len(data) == count
        assert np.isfinite(data).all()
    if 'video' in trace['rendering']:
        metadata = json.loads(
            subprocess.check_output(
                [
                    io.executable('ffprobe'),
                    '-v',
                    'error',
                    '-show_streams',
                    '-of',
                    'json',
                    str(trace_path.parent / trace['rendering']['video']),
                ]
            )
        )
        for stream in metadata['streams']:
            assert abs(float(stream['duration']) - count / sample_rate) <= (
                1 / trace['rendering']['fps'] + 0.01
            )
    return {
        'heads': len(trace['heads']),
        'queries': len(queries),
        'playback_segments': total_segments,
        'duration_seconds': count / sample_rate,
        'guard_violations': 0,
        'maximum_remote_voices': maximum_active,
    }
