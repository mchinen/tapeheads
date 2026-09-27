"""Validate recursive path provenance, sample clocks, gain, and assets."""

import json
import pathlib
import subprocess

import jsonschema
import numpy as np
import soundfile

from tapeheads import io
from tapeheads import trace_types


def validate(
    trace_path: trace_types.PathLike, trace: trace_types.JsonObject
) -> trace_types.JsonObject:
    """Independently audit paths against the two original layer schedules."""
    from tapeheads import validation

    trace_path = pathlib.Path(trace_path)
    root = trace_path.parent
    schema = json.loads((io.ROOT / 'schemas/recursive.schema.json').read_text())
    jsonschema.validate(trace, schema)
    layer_results = []
    tracks = []
    for index, (layer, asset) in enumerate(
        zip(trace['layers'], trace['layer_assets'])
    ):
        path = root / asset['directory'] / 'trace.json'
        assert io.sha256(path) == asset['trace_sha256']
        assert json.loads(path.read_text()) == layer
        assert layer['model']['layer_index'] == index
        layer_results.append(validation.validate(path))
        mapping = {
            (head['index'], voice['voice_id']): voice['playback']
            for head in layer['heads']
            for voice in head['voices']
        }
        mapping[(-1, -1)] = layer['near_monitor']['playback']
        tracks.append(mapping)
    first, second = trace['layers']
    assert first['frames'] == second['frames']
    assert first['encoder'] == second['encoder'] == trace['encoder']
    assert first['model']['revision'] == second['model']['revision']
    assert (
        io.sha256(root / trace['source']['path']) == trace['source']['sha256']
    )
    source, sample_rate = soundfile.read(root / trace['source']['path'])
    count = len(source)
    assert count == trace['source']['samples']
    assert sample_rate == trace['source']['sample_rate_hz']
    other, other_rate = soundfile.read(
        root / trace['layer_assets'][1]['directory'] / second['source']['path']
    )
    assert other_rate == sample_rate and np.array_equal(source, other)
    guard = second['selection']['guard_samples']
    parent_gain = 0.8 / (
        second['selection']['max_layer_blocks']
        + second['near_monitor']['relative_gain']
    )
    child_gain = 1 / (
        first['selection']['max_layer_blocks']
        + first['near_monitor']['relative_gain']
    )
    quiet_paths = 0
    previous_start = 0
    direct_seen = set()
    for event in trace['events']:
        start, end = event['output_start'], event['output_end']
        left, right = event['source_start'], event['source_end']
        middle = event['intermediate_start']
        assert previous_start <= start < end <= count
        previous_start = start
        assert 0 <= left < right <= count and right - left == end - start
        parent_key = (event['parent_head'], event['parent_voice'])
        parent = tracks[1][parent_key][event['parent_segment']]
        assert event['parent_envelope'] == parent
        assert parent['output_start'] <= start < end <= parent['output_end']
        assert middle == parent['source_start'] + start - parent['output_start']
        near = abs(left - start) <= guard
        assert event['near_output'] == near
        relative = (
            second['near_monitor']['relative_gain']
            if event['parent_head'] == -1
            else 1
        )
        if event['branch'] == 'direct':
            assert left == middle
            assert (
                start == parent['output_start'] and end == parent['output_end']
            )
            identifier = (*parent_key, event['parent_segment'])
            assert identifier not in direct_seen
            direct_seen.add(identifier)
            gain = parent_gain * relative * trace['recursion']['direct_weight']
        else:
            child = tracks[0][(event['child_head'], event['child_voice'])][
                event['child_segment']
            ]
            assert event['child_envelope'] == child
            assert (
                child['output_start']
                <= middle
                < middle + end - start
                <= child['output_end']
            )
            assert (
                left == child['source_start'] + middle - child['output_start']
            )
            child_relative = (
                first['near_monitor']['relative_gain']
                if event['child_head'] == -1
                else 1
            )
            gain = (
                parent_gain
                * relative
                * trace['recursion']['recursive_weight']
                * child_gain
                * child_relative
            )
            if near:
                gain *= second['near_monitor']['relative_gain']
                quiet_paths += 1
        assert np.isclose(event['gain'], gain, rtol=0, atol=1e-12)
    assert len(direct_seen) == sum(
        len(segments) for segments in tracks[1].values()
    )
    assert np.isclose(
        trace['recursion']['direct_weight']
        + trace['recursion']['recursive_weight'],
        1,
    )
    for filename in trace['audio']['stems'] + [trace['audio']['mix']]:
        data, actual_rate = soundfile.read(root / filename)
        assert actual_rate == sample_rate and data.shape == (count, 2)
        assert np.isfinite(data).all()
        assert np.abs(data).max() <= 0.8 * np.abs(source).max() + 1e-6
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
                    str(root / trace['rendering']['video']),
                ]
            )
        )
        for stream in metadata['streams']:
            assert (
                abs(float(stream['duration']) - count / sample_rate)
                <= 1 / trace['rendering']['fps'] + 0.01
            )
    return {
        'layers': layer_results,
        'recursive_events': len(trace['events']),
        'quiet_return_paths': quiet_paths,
        'duration_seconds': count / sample_rate,
        'peak': trace['audio']['peak'],
        'path_clock_violations': 0,
    }
