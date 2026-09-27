"""Audit deep shared schedules and reconstruct every exported depth bus."""

import json
import pathlib
import subprocess

import jsonschema
import numpy as np
import soundfile

from tapeheads import deep
from tapeheads import io
from tapeheads import responsive_audio
from tapeheads import shared_audio
from tapeheads import trace_types


def validate(
    path: trace_types.PathLike, trace: trace_types.JsonObject
) -> trace_types.JsonObject:
    """Check native provenance and sample-accurate shared-graph audio."""
    from tapeheads import validation

    schema = json.loads((io.ROOT / 'schemas/deep.schema.json').read_text())
    jsonschema.validate(trace, schema)
    root = pathlib.Path(path).parent
    source, sample_rate = soundfile.read(
        root / trace['source']['path'], dtype='float32'
    )
    assert len(source) == trace['source']['samples']
    assert sample_rate == trace['source']['sample_rate_hz']
    assert (
        io.sha256(root / trace['source']['path']) == trace['source']['sha256']
    )
    assert 1 <= len(trace['layers']) <= 8
    buses = []
    native_results = []
    for index, (layer, asset) in enumerate(
        zip(trace['layers'], trace['audio']['depths'])
    ):
        native_path = root / layer['native_trace']
        assert io.sha256(native_path) == layer['native_trace_sha256']
        native = io.read_json(native_path)
        assert native['model'] == layer['model']
        assert native['model']['layer_index'] == index
        assert native['frames'] == trace['frames']
        assert native['selection'] == layer['selection']
        assert native['near_monitor'] == layer['near_monitor']
        for actual, expected in zip(layer['heads'], native['heads']):
            assert actual == {
                'index': expected['index'],
                'voices': expected['voices'],
            }
        native_results.append(validation.validate(native_path))
        data, actual_rate = soundfile.read(
            root / asset['path'], dtype='float32'
        )
        assert actual_rate == sample_rate and data.shape == (len(source), 2)
        assert np.isfinite(data).all() and np.abs(data).max() <= 0.800001
        if trace['recursion']['version'] == 'per-head-shared/1':
            reconstructed, statistics = shared_audio.render_depth(
                trace['layers'][: index + 1], source, sample_rate
            )
        elif trace['recursion']['version'] == 'responsive-routes/1':
            reconstructed, statistics = responsive_audio.render_depth(
                trace['layers'][: index + 1], source, sample_rate
            )
        else:
            reconstructed, statistics = deep.render_bus(
                layer, source, buses[-1] if buses else None
            )
        assert np.allclose(reconstructed, data, rtol=0, atol=1e-6)
        assert np.isclose(statistics['gain'], asset['gain'])
        buses.append(data)
    assert len(trace['audio']['depths']) == len(trace['layers'])
    assert trace['stages'] == deep.progression(len(source), len(buses))
    expected = deep.progressive_mix(buses, trace['stages'], sample_rate)
    actual, actual_rate = soundfile.read(
        root / trace['audio']['mix'], dtype='float32'
    )
    assert actual_rate == sample_rate and np.allclose(
        expected, actual, rtol=0, atol=1e-6
    )
    previous = 0
    for section in trace['annotations']['sections']:
        assert section['start'] == previous
        assert previous < section['end'] <= len(source)
        previous = section['end']
    assert previous == len(source)
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
                abs(float(stream['duration']) - len(source) / sample_rate)
                <= 1 / trace['rendering']['fps'] + 0.01
            )
    return {
        'layers': len(buses),
        'native_checks': native_results,
        'duration_seconds': len(source) / sample_rate,
        'reconstruction_max_error_bound': 1e-6,
        'peak': float(np.abs(actual).max()),
        'annotations': len(trace['annotations']['sections']),
    }
