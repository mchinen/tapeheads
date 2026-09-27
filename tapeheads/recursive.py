"""Compose second-layer tape paths through first-layer playback schedules."""

import bisect
import pathlib
import time
from collections.abc import Iterator

import numpy as np
import soundfile

from tapeheads import io
from tapeheads import pipeline
from tapeheads import selection
from tapeheads import trace_types


def envelope(
    segment: trace_types.PlaybackSegment, positions: np.ndarray
) -> np.ndarray:
    """Evaluate an existing segment's boundary fades at integer positions."""
    result = np.ones(len(positions), dtype=np.float32)
    fade = segment['fade_in_samples']
    if fade:
        result *= np.clip(
            (positions - segment['output_start']) / max(1, fade - 1), 0, 1
        )
    fade = segment['fade_out_samples']
    if fade:
        result *= np.clip(
            (segment['output_end'] - 1 - positions) / max(1, fade - 1), 0, 1
        )
    return result


def tracks(
    trace: trace_types.JsonObject,
) -> Iterator[tuple[int, int, float, list[trace_types.PlaybackSegment]]]:
    """Expose remote voices and the shared quiet monitor uniformly."""
    for head in trace['heads']:
        for voice in head['voices']:
            yield head['index'], voice['voice_id'], 1.0, voice['playback']
    yield -1, -1, trace['near_monitor']['relative_gain'], trace['near_monitor'][
        'playback'
    ]


def compose(
    first: trace_types.JsonObject,
    second: trace_types.JsonObject,
    direct_gain: float = 0.35,
    recursive_gain: float = 0.65,
) -> list[trace_types.JsonObject]:
    """Flatten both clocks into original-source runs with explicit path IDs.

    Each parent reads the first-layer timeline at its source cursor. Child
    playback is clipped to that read interval, preserving its source offset.
    All selected child voices participate; no dense attention rollout is used.
    """
    guard = second['selection']['guard_samples']
    parent_gain = 0.8 / (
        second['selection']['max_layer_blocks']
        + second['near_monitor']['relative_gain']
    )
    child_gain = 1 / (
        first['selection']['max_layer_blocks']
        + first['near_monitor']['relative_gain']
    )
    children = []
    for head, voice, relative_gain, segments in tracks(first):
        children.append(
            (
                head,
                voice,
                relative_gain,
                segments,
                [item['output_end'] for item in segments],
            )
        )
    events = []
    for parent_head, parent_voice, relative_gain, segments in tracks(second):
        for parent_index, parent in enumerate(segments):
            base = {
                'parent_head': parent_head,
                'parent_voice': parent_voice,
                'parent_segment': parent_index,
                'parent_envelope': parent,
            }
            events.append(
                {
                    **base,
                    'branch': 'direct',
                    'output_start': parent['output_start'],
                    'output_end': parent['output_end'],
                    'source_start': parent['source_start'],
                    'source_end': parent['source_end'],
                    'intermediate_start': parent['source_start'],
                    'gain': parent_gain * relative_gain * direct_gain,
                    'child_head': None,
                    'child_voice': None,
                    'near_output': abs(
                        parent['source_start'] - parent['output_start']
                    )
                    <= guard,
                }
            )
            for (
                child_head,
                child_voice,
                child_relative,
                child_segments,
                ends,
            ) in children:
                index = bisect.bisect_right(ends, parent['source_start'])
                for child_index in range(index, len(child_segments)):
                    child = child_segments[child_index]
                    intermediate_start = max(
                        parent['source_start'], child['output_start']
                    )
                    intermediate_end = min(
                        parent['source_end'], child['output_end']
                    )
                    if child['output_start'] >= parent['source_end']:
                        break
                    if intermediate_start >= intermediate_end:
                        continue
                    output_start = (
                        parent['output_start']
                        + intermediate_start
                        - parent['source_start']
                    )
                    source_start = (
                        child['source_start']
                        + intermediate_start
                        - child['output_start']
                    )
                    near_output = abs(source_start - output_start) <= guard
                    gain = (
                        parent_gain
                        * relative_gain
                        * recursive_gain
                        * child_gain
                        * child_relative
                    )
                    if near_output:
                        gain *= second['near_monitor']['relative_gain']
                    events.append(
                        {
                            **base,
                            'branch': 'recursive',
                            'output_start': output_start,
                            'output_end': output_start
                            + intermediate_end
                            - intermediate_start,
                            'source_start': source_start,
                            'source_end': source_start
                            + intermediate_end
                            - intermediate_start,
                            'intermediate_start': intermediate_start,
                            'child_head': child_head,
                            'child_voice': child_voice,
                            'child_segment': child_index,
                            'child_envelope': child,
                            'gain': gain,
                            'near_output': near_output,
                        }
                    )
    return sorted(
        events, key=lambda event: (event['output_start'], event['branch'])
    )


def render_audio(
    trace: trace_types.JsonObject,
    source: np.ndarray,
    output: trace_types.PathLike,
) -> trace_types.JsonObject:
    """Render composed runs using both inherited fade envelopes."""
    output = pathlib.Path(output)
    stems = {
        name: np.zeros((len(source), 2), dtype=np.float32)
        for name in ('direct', 'recursive')
    }
    heads = len(trace['layers'][1]['heads'])
    for event in trace['events']:
        output_start, output_end = event['output_start'], event['output_end']
        positions = np.arange(output_start, output_end)
        samples = source[event['source_start'] : event['source_end']].copy()
        samples *= envelope(event['parent_envelope'], positions)
        if event['branch'] == 'recursive':
            samples *= envelope(
                event['child_envelope'],
                event['intermediate_start'] + positions - output_start,
            )
        samples *= event['gain']
        pan = (
            event['parent_head'] / max(1, heads - 1)
            if event['parent_head'] >= 0
            else 0.5
        )
        stems[event['branch']][output_start:output_end, 0] += samples * np.cos(
            pan * np.pi / 2
        )
        stems[event['branch']][output_start:output_end, 1] += samples * np.sin(
            pan * np.pi / 2
        )
    mix = stems['direct'] + stems['recursive']
    sample_rate = trace['source']['sample_rate_hz']
    for name, audio in {**stems, 'mix': mix}.items():
        soundfile.write(
            output / f'{name}.wav', audio, sample_rate, subtype='FLOAT'
        )
    return {
        'mix': 'mix.wav',
        'stems': ['direct.wav', 'recursive.wav'],
        'peak': float(np.abs(mix).max()),
        'rms': float(np.sqrt(np.mean(mix**2))),
        'normalization': 'none; fixed branch and voice gains',
        'pan': 'parent head index, equal-power stereo; nearby monitor centered',
        'boundary_treatment': 'product of parent and child boundary fades',
    }


def analyze(
    input_path: trace_types.PathLike,
    output: trace_types.PathLike,
    encoder: str = 'wav2vec2',
    duration: float | None = None,
    config: selection.SelectionConfig | None = None,
    threads: int = 4,
    verify: bool = False,
) -> trace_types.JsonObject:
    """Extract two native layers and save an auditable recursive performance."""
    if encoder == 'clap':
        raise ValueError(
            'Recursive CLAP needs a validated shifted-window mapping; use wav2vec2, hubert, or muq'
        )
    output = pathlib.Path(output)
    if (output / 'trace.json').exists():
        raise FileExistsError('Choose a new recursive output directory')
    started = time.monotonic()
    layers = []
    for index in range(2):
        directory = output / f'layer_{index + 1:02d}'
        layers.append(
            pipeline.analyze(
                input_path,
                directory,
                encoder=encoder,
                duration=duration,
                config=config,
                threads=threads,
                verify=verify,
                layer_index=index,
            )
        )
    first, second = layers
    assert first['frames'] == second['frames']
    first_audio, first_rate = soundfile.read(
        output / 'layer_01/source.wav', dtype='float32'
    )
    second_audio, second_rate = soundfile.read(
        output / 'layer_02/source.wav', dtype='float32'
    )
    if first_rate != second_rate or not np.array_equal(
        first_audio, second_audio
    ):
        raise ValueError('Layer inputs must contain identical audio samples')
    trace = {
        'format': 'tapeheads/3',
        'encoder': encoder,
        'source': {**first['source'], 'path': 'layer_01/source.wav'},
        'layers': layers,
        'layer_assets': [
            {
                'directory': f'layer_{index + 1:02d}',
                'trace_sha256': io.sha256(
                    output / f'layer_{index + 1:02d}' / 'trace.json'
                ),
            }
            for index in range(2)
        ],
        'recursion': {
            'version': 'tape-path-composition/1',
            'direct_weight': 0.35,
            'recursive_weight': 0.65,
            'child_policy': 'all active selected first-layer playback voices at the intermediate cursor, including quiet monitor',
            'near_policy': 'recursive leaves within output guard receive additional near_gain',
            'meaning': 'artistic tape mapping, not exact attribution or attention rollout',
            'clocks': 'output -> intermediate first-layer timeline -> original source',
        },
        'events': compose(first, second),
        'rendering': {'width': 1920, 'height': 1080, 'fps': 30},
    }
    source, _ = soundfile.read(
        output / trace['source']['path'], dtype='float32'
    )
    trace['audio'] = render_audio(trace, source, output)
    trace['elapsed_seconds'] = time.monotonic() - started
    io.write_json(output / 'trace.json', trace)
    print(f'Saved recursive trace: {output}/trace.json', flush=True)
    return trace
