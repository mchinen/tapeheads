"""Render arbitrary-depth tape recursion as a shared schedule graph."""

import dataclasses
import os
import pathlib

import numpy as np
import soundfile
import torch

from tapeheads import annotations
from tapeheads import io
from tapeheads import pipeline
from tapeheads import recursive
from tapeheads import responsive_audio
from tapeheads import selection
from tapeheads import shared_audio
from tapeheads import trace_types


def render_bus(
    layer: trace_types.JsonObject,
    source: np.ndarray,
    previous: np.ndarray | None,
    direct_weight: float = 0.2,
) -> tuple[np.ndarray, dict[str, float]]:
    """Mix source audio and the preceding layer at attended sample positions.

    Args:
        layer: Native voice schedules and gain configuration for this layer.
        source: Mono input audio at the analysis sample rate.
        previous: Stereo bus from the preceding layer, or None for layer one.
        direct_weight: Source-audio share when a preceding bus is present.

    Returns:
        The stereo bus and its applied gain, RMS level, and peak. Every
        scheduled voice contributes; fades multiply through successive buses.
    """
    result = np.zeros((len(source), 2), dtype=np.float32)
    budget = (
        layer['selection']['max_layer_blocks']
        + layer['near_monitor']['relative_gain']
    )
    for head, _, relative_gain, segments in recursive.tracks(layer):
        pan = head / max(1, len(layer['heads']) - 1) if head >= 0 else 0.5
        stereo = np.array([np.cos(pan * np.pi / 2), np.sin(pan * np.pi / 2)])
        for segment in segments:
            output_start, output_end = (
                segment['output_start'],
                segment['output_end'],
            )
            source_start, source_end = (
                segment['source_start'],
                segment['source_end'],
            )
            direct = source[source_start:source_end, None] * stereo[None, :]
            if previous is None:
                values = direct
            else:
                values = (
                    direct_weight * direct
                    + (1 - direct_weight)
                    * previous[source_start:source_end]
                    * stereo[None, :]
                )
            fade = recursive.envelope(
                segment, np.arange(output_start, output_end)
            )
            result[output_start:output_end] += (
                relative_gain / budget * fade[:, None] * values
            ).astype(np.float32)
    root_mean_square = float(np.sqrt(np.mean(result**2)))
    peak = float(np.abs(result).max())
    gain = (
        min(4.0, 0.1 / max(root_mean_square, 1e-12), 0.8 / max(peak, 1e-12))
        if peak
        else 1.0
    )
    result *= gain
    return result, {
        'gain': gain,
        'rms': float(np.sqrt(np.mean(result**2))),
        'peak': float(np.abs(result).max()),
    }


def progression(samples: int, count: int) -> list[dict[str, int]]:
    """Return equal output-time stages with increasing recursive depth."""
    edges = np.linspace(0, samples, count + 1).astype(int)
    return [
        {'start': int(output_start), 'end': int(output_end), 'depth': index + 1}
        for index, (output_start, output_end) in enumerate(
            zip(edges[:-1], edges[1:])
        )
    ]


def progressive_mix(
    buses: list[np.ndarray], stages: list[dict[str, int]], sample_rate: int
) -> np.ndarray:
    """Switch depth with short complementary linear crossfades."""
    result = np.zeros_like(buses[0])
    for stage in stages:
        output_start, output_end, depth = (
            stage['start'],
            stage['end'],
            stage['depth'],
        )
        result[output_start:output_end] = buses[depth - 1][
            output_start:output_end
        ]
        if depth > 1:
            fade = min(round(0.08 * sample_rate), output_end - output_start)
            weight = np.linspace(0, 1, fade)[:, None]
            result[output_start : output_start + fade] = (1 - weight) * buses[
                depth - 2
            ][output_start : output_start + fade] + weight * buses[depth - 1][
                output_start : output_start + fade
            ]
    return result


def analyze(
    input_path: trace_types.PathLike,
    output: trace_types.PathLike,
    encoder: str = 'muq',
    layers: int = 8,
    voices: int = 6,
    duration: float | None = None,
    verify: bool = False,
    config: selection.SelectionConfig | None = None,
    threads: int = 4,
    block_size: int = 128,
    single_pass: bool = False,
    compress: bool = False,
    compact_audio: bool = False,
) -> trace_types.JsonObject:
    """Extract native layers, save compact graph and all depth buses."""
    if not 1 <= voices <= 16:
        raise ValueError('Deep views support between one and sixteen voices')
    if not 1 <= layers <= 8 or encoder == 'clap':
        raise ValueError('Use 1–8 layers of muq, wav2vec2, or hubert')
    output = pathlib.Path(output)
    root_trace = output / ('trace.json.gz' if compact_audio else 'trace.json')
    if root_trace.exists():
        raise FileExistsError('Choose a new output directory')
    config = config or selection.SelectionConfig(
        entry_alpha=1.5,
        exit_alpha=1.1,
        max_layer_blocks=voices,
        shortlist_size=voices,
        average_window_seconds=0.35,
        minimum_block_seconds=0.3,
    )
    responsive = config.tracking_policy == 'current_support'
    if responsive and layers > 3:
        raise ValueError(
            'Responsive route composition supports one to three layers'
        )
    views = None
    if single_pass and not all(
        (
            output
            / f'layer_{index + 1:02d}'
            / ('trace.json.gz' if compress else 'trace.json')
        ).exists()
        for index in range(layers)
    ):
        from tapeheads import encoders
        from tapeheads import multilayer

        torch.set_num_threads(threads)
        analysis_audio = io.decode(
            input_path,
            duration_seconds=duration,
            sample_rate=encoders.ENCODERS[encoder].sample_rate,
        )
        views = multilayer.create_views(
            encoder, analysis_audio, layers, block_size
        )
        print(f'Captured {layers} layers in one forward pass', flush=True)
    graph = []
    source = None
    buses = []
    audio_metadata = []
    for index in range(layers):
        directory = output / f'layer_{index + 1:02d}'
        path = directory / ('trace.json.gz' if compress else 'trace.json')
        if path.exists():
            layer = io.read_json(path)
            assert layer['encoder'] == encoder
            assert layer['model']['layer_index'] == index
            assert layer['source']['context_mode'] == (
                'excerpt' if duration is not None else 'full'
            )
            assert layer['source']['original_sha256'] == io.sha256(input_path)
            for key, value in dataclasses.asdict(config).items():
                assert layer['selection'][key] == value
            if duration is not None:
                assert (
                    abs(layer['source']['duration_seconds'] - duration) < 0.01
                )
        else:
            layer = pipeline.analyze(
                input_path,
                directory,
                encoder=encoder,
                duration=duration,
                config=config,
                layer_index=index,
                verify=verify,
                threads=threads,
                block_size=block_size,
                view=views[index] if views else None,
                compress=compress,
                compact_audio=compact_audio,
            )
        if compact_audio and source is not None:
            shared_path = output / 'layer_01' / layer['source']['path']
            current_path = directory / layer['source']['path']
            if io.sha256(shared_path) == io.sha256(
                current_path
            ) and not os.path.samefile(shared_path, current_path):
                current_path.unlink()
                os.link(shared_path, current_path)
        if source is None:
            source, sample_rate = soundfile.read(
                directory / layer['source']['path'], dtype='float32'
            )
            source_metadata = {
                **layer['source'],
                'path': 'layer_01/' + layer['source']['path'],
            }
            frame_mapping = layer['frames']
        assert layer['frames'] == frame_mapping
        assert layer['source']['samples'] == len(source)
        graph.append(
            {
                'model': layer['model'],
                'selection': layer['selection'],
                'heads': [
                    {'index': head['index'], 'voices': head['voices']}
                    for head in layer['heads']
                ],
                'near_monitor': layer['near_monitor'],
                'native_trace': str(path.relative_to(output)),
                'native_trace_sha256': io.sha256(path),
            }
        )
        if config.tracking_policy == 'per_head_support':
            bus, statistics = shared_audio.render_depth(
                graph, source, sample_rate
            )
        elif responsive:
            bus, statistics = responsive_audio.render_depth(
                graph, source, sample_rate
            )
        else:
            bus, statistics = render_bus(
                layer, source, buses[-1] if buses else None
            )
        buses.append(bus)
        filename = f'depth_{index + 1:02d}.' + (
            'flac' if compact_audio else 'wav'
        )
        soundfile.write(
            output / filename,
            bus,
            sample_rate,
            subtype='PCM_24' if compact_audio else 'FLOAT',
        )
        audio_metadata.append({'path': filename, **statistics})
        print(f'Recursive depth {index + 1}/{layers} rendered', flush=True)
    stages = progression(len(source), layers)
    mix = progressive_mix(buses, stages, sample_rate)
    mix_name = 'mix.flac' if compact_audio else 'mix.wav'
    soundfile.write(
        output / mix_name,
        mix,
        sample_rate,
        subtype='PCM_24' if compact_audio else 'FLOAT',
    )
    trace = {
        'format': 'tapeheads/4',
        'encoder': encoder,
        'source': source_metadata,
        'layers': graph,
        'frames': frame_mapping,
        'stages': stages,
        'annotations': (
            annotations.study_sections
            if config.tracking_policy == 'per_head_support'
            else annotations.create
        )(source, sample_rate, pathlib.Path(input_path).parent.name),
        'recursion': {
            'version': 'shared-tape-graph/1',
            'direct_weight': 0.2,
            'recursive_weight': 0.8,
            'audio_pruning': False,
            'normalization': 'per-layer RMS target 0.1; gain capped at 4; peak capped at 0.8',
            'near_policy': 'each hop obeys local guard; near monitor quiet; composed returns to output are not separately attenuated',
            'depth_crossfade_seconds': 0.08,
            'meaning': 'artistic recursion through mixed frame representations, not head-to-head attribution',
        },
        'audio': {
            'mix': mix_name,
            'depths': audio_metadata,
            'peak': float(np.abs(mix).max()),
        },
        'rendering': {
            'width': 1920,
            'height': 1080,
            'fps': 30,
            'style': 'deep-dark-field/1',
            'max_visual_edges': 128,
            'visual_policy': 'bounded complete routes per root; longer runs win ties; quiet branches hidden; audio includes every branch',
        },
    }
    if responsive:
        trace['recursion'].update(
            {
                'version': 'responsive-routes/1',
                'direct_weight': 0.35,
                'recursive_weight': 0.65,
                'coincident_source_policy': 'mean path gains at identical source offsets',
                'boundary_treatment': 'one 3 ms decaying offset correction per output splice; no multiplied per-hop fades',
                'near_policy': 'local guard at each hop; composed returns within output guard attenuated by near_gain',
            }
        )
        trace['rendering']['show_intermediate_audio'] = True
        if trace['annotations']['method'] != 'generator manifest':
            trace['annotations'] = {
                'method': 'unlabeled; awaiting supplied timed lyrics',
                'sections': [
                    {
                        'start': 0,
                        'end': len(source),
                        'label': '',
                        'confidence': 'unlabeled',
                    }
                ],
                'lyrics': [],
            }
    if config.tracking_policy == 'per_head_support':
        trace['recursion'].update(
            {
                'version': 'per-head-shared/1',
                'direct_weight': 0.35,
                'recursive_weight': 0.65,
                'normalization': 'once per fixed-depth mix; raw buses propagate',
                'coincident_source_policy': 'mean gains for identical reads within each layer; cross-layer paths share convex buses',
                'boundary_treatment': 'one final 3 ms splice correction; no inherited fades',
                'near_policy': 'local guard and quiet monitor at each hop; no extra composed-return attenuation',
            }
        )
        trace['rendering'].update(
            {
                'show_intermediate_audio': True,
                'tick_seconds': 15,
                'max_visual_edges': 384,
            }
        )
    io.write_json(root_trace, trace)
    return trace
