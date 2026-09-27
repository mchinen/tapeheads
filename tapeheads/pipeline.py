"""Build portable traces and stems using the selected encoder adapter."""

import dataclasses
import json
import pathlib
import time

import numpy as np
import soundfile
import torch

from tapeheads import encoders
from tapeheads import io
from tapeheads import per_head
from tapeheads import playback
from tapeheads import selection
from tapeheads import sustained
from tapeheads import trace_types


def analyze(
    input_path: trace_types.PathLike,
    output: trace_types.PathLike,
    encoder: str = 'wav2vec2',
    start: float = 0,
    duration: float | None = None,
    config: selection.SelectionConfig | None = None,
    block_size: int = 128,
    threads: int = 4,
    verify: bool = False,
    layer_index: int = 0,
    view: encoders.AttentionView | None = None,
    compress: bool = False,
    compact_audio: bool = False,
) -> trace_types.JsonObject:
    """Extract first-layer attention and save source, trace, and head stems."""
    if start < 0 or (duration is not None and duration <= 0):
        raise ValueError('Start must be nonnegative and duration positive')
    if block_size < 1 or threads < 1:
        raise ValueError('Block size and thread count must be positive')
    torch.set_num_threads(threads)
    config = config or selection.SelectionConfig()
    encoder_spec = encoders.ENCODERS[encoder]
    input_path = pathlib.Path(input_path).resolve()
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / ('trace.json.gz' if compress else 'trace.json')
    if trace_path.exists():
        raise FileExistsError(
            'Choose a new output directory; trace already exists'
        )
    started = time.monotonic()
    audio = io.decode(input_path, start, duration, encoder_spec.sample_rate)
    source_path = output / ('source.flac' if compact_audio else 'source.wav')
    soundfile.write(
        source_path,
        audio,
        encoder_spec.sample_rate,
        subtype='PCM_24' if compact_audio else 'FLOAT',
    )
    print(
        f'Loading {encoder}; {len(audio) / encoder_spec.sample_rate:.2f} s input',
        flush=True,
    )
    view = view or encoders.create_view(
        encoder, audio, block_size, verify, layer_index
    )
    if len(view.edges) != len(view.queries) + 1:
        raise AssertionError('Time mapping must have one more edge than query')
    if (np.diff(view.edges) <= 0).any():
        raise AssertionError('Source time-bin edges must increase')
    selector_class = (
        per_head.PerHeadSelector
        if config.tracking_policy == 'per_head_support'
        else sustained.LayerSelector
    )
    selector = selector_class(config, view.edges, view.sample_rate, view.heads)
    guard = round(config.guard_seconds * view.sample_rate)
    voices = {}
    voice_heads = {}
    monitor = []
    heads = [{'index': index, 'selections': []} for index in range(view.heads)]
    for query_index, probabilities, valid in view.rows():
        query_sample = int(view.queries[query_index])
        end = (
            int(view.queries[query_index + 1])
            if query_index + 1 < len(view.queries)
            else len(audio)
        )
        diagnostics, active = selector.select(
            probabilities, query_sample, valid
        )
        for index, item in enumerate(diagnostics):
            item.update(
                {
                    'query_index': query_index,
                    'query_sample': query_sample,
                    'output_end': end,
                }
            )
            heads[index]['selections'].append(item)
        for voice in active:
            identifier = voice['voice_id']
            if identifier not in voices:
                voices[identifier] = playback.TapeHead(guard)
                voice_heads[identifier] = voice['head']
            voices[identifier].append(
                query_sample,
                end,
                {
                    'source_start': voice['playback_start'],
                    'source_end': voice['playback_end'],
                },
            )
        if any(item['near_attention']['active'] for item in diagnostics):
            if monitor and monitor[-1]['output_end'] == query_sample:
                monitor[-1]['output_end'] = end
                monitor[-1]['source_end'] = end
            else:
                monitor.append(
                    {
                        'output_start': query_sample,
                        'output_end': end,
                        'source_start': query_sample,
                        'source_end': end,
                    }
                )
        if query_index % 500 == 0:
            print(f'Query {query_index + 1}/{len(view.queries)}', flush=True)
    for head in heads:
        head['voices'] = []
        head['playback'] = []
    for identifier, tape in voices.items():
        segments = playback.finalize_segments(
            tape.segments, view.sample_rate, guard
        )
        head = heads[voice_heads[identifier]]
        head['voices'].append({'voice_id': identifier, 'playback': segments})
        for segment in segments:
            head['playback'].append({**segment, 'voice_id': identifier})
    for head in heads:
        head['playback'].sort(key=lambda segment: segment['output_start'])
    for segment in monitor:
        fade = min(
            round(0.005 * view.sample_rate),
            (segment['output_end'] - segment['output_start']) // 2,
        )
        segment.update({'fade_in_samples': fade, 'fade_out_samples': fade})
    title = input_path.parent.name.replace('_', ' ').title()
    manifest_path = io.ROOT / 'data' / 'sources.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        title = next(
            (
                entry['title']
                for entry in manifest['entries']
                if entry['id'] == input_path.parent.name
            ),
            title,
        )
    trace = {
        'format': 'tapeheads/2',
        'encoder': encoder,
        'source': {
            'path': source_path.name,
            'playback_encoding': 'FLAC PCM24'
            if compact_audio
            else 'WAV float32',
            'title': title,
            'sha256': io.sha256(source_path),
            'original_name': input_path.name,
            'original_sha256': io.sha256(input_path),
            'input_start_seconds': start,
            'context_mode': 'excerpt'
            if duration is not None or start
            else 'full',
            'sample_rate_hz': view.sample_rate,
            'samples': len(audio),
            'channels': 1,
            'duration_seconds': len(audio) / view.sample_rate,
            'playback_bandwidth': 'analysis-rate mono; no pitch/time stretch',
        },
        'model': view.metadata,
        'frames': {
            'edges_samples': view.edges.tolist(),
            'query_samples': view.queries.tolist(),
            'range_convention': 'half-open',
        },
        'selection': {
            **dataclasses.asdict(config),
            'voice_budget': min(view.heads, config.max_active_heads)
            * config.max_head_blocks
            if config.tracking_policy == 'per_head_support'
            else config.max_layer_blocks,
            'version': 'per-head-support/5'
            if config.tracking_policy == 'per_head_support'
            else 'current-support/3'
            if config.tracking_policy == 'current_support'
            else 'average-overlap/2',
            'score': 'sum(max(attention - block.threshold, 0) * dt)',
            'time_unit': 'source seconds',
            'qualification': 'duration-weighted block average above threshold',
            'continuation': 'rank current support; bounded overlap bonus; no historical bounds'
            if config.tracking_policy in ('current_support', 'per_head_support')
            else 'retain overlapping voice; union playback bounds',
            'guard_samples': guard,
        },
        'heads': heads,
        'near_monitor': {
            'playback': monitor,
            'relative_gain': config.near_gain,
            'policy': 'single current-time bed if any head qualifies nearby',
            'counts_toward_remote_limit': False,
        },
        'rendering': {'width': 1920, 'height': 1080, 'fps': 30},
    }
    trace['audio'] = (
        {
            'mode': 'schedule_only',
            'stems': [],
            'mix': None,
            'head_gain': 0,
            'boundary_treatment': 'deferred to recursive mix',
        }
        if compact_audio
        else playback.render_audio(trace, audio, output)
    )
    if config.tracking_policy == 'per_head_support':
        trace['selection'][
            'score'
        ] = 'integrated positive excess over entry threshold, capped to five-second equivalent for ranking'
        trace['selection'][
            'continuation'
        ] = 'current exit-qualified support; extend by elapsed query time; no historical union'
    trace['elapsed_seconds'] = time.monotonic() - started
    io.write_json(trace_path, trace)
    print(
        f"Saved {trace_path} ({trace['elapsed_seconds']:.1f} s)",
        flush=True,
    )
    return trace
