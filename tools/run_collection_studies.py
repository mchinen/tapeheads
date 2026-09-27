"""Run eight-layer music/nature comparisons and retain resumable audit results."""

import argparse
import fcntl
import pathlib
import subprocess
from typing import Any

import numpy as np

from tapeheads import deep
from tapeheads import io
from tapeheads import render
from tapeheads import validation


def diagnostics(
    trace: dict[str, Any], root: pathlib.Path
) -> list[dict[str, Any]]:
    """Summarize silence, short runs, head diversity, and broad selections."""
    results = []
    sample_rate = trace['source']['sample_rate_hz']
    duration = trace['source']['duration_seconds']
    for layer in trace['layers']:
        native = io.read_json(root / layer['native_trace'])
        active = [
            sum(
                len(head['selections'][index]['selected_blocks'])
                for head in native['heads']
            )
            for index in range(len(native['frames']['query_samples']))
        ]
        runs = [
            segment for head in native['heads'] for segment in head['playback']
        ]
        lengths = [
            (segment['output_end'] - segment['output_start']) / sample_rate
            for segment in runs
        ]
        heads_used = [
            head['index'] for head in native['heads'] if head['playback']
        ]
        spans = [
            (block['playback_end'] - block['playback_start']) / sample_rate
            for head in native['heads']
            for item in head['selections']
            for block in item['selected_blocks']
        ]
        flags = []
        silent = sum(value == 0 for value in active) / len(active)
        short = sum(value <= 0.05 for value in lengths) / max(1, len(lengths))
        median_span = float(np.median(spans)) if spans else 0
        if silent > 0.5:
            flags.append('remote voices absent for over half the queries')
        if short > 0.2:
            flags.append('over 20 percent of native runs are at most 50 ms')
        if median_span > duration / 4:
            flags.append(
                'median accumulated span exceeds one quarter of the input'
            )
        results.append(
            {
                'layer': layer['model']['layer_index'] + 1,
                'heads_used': heads_used,
                'silent_query_fraction': silent,
                'runs': len(lengths),
                'short_run_fraction': short,
                'median_run_seconds': float(np.median(lengths))
                if lengths
                else 0,
                'median_playback_span_seconds': median_span,
                'flags': flags,
            }
        )
    return results


def main() -> None:
    """Run complete recordings with all selected encoders' native context."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--encoder', choices=('wav2vec2', 'hubert', 'muq'), required=True
    )
    parser.add_argument('--only', nargs='+')
    arguments = parser.parse_args()
    proof = io.read_json(
        io.ROOT / f'data/single-pass-{arguments.encoder}-validation.json'
    )
    assert max(proof['maximum_errors']) < proof['absolute_tolerance']
    songs = arguments.only or [
        'espresso',
        'fake_plastic_trees',
        'bohemian_rhapsody',
        'nature_montage',
    ]
    report_path = io.ROOT / f'data/collection-{arguments.encoder}-report.json'
    report = (
        io.read_json(report_path)
        if report_path.exists()
        else {
            'generator': 'python -m tools.run_collection_studies',
            'encoder': arguments.encoder,
            'inference_check': proof,
            'studies': {},
        }
    )
    for song in songs:
        output = io.ROOT / f'outputs/{song}_{arguments.encoder}_collection8'
        trace_path = output / 'trace.json'
        print(f'Processing {song} / {arguments.encoder}', flush=True)
        if not trace_path.exists():
            source_path = io.ROOT / f'data/audio/{song}/source.flac'
            if song == 'chirp_sequence':
                source_path = (
                    io.ROOT / 'data/audio/chirp_sequence/mono_48000.wav'
                )
            deep.analyze(
                source_path,
                output,
                encoder=arguments.encoder,
                layers=8,
                voices=6,
                single_pass=True,
                compress=True,
            )
        trace = io.read_json(trace_path)
        if 'video' not in trace['rendering']:
            render.render(trace_path)
        checks = validation.validate(trace_path)
        trace = io.read_json(trace_path)
        findings = diagnostics(trace, output)
        subprocess.run(
            [
                io.executable('ffmpeg'),
                '-v',
                'error',
                '-xerror',
                '-threads',
                '2',
                '-i',
                str(output / 'tapeheads.mp4'),
                '-f',
                'null',
                '-',
            ],
            check=True,
        )
        result = {
            'validation': checks,
            'diagnostics': findings,
            'video_decode': 'passed',
            'trace': str(trace_path.relative_to(io.ROOT)),
        }
        # Independent song jobs share one encoder report. Merge under a lock
        # so one completed song cannot overwrite another job's audit result.
        with report_path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if report_path.exists():
                report = io.read_json(report_path)
            report['studies'][song] = result
            io.write_json(report_path, report)
        print(f'Validated {song} / {arguments.encoder}', flush=True)
    print('Collection complete', flush=True)


if __name__ == '__main__':
    main()
