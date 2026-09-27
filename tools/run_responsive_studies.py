"""Render and audit three-layer studies using current attention support."""

import argparse
import pathlib
import subprocess
from typing import Any

import numpy as np
import soundfile
from scipy import signal

from tapeheads import deep
from tapeheads import io
from tapeheads import playback_audit
from tapeheads import render
from tapeheads import selection
from tapeheads import validation


def cadence_peaks(path: pathlib.Path) -> dict[str, float]:
    """Measure narrow spectral peaks; these are diagnostics, not hum labels."""
    audio, sample_rate = soundfile.read(path, dtype='float32', always_2d=True)
    frequencies, power = signal.welch(
        audio.mean(axis=1),
        sample_rate,
        nperseg=min(len(audio), sample_rate * 4),
    )
    result = {}
    for frequency in (25, 30, 50):
        target = np.abs(frequencies - frequency) <= 0.25
        neighbors = (np.abs(frequencies - frequency) >= 2) & (
            np.abs(frequencies - frequency) <= 5
        )
        result[f'{frequency}_hz_prominence_db'] = float(
            10
            * np.log10(
                max(float(power[target].max()), 1e-20)
                / max(float(np.median(power[neighbors])), 1e-20)
            )
        )
    return result


def audit_layers(path: pathlib.Path) -> list[dict[str, Any]]:
    """Record stale support and playback lengths for each native layer."""
    trace = io.read_json(path)
    sample_rate = trace['source']['sample_rate_hz']
    result = []
    for layer in trace['layers']:
        native = io.read_json(path.parent / layer['native_trace'])
        runs = [
            segment for head in native['heads'] for segment in head['playback']
        ]
        lengths = np.array(
            [
                (segment['output_end'] - segment['output_start']) / sample_rate
                for segment in runs
            ]
        )
        support = playback_audit.support_statistics(native)
        result.append(
            {
                'layer': layer['model']['layer_index'] + 1,
                **support,
                'runs': len(runs),
                'median_run_seconds': float(np.median(lengths))
                if len(lengths)
                else 0,
                'maximum_run_seconds': float(lengths.max())
                if len(lengths)
                else 0,
                'runs_over_15_seconds': int(np.sum(lengths > 15)),
                'runs_at_most_50_ms': int(np.sum(lengths <= 0.05)),
                'near_vertical_run_fraction': sum(
                    abs(segment['source_start'] - segment['output_start'])
                    / trace['source']['samples']
                    * 1560
                    < 8
                    for segment in runs
                )
                / max(1, len(runs)),
            }
        )
    return result


def main() -> None:
    """Run the named recordings and save independent validation reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', choices=('muq', 'wav2vec2'), required=True)
    parser.add_argument(
        '--only',
        nargs='+',
        default=[
            'espresso',
            'bohemian_rhapsody',
            'dodge_image',
            'chirp_sequence',
        ],
    )
    arguments = parser.parse_args()
    config = selection.SelectionConfig(
        entry_alpha=1.5,
        exit_alpha=1.5,
        tracking_policy='current_support',
        switch_ratio=1.05,
        dwell_frames=1,
        max_layer_blocks=3,
        average_window_seconds=0.12,
        minimum_block_seconds=0.12,
    )
    report_path = io.ROOT / f'data/responsive-{arguments.encoder}-report.json'
    report = (
        io.read_json(report_path)
        if report_path.exists()
        else {
            'generator': 'python -m tools.run_responsive_studies',
            'encoder': arguments.encoder,
            'studies': {},
        }
    )
    for source in arguments.only:
        directory = (
            io.ROOT / f'outputs/{source}_{arguments.encoder}_responsive3'
        )
        path = directory / 'trace.json'
        print(f'Processing {source} / {arguments.encoder}', flush=True)
        if not path.exists():
            input_path = io.ROOT / f'data/audio/{source}/source.flac'
            if source == 'chirp_sequence':
                input_path = input_path.with_name('mono_48000.wav')
            deep.analyze(
                input_path,
                directory,
                encoder=arguments.encoder,
                layers=3,
                voices=3,
                config=config,
                single_pass=True,
                compress=True,
            )
        trace = io.read_json(path)
        if 'video' not in trace['rendering']:
            render.render(path)
        checks = validation.validate(path)
        layers = audit_layers(path)
        assert all(
            layer['unsupported_samples'] == 0
            and layer['unselected_samples'] == 0
            for layer in layers
        )
        subprocess.run(
            [
                io.executable('ffmpeg'),
                '-v',
                'error',
                '-xerror',
                '-i',
                str(directory / 'tapeheads.mp4'),
                '-f',
                'null',
                '-',
            ],
            check=True,
        )
        spectra = {
            'source': cadence_peaks(directory / trace['source']['path']),
            'new_depth_3': cadence_peaks(directory / 'depth_03.wav'),
        }
        previous = io.ROOT / f'outputs/{source}_{arguments.encoder}_collection8'
        if previous.exists():
            spectra['previous_depth_3'] = cadence_peaks(
                previous / 'depth_03.wav'
            )
            spectra['previous_depth_8'] = cadence_peaks(
                previous / 'depth_08.wav'
            )
        report['studies'][source] = {
            'trace': str(path.relative_to(io.ROOT)),
            'validation': checks,
            'layers': layers,
            'spectral_diagnostics': spectra,
            'video_decode': 'passed',
        }
        io.write_json(report_path, report)
        print(f'Validated {source} / {arguments.encoder}', flush=True)


if __name__ == '__main__':
    main()
