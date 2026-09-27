#!/usr/bin/env python3
"""Generate four identical chirps separated by diagnostic audio segments.

Run from the project root with:
    .venv/bin/python -m tools.generate_chirp_sequence
"""

import pathlib
import subprocess
from typing import Any

import numpy as np
import soundfile

from tapeheads import io

SAMPLE_RATE = 48000
SEED = 20260919
DTMF_ROWS = (697, 770, 852, 941)
DTMF_COLUMNS = (1209, 1336, 1477, 1633)
DTMF_KEYS = ('123A', '456B', '789C', '*0#D')


def fade(samples: np.ndarray, seconds: float = 0.01) -> np.ndarray:
    """Apply symmetric linear edge fades without changing segment length."""
    samples = samples.copy()
    count = min(round(seconds * SAMPLE_RATE), len(samples) // 2)
    if count:
        samples[:count] *= np.linspace(0, 1, count)
        samples[-count:] *= np.linspace(1, 0, count)
    return samples


def build_sequence() -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return the waveform and a sample-aligned, reproducible event list."""
    random_generator = np.random.default_rng(SEED)
    chunks = []
    events = []
    cursor = 0

    def append(kind, samples, **parameters):
        nonlocal cursor
        samples = np.asarray(samples, dtype=np.float32)
        events.append(
            {
                'kind': kind,
                'start_sample': cursor,
                'end_sample': cursor + len(samples),
                'start_seconds': cursor / SAMPLE_RATE,
                'end_seconds': (cursor + len(samples)) / SAMPLE_RATE,
                'parameters': parameters,
            }
        )
        chunks.append(samples)
        cursor += len(samples)

    time = np.arange(3 * SAMPLE_RATE) / SAMPLE_RATE
    chirp = fade(
        0.25 * np.sin(2 * np.pi * (40 * time + (7000 - 40) * time**2 / (2 * 3)))
    )
    append('silence', np.zeros(SAMPLE_RATE))
    for repetition in range(4):
        append(
            'chirp',
            chirp,
            repetition=repetition + 1,
            start_hz=40,
            end_hz=7000,
            sweep='linear',
            amplitude=0.25,
            fade_seconds=0.01,
        )
        if repetition == 3:
            break
        frequency = (220, 440, 880)[repetition]
        time = np.arange(round(1.5 * SAMPLE_RATE)) / SAMPLE_RATE
        append(
            'sine',
            fade(0.25 * np.sin(2 * np.pi * frequency * time)),
            frequency_hz=frequency,
            amplitude=0.25,
            fade_seconds=0.01,
        )
        append(
            'gaussian_noise',
            fade(random_generator.normal(0, 0.1, len(time))),
            mean=0,
            standard_deviation=0.1,
            seed=SEED,
            sequence_draw=repetition + 1,
            fade_seconds=0.01,
        )
        append('silence', np.zeros(SAMPLE_RATE))
        for symbol in ('123A', '456B', '789C')[repetition]:
            row = next(
                index for index, keys in enumerate(DTMF_KEYS) if symbol in keys
            )
            column = DTMF_KEYS[row].index(symbol)
            frequencies = (DTMF_ROWS[row], DTMF_COLUMNS[column])
            time = np.arange(round(0.35 * SAMPLE_RATE)) / SAMPLE_RATE
            tone = sum(
                0.125 * np.sin(2 * np.pi * hz * time) for hz in frequencies
            )
            append(
                'dtmf',
                fade(tone),
                symbol=symbol,
                frequencies_hz=list(frequencies),
                amplitude_per_tone=0.125,
                fade_seconds=0.01,
            )
            append('silence', np.zeros(round(0.15 * SAMPLE_RATE)))
    append('silence', np.zeros(SAMPLE_RATE))
    return np.concatenate(chunks), events


def main() -> None:
    """Write the master waveform, model-rate derivatives, and manifest."""
    directory = io.ROOT / 'data' / 'audio' / 'chirp_sequence'
    directory.mkdir(parents=True, exist_ok=True)
    source, events = build_sequence()
    master = directory / 'mono_48000.wav'
    soundfile.write(master, source, SAMPLE_RATE, subtype='FLOAT')
    ffmpeg = io.executable('ffmpeg')
    files = []
    for sample_rate in (48000, 16000, 24000):
        destination = directory / f'mono_{sample_rate}.wav'
        if sample_rate != SAMPLE_RATE:
            subprocess.run(
                [
                    ffmpeg,
                    '-v',
                    'error',
                    '-y',
                    '-i',
                    str(master),
                    '-ar',
                    str(sample_rate),
                    '-ac',
                    '1',
                    '-c:a',
                    'pcm_f32le',
                    str(destination),
                ],
                check=True,
            )
        info = soundfile.info(destination)
        files.append(
            {
                'path': str(destination.relative_to(io.ROOT)),
                'sha256': io.sha256(destination),
                'samples': info.frames,
                'sample_rate_hz': info.samplerate,
                'channels': info.channels,
            }
        )
    manifest = {
        'format': 'tapeheads-signals/1',
        'id': 'chirp_sequence',
        'title': 'Four chirps / sine / Gaussian noise / silence / DTMF',
        'generator': 'tools/generate_chirp_sequence.py',
        'generator_sha256': io.sha256(pathlib.Path(__file__)),
        'numpy_version': np.__version__,
        'seed': SEED,
        'sample_rate_hz': SAMPLE_RATE,
        'samples': len(source),
        'duration_seconds': len(source) / SAMPLE_RATE,
        'sample_format': 'float32 WAV',
        'peak': float(np.max(np.abs(source))),
        'rate_policy': '48 kHz master resampled by FFmpeg; no normalization',
        'ffmpeg_version': subprocess.check_output(
            [ffmpeg, '-version'], text=True
        ).splitlines()[0],
        'events': events,
        'files': files,
    }
    io.write_json(io.ROOT / 'data' / 'chirp_sequence.json', manifest)
    print(f'Wrote {len(source) / SAMPLE_RATE:.1f} s sequence to {directory}')


if __name__ == '__main__':
    main()
