#!/usr/bin/env python3
"""Generate deterministic signals with standard-library synthesis and FFmpeg."""

import array
import hashlib
import json
import math
import pathlib
import random
import subprocess
import sys
import wave
from typing import Any

from tools import prepare_audio

ROOT = pathlib.Path(__file__).resolve().parents[1]
DURATION_SECONDS = 30
SAMPLE_RATES = (48000, 16000, 24000)
SEED = 20260918


def generate_sample(
    kind: str, index: int, sample_rate: int, generator: random.Random
) -> float:
    """Return one sample with documented amplitude and frequency parameters."""
    time = index / sample_rate
    if kind == 'sine':
        return 0.25 * math.sin(2 * math.pi * 440 * time)
    if kind == 'chirp':
        slope = (7000 - 40) / DURATION_SECONDS
        return 0.25 * math.sin(2 * math.pi * (40 * time + slope * time**2 / 2))
    if kind == 'gaussian_noise':
        return generator.gauss(0, 0.1)
    if kind == 'gaussian_pulse':
        return 0.5 * math.exp(-0.5 * ((time - 15) / 0.01) ** 2)
    raise ValueError(f'Unknown signal: {kind}')


def write_signal(
    kind: str, sample_rate: int, path: pathlib.Path
) -> dict[str, Any]:
    """Write mono PCM16 WAV and return statistics for validation."""
    generator = random.Random(SEED)
    samples = array.array('h')
    total = DURATION_SECONDS * sample_rate
    clipped = 0
    squared_sum = 0
    peak = 0
    for index in range(total):
        value = generate_sample(kind, index, sample_rate, generator)
        if kind in ('sine', 'chirp'):
            fade_samples = round(0.01 * sample_rate)
            value *= min(
                1, index / fade_samples, (total - 1 - index) / fade_samples
            )
        clipped += abs(value) > 1
        value = max(-1, min(1, value))
        quantized = round(value * 32767)
        samples.append(quantized)
        squared_sum += (quantized / 32768) ** 2
        peak = max(peak, abs(quantized) / 32768)
    if sys.byteorder != 'little':
        samples.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())
    return {
        'path': str(path.relative_to(ROOT)),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'sample_rate_hz': sample_rate,
        'samples': total,
        'channels': 1,
        'duration_seconds': DURATION_SECONDS,
        'peak': peak,
        'rms': math.sqrt(squared_sum / total),
        'clipped_samples': clipped,
    }


def main() -> None:
    """Create fixtures and save their complete generation parameters."""
    parameters = {
        'sine': {'frequency_hz': 440, 'amplitude': 0.25, 'fade_seconds': 0.01},
        'chirp': {
            'start_hz': 40,
            'end_hz': 7000,
            'sweep': 'linear',
            'amplitude': 0.25,
            'fade_seconds': 0.01,
        },
        'gaussian_noise': {'mean': 0, 'standard_deviation': 0.1, 'seed': SEED},
        'gaussian_pulse': {
            'center_seconds': 15,
            'sigma_seconds': 0.01,
            'amplitude': 0.5,
        },
    }
    entries = []
    for kind, configuration in parameters.items():
        directory = ROOT / 'data' / 'audio' / kind
        source = directory / 'mono_48000.wav'
        files = [write_signal(kind, 48000, source)]
        ffmpeg = prepare_audio.find_tool('ffmpeg')
        for sample_rate in SAMPLE_RATES[1:]:
            destination = directory / f'mono_{sample_rate}.wav'
            subprocess.run(
                [
                    ffmpeg,
                    '-v',
                    'error',
                    '-y',
                    '-i',
                    str(source),
                    '-ar',
                    str(sample_rate),
                    '-ac',
                    '1',
                    '-c:a',
                    'pcm_s16le',
                    str(destination),
                ],
                check=True,
            )
            with wave.open(str(destination), 'rb') as audio:
                count = audio.getnframes()
            files.append(
                {
                    'path': str(destination.relative_to(ROOT)),
                    'sha256': hashlib.sha256(
                        destination.read_bytes()
                    ).hexdigest(),
                    'sample_rate_hz': sample_rate,
                    'samples': count,
                    'channels': 1,
                    'duration_seconds': count / sample_rate,
                }
            )
        entries.append(
            {
                'id': kind,
                'kind': 'synthetic',
                'status': 'ready',
                'parameters': configuration,
                'files': files,
            }
        )
        print(kind, 'ready', flush=True)
    manifest = {
        'format': 'tapeheads-signals/1',
        'generator': 'tools/generate_signals.py',
        'python_version': sys.version,
        'sample_format': 'PCM16',
        'rate_policy': 'Generate at 48 kHz; derive 16 and 24 kHz copies with '
        'the FFmpeg default resampler.',
        'ffmpeg_version': prepare_audio.run([ffmpeg, '-version']).splitlines()[
            0
        ],
        'entries': entries,
    }
    path = ROOT / 'data' / 'signals.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
