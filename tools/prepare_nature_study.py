"""Prepare an attributed forest recording and a reproducible nature montage."""

import urllib.request

import numpy as np
import soundfile

from tapeheads import io

BIRD_URL = (
    'https://upload.wikimedia.org/wikipedia/commons/3/38/Birds_forest.ogg'
)
BIRD_PAGE = 'https://commons.wikimedia.org/wiki/File:Birds_forest.ogg'


def level(audio: np.ndarray, target: float = 0.08) -> tuple[np.ndarray, float]:
    """Apply a documented fixed RMS/peak adjustment without a limiter."""
    gain = min(
        target / max(float(np.sqrt(np.mean(audio**2))), 1e-10),
        0.6 / max(float(np.abs(audio).max()), 1e-10),
    )
    return audio * gain, gain


def main() -> None:
    """Download public-domain birds and combine known source excerpts."""
    sample_rate = 24000
    download = io.ROOT / 'data/downloads/forest_birds/birds_forest.ogg'
    download.parent.mkdir(parents=True, exist_ok=True)
    if not download.exists():
        request = urllib.request.Request(
            BIRD_URL, headers={'User-Agent': 'Tapeheads audio study'}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            download.write_bytes(response.read())
    directory = io.ROOT / 'data/audio/forest_birds'
    directory.mkdir(parents=True, exist_ok=True)
    birds = io.decode(download, sample_rate=sample_rate)
    soundfile.write(
        directory / 'source.flac', birds, sample_rate, subtype='PCM_24'
    )
    assets = {
        'forest_birds': directory / 'source.flac',
        'rain': io.ROOT / 'data/audio/rain/source.flac',
        'beach': io.ROOT / 'data/audio/beach/source.flac',
        'chirp_sequence': io.ROOT / 'data/audio/chirp_sequence/mono_48000.wav',
    }
    excerpts = [
        ('forest_birds', 0, len(birds) / sample_rate, 'Forest / birds'),
        ('rain', 0, 16, 'Rain'),
        ('beach', 0, 14, 'Beach / waves'),
        ('chirp_sequence', 8, 2, 'DTMF / beeps'),
        ('chirp_sequence', 10, 3, 'Synthetic chirp'),
    ]
    parts, sections = [], []
    cursor = 0
    for name, start, duration, label in excerpts:
        audio = io.decode(assets[name], start, duration, sample_rate)
        audio, gain = level(audio)
        fade = min(120, len(audio) // 2)
        audio[:fade] *= np.linspace(0, 1, fade)
        audio[-fade:] *= np.linspace(1, 0, fade)
        sections.append(
            {
                'start_seconds': cursor / sample_rate,
                'end_seconds': (cursor + len(audio)) / sample_rate,
                'label': label,
                'confidence': 'exact',
                'components': [
                    {
                        'id': name,
                        'input_start_seconds': start,
                        'gain': gain,
                        'repeated': False,
                    }
                ],
            }
        )
        parts.append(audio)
        cursor += len(audio)
    mixture = np.zeros(15 * sample_rate, dtype=np.float32)
    components = []
    for name, weight in [
        ('forest_birds', 0.45),
        ('rain', 0.25),
        ('beach', 0.2),
        ('chirp_sequence', 0.1),
    ]:
        audio = io.decode(assets[name], sample_rate=sample_rate)
        audio, gain = level(audio)
        repeated = len(audio) < len(mixture)
        mixture += np.resize(audio, len(mixture)) * weight
        components.append(
            {
                'id': name,
                'input_start_seconds': 0,
                'gain': gain * weight,
                'repeated': repeated,
            }
        )
    mixture[:120] *= np.linspace(0, 1, 120)
    mixture[-120:] *= np.linspace(1, 0, 120)
    sections.append(
        {
            'start_seconds': cursor / sample_rate,
            'end_seconds': (cursor + len(mixture)) / sample_rate,
            'label': 'Combined / nature + signals',
            'confidence': 'exact',
            'components': components,
        }
    )
    parts.append(mixture)
    output = io.ROOT / 'data/audio/nature_montage'
    output.mkdir(parents=True, exist_ok=True)
    soundfile.write(
        output / 'source.flac',
        np.concatenate(parts),
        sample_rate,
        subtype='PCM_24',
    )
    io.write_json(
        io.ROOT / 'data/nature_montage.json',
        {
            'generator': 'python -m tools.prepare_nature_study',
            'sample_rate_hz': sample_rate,
            'sections': sections,
            'fade_samples': 120,
            'normalization': 'RMS 0.08; peak cap 0.6; mixture weights recorded',
            'assets': {
                name: {
                    'path': str(path.relative_to(io.ROOT)),
                    'sha256': io.sha256(path),
                }
                for name, path in assets.items()
            },
            'forest_credit': {
                'author': 'Barracuda1983',
                'location': 'Fontainebleau, France',
                'license': 'public domain dedication',
                'page': BIRD_PAGE,
                'download_url': BIRD_URL,
                'download_sha256': io.sha256(download),
            },
            'output_sha256': io.sha256(output / 'source.flac'),
        },
    )
    print(
        f'Saved {output}/source.flac: {sum(len(part) for part in parts) / sample_rate:.2f} s'
    )


if __name__ == '__main__':
    main()
