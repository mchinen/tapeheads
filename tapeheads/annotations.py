"""Create exact synthetic labels and explicitly estimated musical sections."""

import json

import numpy as np

from tapeheads import io
from tapeheads import trace_types


def display_label(label: str) -> str:
    """Normalize generated legacy labels without changing motifs or units."""
    parts = label.split(' / ')
    replacements = {
        'BIRDS': 'Birds',
        'WAVES': 'Waves',
        'BEEPS': 'Beeps',
        'NATURE + SIGNALS': 'Nature + signals',
    }
    for index, part in enumerate(parts):
        if index == 0 and part != 'DTMF' and part.isupper():
            parts[index] = part.capitalize()
        elif part in replacements:
            parts[index] = replacements[part]
    return ' / '.join(parts)


def create(
    source: np.ndarray, sample_rate: int, source_id: str
) -> trace_types.JsonObject:
    """Return source-clock labels, never inferred lyrics or claimed form truth."""
    if source_id == 'nature_montage':
        path = io.ROOT / 'data/nature_montage.json'
        manifest = io.read_json(path)
        sections = []
        for section in manifest['sections']:
            start = round(section['start_seconds'] * sample_rate)
            if start >= len(source):
                break
            sections.append(
                {
                    'start': start,
                    'end': min(
                        len(source), round(section['end_seconds'] * sample_rate)
                    ),
                    'label': section['label'],
                    'confidence': 'exact',
                }
            )
        # Resampling can round the final sample up while timestamp conversion
        # rounds down. Cover that final sample without moving internal edges.
        if sections and abs(sections[-1]['end'] - len(source)) <= 1:
            sections[-1]['end'] = len(source)
        return {
            'method': 'generator manifest',
            'source_sha256': io.sha256(path),
            'sections': sections,
            'lyrics': [],
        }
    if source_id == 'chirp_sequence':
        manifest = json.loads(
            (io.ROOT / 'data/chirp_sequence.json').read_text()
        )
        events = []
        for event in manifest['events']:
            if round(event['start_seconds'] * sample_rate) >= len(source):
                break
            kind, parameters = event['kind'], event['parameters']
            label = kind.replace('_', ' ').capitalize()
            if kind == 'chirp':
                label = f"Chirp {parameters['repetition']} / 40–7000 Hz"
            elif kind == 'sine':
                label = f"Sine / {parameters['frequency_hz']} Hz"
            elif kind == 'dtmf':
                label = f"DTMF / {parameters['symbol']}"
            events.append(
                {
                    'start': round(event['start_seconds'] * sample_rate),
                    'end': min(
                        len(source), round(event['end_seconds'] * sample_rate)
                    ),
                    'label': label,
                    'confidence': 'exact',
                }
            )
        return {
            'method': 'generator manifest',
            'source_sha256': io.sha256(io.ROOT / 'data/chirp_sequence.json'),
            'sections': events,
            'lyrics': [],
        }
    # Compare half-second spectral profiles; peaks suggest structural changes,
    # not semantic verse/chorus labels. Keep uncertainty visible in the film.
    hop_samples = round(sample_rate / 2)
    profiles = []
    for start in range(0, len(source) - hop_samples + 1, hop_samples):
        frame = source[start : start + hop_samples]
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        bands = np.array(
            [np.mean(part) for part in np.array_split(spectrum, 32)]
        )
        bands = np.log1p(bands)
        profiles.append(bands / max(np.linalg.norm(bands), 1e-10))
    profiles = np.array(profiles)
    novelty = np.zeros(len(profiles))
    for index in range(4, len(profiles) - 4):
        novelty[index] = np.linalg.norm(
            profiles[index - 4 : index].mean(0)
            - profiles[index : index + 4].mean(0)
        )
    choices = []
    for index in np.argsort(-novelty):
        sample = int(index * hop_samples)
        if sample < 6 * sample_rate or sample > len(source) - 6 * sample_rate:
            continue
        if all(abs(sample - other) >= 12 * sample_rate for other in choices):
            choices.append(sample)
        if len(choices) >= max(1, round(len(source) / sample_rate / 18)):
            break
    edges = [0, *sorted(choices), len(source)]
    sections = []
    templates = []
    for index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        profile = profiles[
            start
            // hop_samples : max(start // hop_samples + 1, end // hop_samples)
        ].mean(0)
        similarities = [
            float(
                np.dot(profile, prior)
                / max(np.linalg.norm(profile) * np.linalg.norm(prior), 1e-10)
            )
            for prior in templates
        ]
        match = int(np.argmax(similarities)) if similarities else -1
        repeated = match >= 0 and similarities[match] > 0.985
        if repeated:
            motif = chr(65 + match)
        else:
            motif = chr(65 + len(templates))
            templates.append(profile)
        role = (
            'Opening' if index == 0 else ('Return' if repeated else 'Contrast')
        )
        sections.append(
            {
                'start': start,
                'end': end,
                'label': f'{role} / {motif}',
                'confidence': 'estimated',
                'motif': motif,
            }
        )
    return {
        'method': 'spectral novelty and coarse similarity; not verse/chorus recognition',
        'configuration': {
            'hop_seconds': 0.5,
            'context_seconds': 2,
            'minimum_boundary_spacing_seconds': 12,
            'repeat_cosine': 0.985,
        },
        'sections': sections,
        'lyrics': [],
    }


def study_sections(
    source: np.ndarray, sample_rate: int, source_id: str
) -> trace_types.JsonObject:
    """Use generator labels or explicitly approximate external section times."""
    if source_id in ('chirp_sequence', 'nature_montage'):
        return create(source, sample_rate, source_id)
    path = io.ROOT / 'data/section_labels.json'
    references = io.read_json(path)
    reference = references['recordings'].get(source_id)
    if reference:
        points = [
            (round(seconds * sample_rate), label)
            for seconds, label in reference['sections']
            if round(seconds * sample_rate) < len(source)
        ]
        sections = [
            {
                'start': start,
                'end': points[index + 1][0]
                if index + 1 < len(points)
                else len(source),
                'label': label + ' ~',
                'confidence': 'reference timing; audio alignment unverified',
            }
            for index, (start, label) in enumerate(points)
        ]
        return {
            'method': 'reference sections; approximate',
            'reference_url': reference['url'],
            'source_sha256': io.sha256(path),
            'sections': sections,
            'lyrics': [],
        }
    manifest = io.read_json(io.ROOT / 'data/sources.json')
    item = next(
        (entry for entry in manifest['entries'] if entry['id'] == source_id), {}
    )
    label = (
        ''
        if item.get('kind') == 'music'
        else source_id.replace('_', ' ').capitalize()
    )
    return {
        'method': 'source description; no inferred musical form',
        'sections': [
            {
                'start': 0,
                'end': len(source),
                'label': label,
                'confidence': 'unlabeled'
                if not label
                else 'source description',
            }
        ],
        'lyrics': [],
    }
