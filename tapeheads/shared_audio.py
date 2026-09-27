"""Compose per-head voices through shared, unnormalized layer buses."""

import itertools

import numpy as np

from tapeheads import recursive
from tapeheads import trace_types


def render_depth(
    layers: list[trace_types.JsonObject], source: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Include every selected branch without expanding a recursive path tree.

    Identical local reads share mean gains. Ancestor gains remain scalar, so
    parent panning cannot silence opposite-side child heads. Only the final
    bus receives splice correction and loudness normalization.
    """
    previous = None
    previous_boundaries = np.array([], dtype=int)
    result = np.zeros((len(source), 2), dtype=np.float32)
    active_samples = np.zeros(len(source), dtype=bool)
    boundaries: set[int] = set()
    for layer in layers:
        result = np.zeros_like(result)
        active_samples = np.zeros(len(source), dtype=bool)
        boundaries = set()
        budget = (
            layer['selection']['voice_budget']
            + layer['near_monitor']['relative_gain']
        )
        reads = []
        for head, _, relative_gain, segments in recursive.tracks(layer):
            pan = head / max(1, len(layer['heads']) - 1) if head >= 0 else 0.5
            gains = (
                np.array(
                    [np.cos(pan * np.pi / 2), np.sin(pan * np.pi / 2), 1.0]
                )
                * relative_gain
                / budget
            )
            for segment in segments:
                offset = segment['source_start'] - segment['output_start']
                reads.append(
                    (
                        offset,
                        segment['output_start'],
                        segment['output_end'],
                        gains,
                    )
                )
        reads.sort(key=lambda read: read[0])
        for offset, group in itertools.groupby(reads, key=lambda read: read[0]):
            events = []
            for identifier, (_, start, end, gains) in enumerate(group):
                events.extend(
                    [
                        (start, 1, identifier, gains),
                        (end, -1, identifier, gains),
                    ]
                )
            active = {}
            previous_sample = 0
            gains = np.zeros(3)
            for sample, changes in itertools.groupby(
                sorted(
                    events, key=lambda event: (event[0], event[1], event[2])
                ),
                key=lambda event: event[0],
            ):
                if active and sample > previous_sample:
                    source_start, source_end = (
                        previous_sample + offset,
                        sample + offset,
                    )
                    direct = source[source_start:source_end, None] * gains[:2]
                    values = (
                        direct
                        if previous is None
                        else 0.35 * direct
                        + 0.65 * gains[2] * previous[source_start:source_end]
                    )
                    result[previous_sample:sample] += values.astype(np.float32)
                    active_samples[previous_sample:sample] = True
                    if previous is not None:
                        left, right = np.searchsorted(
                            previous_boundaries, [source_start, source_end]
                        )
                        boundaries.update(
                            (previous_boundaries[left:right] - offset).tolist()
                        )
                for _, action, identifier, value in changes:
                    if action == 1:
                        active[identifier] = value
                    else:
                        del active[identifier]
                updated = (
                    np.mean(list(active.values()), axis=0)
                    if active
                    else np.zeros(3)
                )
                if not np.allclose(updated, gains, atol=1e-12, rtol=0):
                    boundaries.add(sample)
                previous_sample, gains = sample, updated
        previous = result
        previous_boundaries = np.array(sorted(boundaries), dtype=int)
    correction = max(1, round(sample_rate * 0.003))
    for sample in sorted(boundaries):
        if sample >= len(source) or not active_samples[sample]:
            continue
        end = min(len(source), sample + correction)
        before = result[sample - 1].copy() if sample else np.zeros(2)
        difference = before - result[sample]
        decay = (1 + np.cos(np.linspace(0, np.pi, end - sample))) / 2
        result[sample:end] += (
            decay[:, None] * difference * active_samples[sample:end, None]
        ).astype(np.float32)
    starts = np.flatnonzero(
        np.diff(np.r_[False, active_samples].astype(np.int8)) == 1
    )
    stops = (
        np.flatnonzero(
            np.diff(np.r_[active_samples, False].astype(np.int8)) == -1
        )
        + 1
    )
    for start, end in zip(starts, stops, strict=True):
        fade = min(correction, (end - start) // 2)
        if fade:
            result[end - fade : end] *= np.linspace(1, 0, fade)[:, None]
    rms = float(np.sqrt(np.mean(result**2)))
    peak = float(np.abs(result).max())
    gain = (
        min(4.0, 0.1 / max(rms, 1e-12), 0.8 / max(peak, 1e-12)) if peak else 1.0
    )
    result *= gain
    return result, {
        'gain': gain,
        'peak': float(np.abs(result).max()),
        'rms': float(np.sqrt(np.mean(result**2))),
        'splice_corrections': len(boundaries),
    }
