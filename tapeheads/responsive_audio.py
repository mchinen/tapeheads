"""Render shallow recursion without repeated per-layer amplitude gates."""

import dataclasses
import itertools

import numpy as np

from tapeheads import recursive
from tapeheads import trace_types


@dataclasses.dataclass(frozen=True)
class SourceRoute:
    """A unit-speed source read after composing all intermediate clocks."""

    output_start: int
    output_end: int
    source_offset: int
    left_gain: float
    right_gain: float


def compose_routes(
    layers: list[trace_types.JsonObject],
    samples: int,
    direct_weight: float = 0.35,
) -> list[SourceRoute]:
    """Include direct audio at each attended frame and all lower-layer reads."""
    bucket_samples = max(1, (samples + 1023) // 1024)
    tracks = []
    lookups = []
    for layer in layers:
        entries = []
        buckets: dict[int, list[int]] = {}
        for head, _, relative_gain, segments in recursive.tracks(layer):
            for segment in segments:
                identifier = len(entries)
                entries.append((head, relative_gain, segment))
                for bucket in range(
                    segment['output_start'] // bucket_samples,
                    (segment['output_end'] - 1) // bucket_samples + 1,
                ):
                    buckets.setdefault(bucket, []).append(identifier)
        tracks.append(entries)
        lookups.append(buckets)
    routes = []
    guard = layers[-1]['selection']['guard_samples']
    near_gain = layers[-1]['near_monitor']['relative_gain']

    def visit(
        layer_index: int,
        output_start: int,
        output_end: int,
        clock_start: int,
        gain: float,
        hops: int,
    ) -> None:
        layer = layers[layer_index]
        clock_end = clock_start + output_end - output_start
        budget = (
            layer['selection']['max_layer_blocks']
            + layer['near_monitor']['relative_gain']
        )
        # Buckets limit each recursive read to overlapping playback intervals.
        identifiers = set()
        for bucket in range(
            clock_start // bucket_samples,
            (clock_end - 1) // bucket_samples + 1,
        ):
            identifiers.update(lookups[layer_index].get(bucket, []))
        for identifier in sorted(identifiers):
            head, relative_gain, segment = tracks[layer_index][identifier]
            pan = head / max(1, len(layer['heads']) - 1) if head >= 0 else 0.5
            stereo = np.array(
                [np.cos(pan * np.pi / 2), np.sin(pan * np.pi / 2)]
            )
            # Pan each audible branch without silencing opposite-side children.
            voice_gain = gain * relative_gain / budget
            start = max(clock_start, segment['output_start'])
            end = min(clock_end, segment['output_end'])
            if start >= end:
                continue
            start_output = output_start + start - clock_start
            end_output = start_output + end - start
            start_source = (
                segment['source_start'] + start - segment['output_start']
            )
            offset = start_source - start_output
            direct_gain = (
                voice_gain * stereo * (direct_weight if layer_index else 1)
            )
            if hops > 1 and abs(offset) <= guard:
                direct_gain = direct_gain * near_gain
            routes.append(
                SourceRoute(
                    start_output,
                    end_output,
                    offset,
                    float(direct_gain[0]),
                    float(direct_gain[1]),
                )
            )
            if layer_index:
                visit(
                    layer_index - 1,
                    start_output,
                    end_output,
                    start_source,
                    voice_gain * (1 - direct_weight),
                    hops + 1,
                )

    visit(len(layers) - 1, 0, samples, 0, 1.0, 1)
    return routes


def mix_routes(
    routes: list[SourceRoute],
    source: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Average coincident routes, then correct discontinuities once per mix.

    The correction decays over 3 ms and never reads an old source interval.
    Unlike multiplying fades at every recursive hop, it does not force the
    signal to zero at the model's query cadence. Corrections stop in silence.
    """
    result = np.zeros((len(source), 2), dtype=np.float32)
    active_samples = np.zeros(len(source), dtype=bool)
    boundaries = set()
    coincident_samples = 0
    ordered = sorted(routes, key=lambda route: route.source_offset)
    for offset, group in itertools.groupby(
        ordered, key=lambda route: route.source_offset
    ):
        events = []
        for identifier, route in enumerate(group):
            events.append((route.output_start, 1, identifier, route))
            events.append((route.output_end, -1, identifier, route))
        active: dict[int, SourceRoute] = {}
        previous_sample = 0
        previous_gain = np.zeros(2)
        for sample, changes in itertools.groupby(
            sorted(events), key=lambda event: event[0]
        ):
            if active and sample > previous_sample:
                result[previous_sample:sample] += (
                    source[previous_sample + offset : sample + offset, None]
                    * previous_gain[None, :]
                ).astype(np.float32)
                active_samples[previous_sample:sample] = True
                if len(active) > 1:
                    coincident_samples += sample - previous_sample
            for _, action, identifier, route in changes:
                if action == 1:
                    active[identifier] = route
                else:
                    del active[identifier]
            current_gain = np.array(
                [
                    sum(route.left_gain for route in active.values()),
                    sum(route.right_gain for route in active.values()),
                ]
            ) / max(1, len(active))
            if not np.allclose(current_gain, previous_gain, rtol=0, atol=1e-12):
                boundaries.add(sample)
            previous_sample = sample
            previous_gain = current_gain

    correction_samples = max(1, round(0.003 * sample_rate))
    correction_count = 0
    for sample in sorted(boundaries):
        if sample >= len(source) or not active_samples[sample]:
            continue
        end = min(len(source), sample + correction_samples)
        previous = result[sample - 1].copy() if sample else np.zeros(2)
        difference = previous - result[sample]
        decay = (1 + np.cos(np.linspace(0, np.pi, end - sample))) / 2
        # Restrict the correction to samples that still have a selected route.
        result[sample:end] += (
            decay[:, None]
            * difference[None, :]
            * active_samples[sample:end, None]
        ).astype(np.float32)
        correction_count += 1
    # Fade before an actual stop, not at every source seek or layer boundary.
    stops = (
        np.flatnonzero(
            np.diff(np.r_[active_samples, False].astype(np.int8)) == -1
        )
        + 1
    )
    starts = np.flatnonzero(
        np.diff(np.r_[False, active_samples].astype(np.int8)) == 1
    )
    for start, end in zip(starts, stops, strict=True):
        fade = min(correction_samples, (end - start) // 2)
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
        'rms': float(np.sqrt(np.mean(result**2))),
        'peak': float(np.abs(result).max()),
        'routes': len(routes),
        'coincident_source_samples': coincident_samples,
        'splice_corrections': correction_count,
        'splice_correction_samples': correction_samples,
    }


def render_depth(
    layers: list[trace_types.JsonObject],
    source: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Compose and render every route through the requested depth."""
    return mix_routes(compose_routes(layers, len(source)), source, sample_rate)
