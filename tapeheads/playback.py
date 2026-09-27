"""Convert selected spans into an auditable, sample-accurate tape schedule."""

import pathlib

import numpy as np
import soundfile

from tapeheads import trace_types


class TapeHead:
    """Schedule continuous source playback, with explicit seeks and loops."""

    def __init__(self, guard_samples: int) -> None:
        self.guard_samples = guard_samples
        self.cursor: int | None = None
        self.previous_bounds: tuple[int, int] | None = None
        self.segments: list[trace_types.PlaybackSegment] = []

    def append(
        self,
        output_start: int,
        output_end: int,
        selected: trace_types.SourceSpan | None,
    ) -> None:
        """Schedule a query interval while enforcing the guard at every sample."""
        if selected is None:
            self.cursor = None
            self.previous_bounds = None
            return
        source_start, source_end = (
            selected['source_start'],
            selected['source_end'],
        )
        output_cursor = output_start
        adjacent = self.previous_bounds and (
            source_start < self.previous_bounds[1]
            and self.previous_bounds[0] < source_end
        )
        if (
            not adjacent
            or self.cursor is None
            or not source_start <= self.cursor < source_end
        ):
            self.cursor = source_start
        while output_cursor < output_end:
            if self.cursor >= source_end:
                self.cursor = source_start
            if abs(self.cursor - output_cursor) <= self.guard_samples:
                # A future region may have approached the moving query cursor.
                past_end = min(source_end, output_cursor - self.guard_samples)
                future_start = max(
                    source_start, output_cursor + self.guard_samples + 1
                )
                if source_start < past_end:
                    self.cursor = source_start
                elif future_start < source_end:
                    self.cursor = future_start
                else:
                    self.cursor = None
                    break
            length = min(output_end - output_cursor, source_end - self.cursor)
            segment: trace_types.PlaybackSegment = {
                'output_start': output_cursor,
                'output_end': output_cursor + length,
                'source_start': self.cursor,
                'source_end': self.cursor + length,
                'bracket_start': source_start,
                'bracket_end': source_end,
            }
            if self.segments and (
                self.segments[-1]['output_end'] == output_cursor
                and self.segments[-1]['source_end'] == self.cursor
            ):
                self.segments[-1]['output_end'] += length
                self.segments[-1]['source_end'] += length
            else:
                self.segments.append(segment)
            output_cursor += length
            self.cursor += length
        self.previous_bounds = (source_start, source_end)


def finalize_segments(
    segments: list[trace_types.PlaybackSegment],
    sample_rate: int,
    guard_samples: int,
) -> list[trace_types.PlaybackSegment]:
    """Assign short fades and validate every source-to-output offset.

    Discontinuous spans use nonoverlapping boundary fades in this prototype.
    They cannot leak current-time audio through a second crossfade branch.
    """
    for segment in segments:
        length = segment['output_end'] - segment['output_start']
        if length != segment['source_end'] - segment['source_start']:
            raise AssertionError('Playback speed must remain exactly 1x')
        if (
            abs(segment['source_start'] - segment['output_start'])
            <= guard_samples
        ):
            raise AssertionError('Playback violates the temporal guard')
        fade = min(round(0.005 * sample_rate), length // 2)
        segment['fade_in_samples'] = fade
        segment['fade_out_samples'] = fade
    return segments


def render_audio(
    trace: trace_types.JsonObject,
    source: np.ndarray,
    directory: trace_types.PathLike,
) -> trace_types.JsonObject:
    """Render all stems and an equal-power stereo mix from saved segments."""
    directory = pathlib.Path(directory)
    sample_rate = trace['source']['sample_rate_hz']
    stems = []
    count = len(trace['heads'])
    mix = np.zeros((len(source), 2), dtype=np.float32)
    voice_limit = trace['selection'].get('max_layer_blocks', count)
    near_gain = trace.get('near_monitor', {}).get('relative_gain', 0)
    gain = 0.8 / max(1, voice_limit + near_gain)
    for head in trace['heads']:
        stem = np.zeros(len(source), dtype=np.float32)
        for segment in head['playback']:
            samples = source[
                segment['source_start'] : segment['source_end']
            ].copy()
            fade_in = segment['fade_in_samples']
            fade_out = segment['fade_out_samples']
            if fade_in:
                samples[:fade_in] *= np.linspace(0, 1, fade_in)
            if fade_out:
                samples[-fade_out:] *= np.linspace(1, 0, fade_out)
            stem[segment['output_start'] : segment['output_end']] += samples
        path = directory / f"head_{head['index']:02d}.wav"
        soundfile.write(path, stem, sample_rate, subtype='FLOAT')
        pan = head['index'] / max(1, count - 1)
        mix[:, 0] += gain * np.cos(pan * np.pi / 2) * stem
        mix[:, 1] += gain * np.sin(pan * np.pi / 2) * stem
        stems.append(path.name)
    if 'near_monitor' in trace:
        nearby = np.zeros(len(source), dtype=np.float32)
        for segment in trace['near_monitor']['playback']:
            start, end = segment['output_start'], segment['output_end']
            samples = source[start:end].copy()
            fade_in = segment['fade_in_samples']
            fade_out = segment['fade_out_samples']
            if fade_in:
                samples[:fade_in] *= np.linspace(0, 1, fade_in)
            if fade_out:
                samples[-fade_out:] *= np.linspace(1, 0, fade_out)
            nearby[start:end] += samples * near_gain
        soundfile.write(
            directory / 'nearby.wav', nearby, sample_rate, subtype='FLOAT'
        )
        mix += (gain / np.sqrt(2) * nearby)[:, None]
    soundfile.write(directory / 'mix.wav', mix, sample_rate, subtype='FLOAT')
    return {
        'stems': stems,
        'mix': 'mix.wav',
        'head_gain': gain,
        'remote_voice_limit': voice_limit,
        'near_relative_gain': near_gain,
        'near_stem': 'nearby.wav' if 'near_monitor' in trace else None,
        'pan': 'head index, equal-power, left to right',
        'normalization': 'none',
        'boundary_treatment': '5 ms fade-out/fade-in, no overlap',
        'peak': float(np.max(np.abs(mix))),
        'rms': float(np.sqrt(np.mean(mix**2))),
    }
