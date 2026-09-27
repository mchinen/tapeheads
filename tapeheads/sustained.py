"""Select layer-wide voices using average attention and overlap continuity."""

import numpy as np

from tapeheads import selection
from tapeheads import trace_types


def average_blocks(
    row: np.ndarray,
    eligible: np.ndarray,
    threshold: float,
    edges: np.ndarray,
    sample_rate: int,
    window_seconds: float,
    minimum_seconds: float,
    prepared: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[trace_types.AttentionBlock]:
    """Find contiguous regions whose duration-weighted average is high.

    A box average proposes connected regions, allowing individual weak frames
    inside them. The original probabilities independently qualify each region.
    """
    if prepared is None:
        durations = np.diff(edges) / sample_rate
        window_frames = min(
            len(row), max(1, round(window_seconds / np.median(durations)))
        )
        kernel = np.ones(window_frames)
        weight = durations * eligible
        denominator = np.convolve(weight, kernel, mode='same')
        numerator = np.convolve(row * weight, kernel, mode='same')
        smoothed_attention = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(row),
            where=denominator > 0,
        )
    else:
        durations, smoothed_attention = prepared
    mask = eligible & (smoothed_attention > threshold)
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    if not len(starts):
        return []
    volumes = np.add.reduceat(
        np.where(mask, np.maximum(row - threshold, 0) * durations, 0), starts
    )
    integrals = np.add.reduceat(np.where(mask, row * durations, 0), starts)
    masses = np.add.reduceat(np.where(mask, row, 0), starts)
    peaks = np.maximum.reduceat(np.where(mask, row, 0), starts)
    result: list[trace_types.AttentionBlock] = []
    for index, (start, end) in enumerate(zip(starts, ends)):
        duration = (edges[end] - edges[start]) / sample_rate
        mean = integrals[index] / duration
        if duration < minimum_seconds or mean <= threshold:
            continue
        result.append(
            {
                'key_start': int(start),
                'key_end': int(end),
                'source_start': int(edges[start]),
                'source_end': int(edges[end]),
                'volume': float(volumes[index]),
                'mass': float(masses[index]),
                'peak': float(peaks[index]),
                'mean_attention': float(mean),
                'threshold': float(threshold),
            }
        )
    return sorted(
        result,
        key=lambda block: (
            -block['volume'],
            -block['mean_attention'],
            block['key_start'],
        ),
    )


class LayerSelector:
    """Keep overlapping voices within the configured layer-wide limit."""

    def __init__(
        self,
        config: selection.SelectionConfig,
        edges: np.ndarray,
        sample_rate: int,
        heads: int,
    ) -> None:
        self.config = config
        self.edges = np.asarray(edges)
        self.sample_rate = sample_rate
        self.heads = heads
        self.active: list[trace_types.SelectedVoice] = []
        self.next_voice_id = 0
        self.durations = np.diff(self.edges) / sample_rate
        window_frames = min(
            len(self.durations),
            max(
                1,
                round(
                    config.average_window_seconds / np.median(self.durations)
                ),
            ),
        )
        self.kernel = np.ones(window_frames)

    def select(
        self, rows: np.ndarray, query_sample: int, valid: np.ndarray
    ) -> tuple[list[trace_types.JsonObject], list[trace_types.SelectedVoice]]:
        """Return head diagnostics and persistent remote voice assignments."""
        valid_key_count = int(np.sum(valid))
        if (
            not valid_key_count
            or not np.isfinite(rows).all()
            or (rows < 0).any()
        ):
            raise ValueError('Expected finite probabilities and visible keys')
        guard_samples = round(self.config.guard_seconds * self.sample_rate)
        near = (
            valid
            & (self.edges[1:] > query_sample - guard_samples)
            & (self.edges[:-1] <= query_sample + guard_samples)
        )
        safe = valid & ~near
        durations = self.durations
        weight = durations * safe
        denominator = np.convolve(weight, self.kernel, mode='same')
        active_heads = {voice['head'] for voice in self.active}
        near_duration = float(durations[near].sum())
        diagnostics = []
        support: list[list[trace_types.AttentionBlock]] = []
        for head_index, row in enumerate(rows):
            numerator = np.convolve(row * weight, self.kernel, mode='same')
            smoothed_attention = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(row),
                where=denominator > 0,
            )
            prepared = (durations, smoothed_attention)
            candidates = average_blocks(
                row,
                safe,
                self.config.entry_alpha / valid_key_count,
                self.edges,
                self.sample_rate,
                self.config.average_window_seconds,
                self.config.minimum_block_seconds,
                prepared=prepared,
            )
            support.append(
                average_blocks(
                    row,
                    safe,
                    self.config.exit_alpha / valid_key_count,
                    self.edges,
                    self.sample_rate,
                    self.config.average_window_seconds,
                    self.config.minimum_block_seconds,
                    prepared=prepared,
                )
                if head_index in active_heads
                and self.config.tracking_policy == 'overlap_union'
                else []
            )
            near_mean = (
                float(np.sum(row[near] * durations[near])) / near_duration
                if near_duration
                else 0.0
            )
            diagnostics.append(
                {
                    'selected_blocks': [],
                    'selected': None,
                    'candidates': candidates[
                        : max(
                            self.config.shortlist_size,
                            self.config.max_layer_blocks,
                        )
                    ],
                    'candidate_count': len(candidates),
                    'near_attention': {
                        'mass': float(row[near].sum()),
                        'relative_mean': near_mean * valid_key_count,
                        'active': near_mean * valid_key_count
                        > self.config.entry_alpha,
                        'source_start': max(0, query_sample - guard_samples),
                        'source_end': min(
                            int(self.edges[-1]), query_sample + guard_samples
                        ),
                    },
                }
            )
        if self.config.tracking_policy == 'current_support':
            return self._select_current(diagnostics)
        retained: list[trace_types.SelectedVoice] = []
        used = set()
        for active in self.active:
            head = active['head']
            matches = [
                block
                for block in support[head]
                if selection.overlaps(active, block)
                and (head, block['key_start'], block['key_end']) not in used
            ]
            if not matches:
                continue
            block = max(
                matches,
                key=lambda item: (
                    min(item['key_end'], active['key_end'])
                    - max(item['key_start'], active['key_start']),
                    item['volume'],
                ),
            )
            used.add((head, block['key_start'], block['key_end']))
            retained.append(
                {
                    **block,
                    'head': head,
                    'voice_id': active['voice_id'],
                    'reason': 'overlap',
                    'playback_start': min(
                        active['playback_start'], block['source_start']
                    ),
                    'playback_end': max(
                        active['playback_end'], block['source_end']
                    ),
                }
            )
        available: list[tuple[int, trace_types.AttentionBlock]] = []
        for head, item in enumerate(diagnostics):
            for block in item['candidates']:
                if not any(
                    voice['head'] == head and selection.overlaps(block, voice)
                    for voice in retained
                ):
                    available.append((head, block))
        available.sort(
            key=lambda candidate: (
                -candidate[1]['volume'],
                candidate[0],
                candidate[1]['key_start'],
            )
        )
        for head, block in available:
            if len(retained) >= self.config.max_layer_blocks:
                break
            if any(
                voice['head'] == head and selection.overlaps(block, voice)
                for voice in retained
            ):
                continue
            retained.append(
                {
                    **block,
                    'voice_id': self.next_voice_id,
                    'head': head,
                    'reason': 'acquire',
                    'playback_start': block['source_start'],
                    'playback_end': block['source_end'],
                }
            )
            self.next_voice_id += 1
        self.active = retained
        for voice in retained:
            diagnostics[voice['head']]['selected_blocks'].append(dict(voice))
        for item in diagnostics:
            # Retain the primary-block field for older trace consumers.
            item['selected'] = next(iter(item['selected_blocks']), None)
        return diagnostics, retained

    def _select_current(
        self, diagnostics: list[trace_types.JsonObject]
    ) -> tuple[list[trace_types.JsonObject], list[trace_types.SelectedVoice]]:
        """Rank current blocks, preserving cursors only for matching support.

        A small score bonus prevents ties from changing voice identity. It
        cannot keep an unsupported block alive. Overlapping source regions
        from different heads share one slot rather than doubling the audio.
        """
        candidates = []
        overlap_limit = self.config.minimum_overlap_fraction
        for head_index, item in enumerate(diagnostics):
            for block in item['candidates']:
                matches = [
                    voice
                    for voice in self.active
                    if voice['head'] == head_index
                    and overlap_fraction(block, voice) >= overlap_limit
                ]
                previous = max(
                    matches,
                    key=lambda voice: overlap_fraction(block, voice),
                    default=None,
                )
                score = block['volume'] * (
                    self.config.switch_ratio if previous else 1
                )
                candidates.append((score, head_index, block, previous))
        candidates.sort(
            key=lambda item: (-item[0], item[1], item[2]['key_start'])
        )
        selected: list[trace_types.SelectedVoice] = []
        used_identifiers = set()
        for _, head_index, block, previous in candidates:
            if len(selected) >= self.config.max_layer_blocks:
                break
            if any(
                overlap_fraction(block, voice) >= overlap_limit
                for voice in selected
            ):
                continue
            if previous and previous['voice_id'] not in used_identifiers:
                identifier = previous['voice_id']
                reason = 'current_overlap'
            else:
                identifier = self.next_voice_id
                self.next_voice_id += 1
                reason = 'acquire'
            used_identifiers.add(identifier)
            selected.append(
                {
                    **block,
                    'head': head_index,
                    'voice_id': identifier,
                    'reason': reason,
                    'playback_start': block['source_start'],
                    'playback_end': block['source_end'],
                }
            )
        self.active = selected
        for voice in selected:
            diagnostics[voice['head']]['selected_blocks'].append(dict(voice))
        for item in diagnostics:
            item['selected'] = next(iter(item['selected_blocks']), None)
        return diagnostics, selected


def overlap_fraction(
    left: trace_types.AttentionBlock | trace_types.SelectedVoice,
    right: trace_types.AttentionBlock | trace_types.SelectedVoice,
) -> float:
    """Return intersection over union of two source intervals."""
    intersection = max(
        0,
        min(left['source_end'], right['source_end'])
        - max(left['source_start'], right['source_start']),
    )
    union = max(left['source_end'], right['source_end']) - min(
        left['source_start'], right['source_start']
    )
    return intersection / union
