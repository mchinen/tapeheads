"""Select sustained attention spans, excluding the query neighborhood."""

import dataclasses

import numpy as np

from tapeheads import trace_types


@dataclasses.dataclass(frozen=True)
class SelectionConfig:
    """Selection thresholds and continuity rules with explicit units."""

    entry_alpha: float = 2.0
    exit_alpha: float = 1.25
    guard_seconds: float = 0.1
    dwell_frames: int = 3
    switch_ratio: float = 1.2
    shortlist_size: int = 3
    max_layer_blocks: int = 3
    average_window_seconds: float = 0.2
    minimum_block_seconds: float = 0.12
    near_gain: float = 0.15
    tracking_policy: str = 'overlap_union'
    minimum_overlap_fraction: float = 0.5
    max_active_heads: int = 3
    max_head_blocks: int = 3
    max_initial_block_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.tracking_policy not in (
            'overlap_union',
            'current_support',
            'per_head_support',
        ):
            raise ValueError('Unknown tracking policy')
        if not 0 < self.minimum_overlap_fraction <= 1:
            raise ValueError('Overlap fraction must be in (0, 1]')
        if (
            self.max_active_heads < 1
            or self.max_head_blocks < 1
            or self.max_initial_block_seconds <= 0
        ):
            raise ValueError(
                'Head count and initial block duration must be positive'
            )
        if self.max_layer_blocks < 1:
            raise ValueError('Layer block count must be positive')
        if self.average_window_seconds < 0 or self.minimum_block_seconds < 0:
            raise ValueError('Block durations must be nonnegative')
        if not 0 <= self.near_gain < 1:
            raise ValueError('Near gain must be in [0, 1)')
        if not 0 < self.exit_alpha <= self.entry_alpha:
            raise ValueError('Require 0 < exit_alpha <= entry_alpha')
        if self.guard_seconds < 0 or self.dwell_frames < 1:
            raise ValueError(
                'Guard must be nonnegative; dwell must be positive'
            )
        if self.switch_ratio < 1 or self.shortlist_size < 1:
            raise ValueError(
                'Switch ratio and shortlist size must be at least 1'
            )


def overlaps(
    left: trace_types.AttentionBlock | trace_types.SelectedVoice,
    right: trace_types.AttentionBlock | trace_types.SelectedVoice,
) -> bool:
    """Return whether two half-open key intervals overlap."""
    return left['key_start'] < right['key_end'] and (
        right['key_start'] < left['key_end']
    )


def blocks(
    row: np.ndarray,
    eligible: np.ndarray,
    threshold: float,
    edges: np.ndarray,
    sample_rate: int,
) -> list[trace_types.AttentionBlock]:
    """Rank consecutive eligible keys by integrated excess attention."""
    mask = eligible & (row > threshold)
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    if not len(starts):
        return []
    durations = np.diff(edges) / sample_rate
    volumes = np.add.reduceat(
        np.where(mask, (row - threshold) * durations, 0), starts
    )
    masses = np.add.reduceat(np.where(mask, row, 0), starts)
    peaks = np.maximum.reduceat(np.where(mask, row, 0), starts)
    order = np.lexsort((starts, -peaks, -volumes))
    return [
        {
            'key_start': int(starts[index]),
            'key_end': int(ends[index]),
            'source_start': int(edges[starts[index]]),
            'source_end': int(edges[ends[index]]),
            'volume': float(volumes[index]),
            'mass': float(masses[index]),
            'peak': float(peaks[index]),
        }
        for index in order
    ]


class HeadSelector:
    """Track a head across query rows without allowing unsafe span holds."""

    def __init__(
        self, config: SelectionConfig, edges: np.ndarray, sample_rate: int
    ) -> None:
        self.config = config
        self.edges = np.asarray(edges)
        self.sample_rate = sample_rate
        self.current: trace_types.AttentionBlock | None = None
        self.pending: trace_types.AttentionBlock | None = None
        self.pending_count = 0

    def select(
        self,
        row: np.ndarray,
        query_sample: int,
        valid: np.ndarray | None = None,
    ) -> trace_types.JsonObject:
        """Return the current block and ranked candidates for one query.

        Args:
            row: Original, unmasked post-softmax attention probabilities.
            query_sample: Query anchor in analysis audio samples.
            valid: Audio keys visible to this query before temporal exclusion.

        Returns:
            Selection diagnostics including the supported committed interval.
        """
        row = np.asarray(row, dtype=np.float64)
        if not np.isfinite(row).all() or (row < 0).any():
            raise ValueError(
                'Attention probabilities must be finite and positive'
            )
        if valid is None:
            valid = np.ones(len(row), dtype=bool)
        valid_key_count = int(np.sum(valid))
        if valid_key_count == 0:
            self.current = None
            return {
                'selected': None,
                'candidates': [],
                'candidate_count': 0,
                'excluded_mass': 0.0,
                'normalized_entropy': 0.0,
                'reason': 'no_context_keys',
            }
        threshold = self.config.entry_alpha / valid_key_count
        guard_samples = round(self.config.guard_seconds * self.sample_rate)
        safe = (
            (self.edges[1:] <= query_sample - guard_samples)
            | (self.edges[:-1] > query_sample + guard_samples)
        ) & valid
        candidates = blocks(row, safe, threshold, self.edges, self.sample_rate)
        original = blocks(row, valid, threshold, self.edges, self.sample_rate)
        original_ranks = np.zeros(len(row), dtype=np.int64)
        for rank, block in enumerate(original, 1):
            original_ranks[block['key_start'] : block['key_end']] = rank
        for candidate in candidates:
            candidate['original_rank'] = int(
                original_ranks[candidate['key_start']]
            )
        supported = blocks(
            row,
            safe,
            self.config.exit_alpha / valid_key_count,
            self.edges,
            self.sample_rate,
        )
        continuing = None
        if self.current is not None:
            matches = [
                block for block in supported if overlaps(block, self.current)
            ]
            if matches:
                continuing = max(
                    matches,
                    key=lambda block: min(
                        block['key_end'], self.current['key_end']
                    )
                    - max(block['key_start'], self.current['key_start']),
                )
                start, end = continuing['key_start'], continuing['key_end']
                durations = (
                    np.diff(self.edges[start : end + 1]) / self.sample_rate
                )
                continuing['volume'] = float(
                    np.sum(
                        np.maximum(row[start:end] - threshold, 0) * durations
                    )
                )
                continuing['original_rank'] = None
        winner = candidates[0] if candidates else None
        reason = 'hold'
        if continuing is None:
            self.current = winner
            self.pending_count = 0
            self.pending = None
            reason = 'acquire' if winner else 'no_eligible_block'
        elif winner and overlaps(winner, continuing):
            self.current = winner
            self.pending_count = 0
            self.pending = None
        elif winner and winner['volume'] > (
            continuing['volume'] * self.config.switch_ratio
        ):
            if self.pending and overlaps(winner, self.pending):
                self.pending_count += 1
            else:
                self.pending_count = 1
            self.pending = winner
            self.current = continuing
            if self.pending_count >= self.config.dwell_frames:
                self.current = winner
                self.pending = None
                self.pending_count = 0
                reason = 'switch'
        else:
            self.current = continuing
            self.pending = None
            self.pending_count = 0
        positive = row[row > 0]
        entropy = -float(np.sum(positive * np.log(positive)))
        return {
            'selected': dict(self.current) if self.current else None,
            'candidates': candidates[: self.config.shortlist_size],
            'candidate_count': len(candidates),
            'excluded_mass': float(row[~safe].sum()),
            'normalized_entropy': entropy / np.log(valid_key_count)
            if valid_key_count > 1
            else 0,
            'reason': reason,
        }
