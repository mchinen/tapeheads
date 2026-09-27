"""Select independently sustained, currently supported blocks for each head."""

import dataclasses

import numpy as np

from tapeheads import selection
from tapeheads import sustained
from tapeheads import trace_types


class PerHeadSelector(sustained.LayerSelector):
    """Use per-head hysteresis without retaining historical playback bounds."""

    def __init__(self, config, edges, sample_rate, heads):
        super().__init__(
            dataclasses.replace(config, tracking_policy='current_support'),
            edges,
            sample_rate,
            heads,
        )
        self.previous_query = 0
        self.query_sample = 0
        self.rows = np.empty((0, 0))
        self.safe = np.zeros(len(edges) - 1, dtype=bool)
        self.valid_count = 0

    def select(self, rows, query_sample, valid):
        self.rows = rows
        self.query_sample = query_sample
        self.valid_count = int(valid.sum())
        guard = round(self.config.guard_seconds * self.sample_rate)
        self.safe = valid & (
            (self.edges[1:] <= query_sample - guard)
            | (self.edges[:-1] > query_sample + guard)
        )
        result = super().select(rows, query_sample, valid)
        self.previous_query = query_sample
        return result

    def _crop(
        self,
        row: np.ndarray,
        block: trace_types.AttentionBlock,
        threshold: float,
        previous: trace_types.SelectedVoice | None = None,
    ) -> trace_types.AttentionBlock | None:
        start, end = block['key_start'], block['key_end']
        limit = round(self.config.max_initial_block_seconds * self.sample_rate)
        if previous is None and self.edges[end] - self.edges[start] > limit:
            width = max(1, int(limit / np.median(np.diff(self.edges))))
            excess = np.maximum(row[start:end] - threshold, 0)
            integral = np.r_[0, np.cumsum(excess * self.durations[start:end])]
            offset = int(np.argmax(integral[width:] - integral[:-width]))
            start += offset
            end = min(end, start + width)
        elif previous is not None:
            start = max(start, previous['key_start'])
            extension = self.query_sample - self.previous_query
            end = min(
                end,
                int(
                    np.searchsorted(
                        self.edges,
                        previous['source_end'] + extension,
                        side='right',
                    )
                )
                - 1,
            )
        if start >= end:
            return None
        duration = float(self.durations[start:end].sum())
        values = row[start:end]
        mean = float(np.sum(values * self.durations[start:end]) / duration)
        if duration < self.config.minimum_block_seconds or mean <= threshold:
            return None
        entry = self.config.entry_alpha / self.valid_count
        return {
            'key_start': int(start),
            'key_end': int(end),
            'source_start': int(self.edges[start]),
            'source_end': int(self.edges[end]),
            'volume': float(
                np.sum(
                    np.maximum(values - entry, 0) * self.durations[start:end]
                )
            ),
            'mass': float(values.sum()),
            'peak': float(values.max()),
            'mean_attention': mean,
            'threshold': threshold,
        }

    def _select_current(self, diagnostics):
        selected: list[trace_types.SelectedVoice] = []
        entry = self.config.entry_alpha / self.valid_count
        continuation = self.config.exit_alpha / self.valid_count
        for head, item in enumerate(diagnostics):
            row = self.rows[head]
            choices: list[tuple[trace_types.AttentionBlock, int | None]] = []
            previous = [voice for voice in self.active if voice['head'] == head]
            support = (
                sustained.average_blocks(
                    row,
                    self.safe,
                    continuation,
                    self.edges,
                    self.sample_rate,
                    self.config.average_window_seconds,
                    self.config.minimum_block_seconds,
                )
                if previous
                else []
            )
            for voice in previous:
                for region in support:
                    if not selection.overlaps(region, voice):
                        continue
                    block = self._crop(row, region, continuation, voice)
                    if block is not None:
                        choices.append((block, voice['voice_id']))
            for region in item['candidates']:
                if any(
                    selection.overlaps(region, block) for block, _ in choices
                ):
                    continue
                block = self._crop(row, region, entry)
                if block is not None:
                    choices.append((block, None))

            def score(candidate):
                block, identifier = candidate
                duration = (
                    block['source_end'] - block['source_start']
                ) / self.sample_rate
                bounded_volume = block['volume'] * min(
                    1, self.config.max_initial_block_seconds / duration
                )
                return bounded_volume * (
                    self.config.switch_ratio if identifier is not None else 1
                )

            choices.sort(
                key=lambda candidate: (
                    -score(candidate),
                    candidate[0]['key_start'],
                )
            )
            head_voices: list[trace_types.SelectedVoice] = []
            for block, identifier in choices:
                if len(head_voices) >= self.config.max_head_blocks:
                    break
                if any(
                    selection.overlaps(block, voice) for voice in head_voices
                ):
                    continue
                if identifier is not None and any(
                    voice['voice_id'] == identifier for voice in head_voices
                ):
                    continue
                reason = 'supported_extension'
                if identifier is None:
                    identifier = self.next_voice_id
                    self.next_voice_id += 1
                    reason = 'acquire'
                voice: trace_types.SelectedVoice = {
                    **block,
                    'head': head,
                    'voice_id': identifier,
                    'reason': reason,
                    'playback_start': block['source_start'],
                    'playback_end': block['source_end'],
                }
                head_voices.append(voice)
            item['selected_blocks'] = head_voices
            item['selected'] = next(iter(head_voices), None)
            selected.extend(head_voices)
        # Rank heads by their strongest region so extra regions do not win slots.
        previous_heads = {voice['head'] for voice in self.active}
        head_scores = {}
        for voice in selected:
            duration = (
                voice['source_end'] - voice['source_start']
            ) / self.sample_rate
            value = voice['volume'] * min(
                1, self.config.max_initial_block_seconds / duration
            )
            if voice['head'] in previous_heads:
                value *= self.config.switch_ratio
            head_scores[voice['head']] = max(
                head_scores.get(voice['head'], 0), value
            )
        retained_heads = set(
            sorted(head_scores, key=lambda head: (-head_scores[head], head))[
                : self.config.max_active_heads
            ]
        )
        selected = [
            voice for voice in selected if voice['head'] in retained_heads
        ]
        for head, item in enumerate(diagnostics):
            if head not in retained_heads:
                item['selected_blocks'] = []
                item['selected'] = None
        self.active = selected
        return diagnostics, selected
