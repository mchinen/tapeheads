"""Shared types for selection and sample-based playback schedules.

All intervals are half-open. Source and output positions use the analysis
sample rate. Validation commands check versioned JSON documents against the
schemas; their metadata remains extensible through JsonObject.
"""

import os
from typing import Any
from typing import NotRequired
from typing import TypeAlias
from typing import TypedDict

JsonObject: TypeAlias = dict[str, Any]
PathLike: TypeAlias = str | os.PathLike[str]


class SourceSpan(TypedDict):
    """An interval on the input audio timeline."""

    source_start: int
    source_end: int


class AttentionBlock(SourceSpan):
    """A contiguous key interval with its attention statistics."""

    key_start: int
    key_end: int
    volume: float
    mass: float
    peak: float
    mean_attention: NotRequired[float]
    threshold: NotRequired[float]
    original_rank: NotRequired[int | None]


class SelectedVoice(AttentionBlock):
    """A persistent voice and its accumulated source playback interval."""

    head: int
    voice_id: int
    reason: str
    playback_start: int
    playback_end: int


class PlaybackSegment(SourceSpan):
    """A source run mapped at unit speed to the output timeline.

    Fades are assigned after scheduling. Brackets and voice identifiers are
    present only in trace formats that record them.
    """

    output_start: int
    output_end: int
    fade_in_samples: NotRequired[int]
    fade_out_samples: NotRequired[int]
    bracket_start: NotRequired[int]
    bracket_end: NotRequired[int]
    voice_id: NotRequired[int]
