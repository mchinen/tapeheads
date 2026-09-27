# Dark recursive studies

These links refer to local artifacts under `outputs/`, which are excluded
from Git. Reproduce the studies to open them locally. Recorded results and
batch statuses are snapshots of the original runs, not live job status.

Each study follows selected second-layer frames through every selected
first-layer playback voice active at those timestamps. Connections can cross
head indices. Native attention addresses mixed frame representations; these
head branches are the sonification mapping, not a separately measured
head-to-head attention matrix.

| Encoder | Input | Video | JSON trace | Stereo mix |
| --- | --- | --- | --- | --- |
| wav2vec2 | Four chirps | [Watch](../outputs/chirp_sequence_wav2vec2_dark/tapeheads.mp4) | [Trace](../outputs/chirp_sequence_wav2vec2_dark/trace.json) | [Listen](../outputs/chirp_sequence_wav2vec2_dark/mix.wav) |
| wav2vec2 | Full Blue Suede Shoes | [Watch](../outputs/blue_suede_shoes_full_wav2vec2_dark/tapeheads.mp4) | [Trace](../outputs/blue_suede_shoes_full_wav2vec2_dark/trace.json) | [Listen](../outputs/blue_suede_shoes_full_wav2vec2_dark/mix.wav) |
| MuQ | Four chirps | [Watch](../outputs/chirp_sequence_muq_dark/tapeheads.mp4) | [Trace](../outputs/chirp_sequence_muq_dark/trace.json) | [Listen](../outputs/chirp_sequence_muq_dark/mix.wav) |
| MuQ | Full Blue Suede Shoes | [Watch](../outputs/blue_suede_shoes_full_muq_dark/tapeheads.mp4) | [Trace](../outputs/blue_suede_shoes_full_muq_dark/trace.json) | [Listen](../outputs/blue_suede_shoes_full_muq_dark/mix.wav) |
| HuBERT | Four chirps | [Watch](../outputs/chirp_sequence_hubert_dark/tapeheads.mp4) | [Trace](../outputs/chirp_sequence_hubert_dark/trace.json) | [Listen](../outputs/chirp_sequence_hubert_dark/mix.wav) |
| HuBERT | Full Blue Suede Shoes | [Watch](../outputs/blue_suede_shoes_full_hubert_dark/tapeheads.mp4) | [Trace](../outputs/blue_suede_shoes_full_hubert_dark/trace.json) | [Listen](../outputs/blue_suede_shoes_full_hubert_dark/mix.wav) |

The same folders contain independent `direct.wav` and `recursive.wav` stems,
plus both native layer traces. The entry threshold is 1.5 times uniform
attention; continuation uses 1.1 times uniform. The averaging window is 350 ms
and minimum qualifying block duration is 300 ms. A qualifying block can still
end early when attention leaves it. All full recordings use full-context keys.

Brackets mark accumulated playback bounds on the receiving plane; the curve
endpoint marks the actual playback cursor inside that span. Quiet near-time
paths remain in the audio at reduced gain and are omitted visually. The text
instructions and amber category are removed.

Reproduce these outputs with `tools/render_dark_studies.py`, and inspect
`data/dark-path-audit.json` for bracket and continuity measurements.

## Playback measurements

The following are medians of uninterrupted native playback runs. Recursive
paths can be shorter because a parent run intersects a child run. These
measurements do not count every intersection as a new source loop.

| Encoder | Chirp layer 1 | Chirp layer 2 | Elvis layer 1 | Elvis layer 2 |
| --- | --- | --- | --- | --- |
| wav2vec2 | 0.90 s | 1.30 s | 0.34 s | 0.48 s |
| MuQ | 1.68 s | 1.44 s | 2.56 s | 0.36 s |
| HuBERT | 0.58 s | 1.84 s | 0.95 s | 0.84 s |

For Elvis/wav2vec2, layer-one runs at or below 50 ms fell from 101 in the previous
recursive study to 25 here. The median rose from 0.26 s to 0.34 s. The second
layer changed less, from 0.46 s to 0.48 s median. Lowering thresholds reduces
some short retargeting, but does not eliminate every short run or loop.
