# Prototype usage

This page records the earlier prototype experiments. New CLI runs default to
current-support selection; see [Current-support studies](responsive-studies.md).
Add `--tracking-policy overlap_union` to an analysis command to reproduce the
older selection policy and recursive renderer described here.

The first-layer pipeline now has four encoder choices, one selection engine,
and one trace-driven audio/video renderer. This is a command-line prototype;
an interactive encoder picker and playback controls are future work.

## Encoder choices

| CLI name | Checkpoint | First-layer heads | Context and mapping |
| --- | --- | ---: | --- |
| `wav2vec2` | `facebook/wav2vec2-base` | 12 | Full input, nominal 20 ms bins |
| `hubert` | `facebook/hubert-base-ls960` | 12 | Full input, nominal 20 ms bins |
| `muq` | `OpenMuQ/MuQ-large-msd-iter` | 16 | Full input, nominal 40 ms bins |
| `clap` | `laion/clap-htsat-unfused` | 4 | Independent 10-second contexts, local Swin windows |

Revisions are pinned in `tapeheads/encoders.py` and written into each trace.
The CPU implementation uses float32. MuQ requires `tools/setup_muq.sh`.
Wav2vec and HuBERT adapters stop before first-layer attention, then evaluate
query blocks against every key. MuQ additionally preserves its native rotary
position transformation. These paths do not compute the deeper layers.

CLAP loads the full checkpoint and selects its audio branch, rejecting missing
or mismatched weights. A pre-hook captures inputs to the first block's actual
self-attention, not the last block of its first stage. Native probabilities
include the learned relative position bias. The adapter sums attention across
key-frequency patches and averages query-frequency patches, retaining all four
native heads and their local context masks. This is an explicit time projection
of patch attention, not unmodified temporal attention.

Short CLAP contexts are zero-padded rather than repeated. Padded source keys
are excluded without renormalizing the surviving probabilities. There is no
random long-audio crop: the whole supplied input is traversed in 10-second
contexts. Its time mapping uses patch centers after the model's spectrogram
interpolation, with midpoint boundaries. This is a playback-bin convention;
STFT windows and convolutional patches have wider receptive fields.

## Commands

Compare another encoder on the same input:

```sh
.venv/bin/python -m tapeheads analyze data/audio/dodge_image/source.flac \
  --encoder clap --duration 12 --output outputs/my_clap --video
```

Tune the threshold and guard:

```sh
.venv/bin/python -m tapeheads analyze data/audio/dodge_image/source.flac \
  --encoder clap --duration 12 --alpha 1.1 --guard-ms 100 \
  --output outputs/my_clap_sensitive --video
```

`--alpha` is the entry threshold relative to uniform attention over visible
context keys. The default is 2. CLAP's local windows can become mostly silent
at that threshold after the 100 ms guard is applied; lower thresholds are
explicit experiments, not automatic changes to the data. Every trace records
its threshold. Scores integrate attention above threshold over source duration.
The score controls selection, not audio gain.

For full-input analysis, omit duration:

```sh
.venv/bin/python -m tapeheads analyze data/audio/dodge_image/source.flac \
  --encoder wav2vec2 --output outputs/my_full_piece
.venv/bin/python -m tapeheads render outputs/my_full_piece/trace.json \
  --width 1280 --height 720 --fps 30
```

`--start` and `--duration` define the encoder input context itself. A 12-second
excerpt does not retain attention to the rest of its parent song. This
limitation is recorded as `source.context_mode = "excerpt"`.

`--query-block-size` defaults to 128; `--threads` defaults to 4. Increasing the
input length still increases compute quadratically for temporal encoders, and
the convolutional frontend retains full-input activations. Query blocking
reduces attention-matrix memory, not all memory or total compute.

## Outputs

Each output directory contains:

- `trace.json`: model provenance, time mapping, ranked blocks, selection
  diagnostics, exact playback segments, and render configuration.
- `source.wav`: the actual mono, model-rate waveform used for playback.
- `head_NN.wav`: one float32 mono stem per native head, indexed from zero.
- `mix.wav`: stereo mix with fixed gain and equal-power head positioning.
- `tapeheads.mp4`: synchronized video when requested.
- `preview.png`: a still chosen near four seconds among the most active rows.

The prototype auditions model-rate mono audio. It does not yet map the
schedule back to a higher-rate stereo original. Rendering uses fixed head
gains and no loudness normalization. Discontinuous playback has 5 ms boundary
fades, limited to half a segment. Overlapping crossfades are not implemented;
there can be short level dips at splices. Neither faint audio nor silence is
replaced with an invented attention event.

A head plays forward at 1x and continues across query rows while its region
remains supported. Reaching a span boundary creates an explicit loop. Source
and output sample offsets are checked before every scheduled run. Hysteresis
never overrides the 100 ms guard. Unsupported spans are dropped immediately
in this version rather than held through a silence grace period.

The video shows query time as a vertical line, an arm to the selected bracket,
and a separate playback dot. The faint band is the exclusion neighborhood.
Time-frequency projections and excerpt context are described in the trace.
Audio and video can be rerendered from the saved schedule without inference.
The current CLI exposes video rerendering; `playback.render_audio` is the shared
audio-rendering function.

## Validation

```sh
.venv/bin/python -m tapeheads verify outputs/my_example/trace.json
.venv/bin/python -m unittest discover -s tests -v
```

The verifier checks the [JSON Schema](../schemas/trace.schema.json), source
hash, frame boundaries, source exclusion, playback range/speed/guard, stem
lengths, finite audio, and video/audio duration agreement within one frame.
Unit tests cover diagonal exclusion, selection by volume rather than peak,
dwell, continuity, silence, looping, and CLAP frequency projection.

For short temporal-encoder fixtures, compare blocked attention to native
attention before export:

```sh
.venv/bin/python -m tapeheads analyze data/audio/chirp/mono_16000.wav \
  --encoder wav2vec2 --duration 4 --verify-attention \
  --output outputs/my_reference_check
```

The comparison is limited to five seconds to bound dense reference memory.
Wav2vec and HuBERT compare against the complete model's first-layer output;
MuQ compares against its native first attention module. CLAP always invokes
the native module and checks projected mass conservation; the flag does not
add a separate CLAP full-model comparison. Unit tests verify its projection
on known identity and uniform native attention maps.

## Adding an encoder

Keep model-specific work in an adapter. Register an `EncoderSpec` and dispatch
it from `create_view`. Return an `AttentionView` containing source-bin edges,
query anchors, native head count, provenance, and an iterator over query rows.
The iterator supplies a visible-key mask so local windows and padding are
preserved before temporal exclusion. Selection and rendering must not import
model-specific modules.

An adapter must distinguish native temporal attention from a time projection
of patch attention; preserve positional terms and masks; pin its checkpoint;
and reject unsupported layouts. Add reference comparisons and time-mapping
tests before listing a new adapter as supported. Deeper-layer inspection,
recursive composition, arbitrary checkpoint overrides, and fused CLAP variants
are not implemented in this prototype.

## Prepared examples and measured results

The local `outputs/` directory contains these reviewable artifacts:

| Directory | Input | Encoder | Duration |
| --- | --- | --- | ---: |
| `dodge_full_wav2vec2` | Complete Charles Dodge recording | wav2vec 2.0 | 122.64 s |
| `blue_suede_shoes_full_wav2vec2` | Complete Elvis Presley official audio | wav2vec 2.0 | 122.55 s |
| `dodge_wav2vec2` | Opening excerpt | wav2vec 2.0 | 12 s |
| `dodge_muq` | Opening excerpt | MuQ | 12 s |
| `dodge_clap` | Opening excerpt, alpha 2 | CLAP | 12 s |
| `dodge_clap_sensitive` | Opening excerpt, alpha 1.1 | CLAP | 12 s |
| `chirp_wav2vec2` | Chirp fixture | wav2vec 2.0 | 4 s |
| `chirp_hubert` | Chirp fixture | HuBERT | 4 s |
| `chirp_muq` | Chirp fixture | MuQ | 4 s |

The complete piece has 12 heads, 6,131 query frames, and 19,977 playback
segments. Extraction, selection, stem rendering, and trace preparation took
127.35 seconds on this machine with four CPU threads. Video rendering is a
separate step; the full-piece video is 1280×720 at 30 fps. Its verbose JSON is
about 123 MiB because it retains per-query diagnostics and candidate blocks.
This is measured prototype output size, not the compact event format target.

Four-second wav2vec, HuBERT, and MuQ fixtures each matched their native
attention reference with maximum absolute error 0 in the tested environment.
This does not prove equivalence on every signal or dependency version. The
[validation report](../data/prototype-validation.json) records the exercised
traces and checks. Full-song peak RAM and MuQ full-song performance remain
unmeasured.

## Sustained playback, version 2

New analysis uses `average-overlap/2` and exports `tapeheads/2`. This supersedes
version 1's single winner per head and strict per-frame qualification.

- `--max-layer-blocks 3`: total remote voices across the layer, with silent heads
  allowed and multiple blocks per head supported.
- `--average-window-ms 200` and `--min-block-ms 120`: propose sustained regions
  and qualify their original duration-weighted average.
- `--near-gain 0.15`: one quiet current-time monitor when any head's nearby
  average qualifies. The existing `--guard-ms 100` is a symmetric radius.

Overlapping support retains a voice identifier and accumulates playback bounds,
so small changes in a bracket do not restart the source cursor. The green
bracket shows current attention support; the pale underline shows accumulated
playback bounds. Dots show actual source playback. Amber brackets mark heads
with qualifying current-neighborhood attention. The quiet monitor is separate
from the three remote slots. Remote playback still obeys the temporal guard.

Legacy dwell and challenger-ratio configuration fields do not affect version 2.
Persistent voice identifiers and explicit source/output mappings feed the
recursive second-layer implementation described below.

Updated examples are in `outputs/chirp_sequence_wav2vec2_v2` and
`outputs/blue_suede_shoes_full_wav2vec2_v2`. Each directory contains the JSON
trace, source, head stems, `nearby.wav`, `mix.wav`, and `tapeheads.mp4`.

### Measured continuity

The default version 2 settings increased median uninterrupted playback from
40 ms to 360 ms on the 32-second chirp sequence, and from 40 ms to 260 ms on the
full 122.55-second Elvis recording. The longest runs were 9 seconds and
5 seconds, respectively. These statistics measure source-to-output continuity;
loops and seeks start new runs even when the voice identifier persists.

Reproduce verification and comparisons with:

```sh
.venv/bin/python -m tools.validate_sustained_examples
```

The machine-readable report is `data/sustained-validation.json`.

## Recursive second layer

Generate a recursive study with the same selection settings:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python -m tapeheads analyze \
  data/audio/chirp_sequence/mono_48000.wav \
  --encoder wav2vec2 --recursive --video --tracking-policy overlap_union \
  --output outputs/chirp_sequence_wav2vec2_layer2
```

Use a new output directory for each run. For the full Elvis study, use
`data/audio/blue_suede_shoes/source.flac` and
`outputs/blue_suede_shoes_full_wav2vec2_layer2`. Recursive mode supports
`wav2vec2`, `hubert`, and `muq`. CLAP remains available for first-layer studies;
its second shifted Swin block is not yet mapped for recursion.

The output directory contains `trace.json`, `mix.wav`, `direct.wav`,
`recursive.wav`, the video and preview, and `layer_01`/`layer_02` subdirectories
with native layer traces and stems. The top-level trace embeds both native
traces and all composed playback events; layer asset paths resolve within the
corresponding `layer_assets[].directory`.

The recursive branch follows the layer-one playback timeline at the frame
pointed to by layer two. It includes every selected child voice there, not just
the corresponding head index. The direct branch plays the original source at
that intermediate frame. Weights are 35% direct and 65% recursive, with fixed
voice-budget attenuation and an additional reduction for current-time returns.
The video separates output, intermediate, and original-source positions with
directional vectors. Dot textures and waveforms are generated entirely in code.

Use `--duration 4 --verify-attention` to check short native extraction, and
`python -m tapeheads verify OUTPUT/trace.json` to audit paths, gain, timing,
source integrity, and exported assets. This implements recursive tape playback;
it does not claim exact attribution of learned representations to waveform
samples.

## Dark comparisons

Run the saved batch script once per encoder:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python -m tools.render_dark_studies --encoder wav2vec2
HF_HUB_OFFLINE=1 .venv/bin/python -m tools.render_dark_studies --encoder muq
HF_HUB_OFFLINE=1 .venv/bin/python -m tools.render_dark_studies --encoder hubert
```

Each run creates both `outputs/chirp_sequence_ENCODER_dark` and
`outputs/blue_suede_shoes_full_ENCODER_dark`, with full recursive traces, branch
stems, mixes, and 1080p videos. Selection settings are stored in both layer
traces. Reports are saved to `data/dark-ENCODER-validation.json`.

The dark renderer removes the instructional labels and nearby-path overlays.
Bezier endpoints track playback time; brackets show the corresponding voice's
accumulated playback bounds. Native second-layer attention selects first-layer
frame representations, which mix first-layer heads. The displayed branching
across selected child voices is the recursive sonification policy.

Audit that the visible parent and child endpoints lie inside their own
playback brackets, and compare uninterrupted run lengths with:

```sh
.venv/bin/python -m tools.audit_dark_paths
```

This writes `data/dark-path-audit.json`, including cross-head path counts.

## Eight-layer MuQ experiment

```sh
HF_HUB_OFFLINE=1 .venv/bin/python -m tapeheads analyze \
  data/audio/chirp_sequence/mono_48000.wav --encoder muq --recursive \
  --layers 8 --max-layer-blocks 6 --output outputs/chirp_sequence_muq_depth8 \
  --video --tracking-policy overlap_union
```

The matching full recording is in `outputs/blue_suede_shoes_full_muq_depth8`.
Each directory includes `depth_01.wav` through `depth_08.wav` for fixed-depth
listening, a progressively deeper `mix.wav`, a video, the compact graph in
`trace.json`, and eight independently auditable native traces. Run the usual
`verify` command on the root trace to reconstruct and compare every depth bus.

The eight-layer experiment uses the broader study preset (entry 1.5,
continuation 1.1, 350 ms averaging, 300 ms minimum blocks). The voice count is
configurable from 1 through 16. The selection flags, including `--exit-alpha`,
can override the preset. Two-layer mode with `overlap_union` retains its earlier
behavior. CLAP is still first-layer only.

The faint diagonal labels identify exact synthetic events or estimated musical
sections. Correct `annotations.sections` manually for verified verse/chorus
labels. Optional user-provided `annotations.lyrics` entries use source-sample
`start` and `end` indices and a `text` field; then rerun `render`. The samples are
at the analysis rate recorded in the root trace, not the original recording's
sample rate.
