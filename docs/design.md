# Tapeheads design

Status: Design with an initial implementation. Started September 18, 2026.
See [Prototype usage](prototype.md) for implemented behavior and limitations.

## Purpose

Sonify the first transformer layer of open audio encoders, starting with
wav2vec 2.0 and MuQ. Show every attention head above a shared audio waveform.
Trace the source intervals each head attends to as query time advances, and
play sustained intervals until attention moves elsewhere.

Inputs include songs, speech, sine waves, and chirps. Outputs will include a
reproducible JSON trace and a minimal video synchronized to the sonification.
Later work can extend the trace to deeper layers.

## Style

Follow the Google style guides for each implementation language and Google's
technical writing guidance for documentation. Use clear names, sentence-case
headings, and explicit units. JSON uses two-space indentation.

The visual direction combines restrained fintech and ElevenLabs interfaces:
clear typography, generous space, a waveform, and precise head markers. Draw
subtle inspiration from Nam June Paik and John Cage's tape work through motion,
splicing, repetition, and silence.

## Research status

The initial survey is recorded in [Resource budget](resources.md) and
[Related work](related-work.md). Checkpoint metadata and upstream extraction
code were inspected. The first-layer prototype now includes wav2vec, HuBERT,
MuQ, and CLAP adapters; its dependencies and pinned weights are installed.

## Scope and terminology

The first layer means transformer block index 0, after the audio frontend. The
frontend itself has no attention heads. Start with pretrained
`facebook/wav2vec2-base`; add `OpenMuQ/MuQ-large-msd-iter` next. An ASR-finetuned
wav2vec checkpoint is a separate comparison, not an interchangeable baseline.

These encoders use bidirectional context. They process an input sequence; our
animation scans its query rows afterward. The moving query cursor represents
an inspection timeline, not a causal account of inference or human listening.
Attention weights describe mixing of representations, not isolated waveform
contributions or a complete explanation of a model's decision.

Use three explicit clocks:

- Query time: the audio position of the attention row being inspected.
- Source time: the key interval selected by that row.
- Output time: the playback position in the sonification and video.

Default output duration equals input duration, with query time advancing at
normal speed. Each head has an independent source playback cursor.

## Attention extraction

Use evaluation mode, disable gradient recording and training augmentation, and
record actual preprocessing. Preserve the original audio for audition; create
a resampled mono analysis signal for each model.

For wav2vec 2.0, start with eager attention and the documented attention outputs
on short clips. Confirm the installed backend returns real per-head matrices.
Capture only block 0 for production. Deeper blocks cannot affect block 0 in a
feedforward encoder, so an adapter can stop once the required attention has
been obtained. Check this against a complete forward pass on a short fixture.
The [Transformers documentation](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2)
describes per-layer outputs shaped as batch, heads, queries, and keys.

MuQ needs a dedicated adapter. Its public `forward` returns hidden states, and
its internal encoder call does not request attentions. Capture the first
Conformer attention module's real normalized probabilities, or modify the
adapter's call to request them. Preserve positional terms, scaling, masks,
and preprocessing; plain QK multiplication may omit model-specific behavior.
See the upstream [wrapper](https://github.com/tencent-ailab/MuQ/blob/main/src/muq/muq/muq.py)
and [model implementation](https://github.com/tencent-ailab/MuQ/blob/main/src/muq/muq/models/muq_model.py).

For full songs, retain full-sequence keys and compute attention in query
blocks, reducing each block to events immediately. Softmax must normalize over
all valid keys. This reduces peak matrix storage without changing the context;
it does not remove quadratic compute. Implement model-specific positional
indexing before claiming equivalence. Splitting the audio into independent
windows changes attention and must be exported as a separate context mode.

Frame mapping belongs to each adapter. Wav2vec 2.0 has a 320-sample stride at
16 kHz and a 400-sample convolutional receptive field. Its nominal step is
20 ms; the receptive field is 25 ms. These are frontend anchors, not the full
context of transformer representations. MuQ has a nominal 40 ms step; derive
its exact offset and boundary behavior from STFT centering and convolution
padding. Verify both mappings with impulse and boundary fixtures. Store
explicit frame boundaries when a constant offset and hop are insufficient.

## Selecting sustained source spans

The implemented `average-overlap/2` policy operates on native post-softmax
probabilities and uses these defaults:

1. Normalize thresholds by the number of visible audio keys: entry `2 / N`,
   continuation `1.25 / N`. Never renormalize after temporal masking.
2. Reserve a symmetric ±100 ms neighborhood for quiet current-time monitoring.
   Remote blocks exclude every frame bin intersecting that neighborhood.
3. Propose connected regions with a duration-weighted 200 ms box average.
   Qualify each region using its original duration-weighted mean, and require
   at least 120 ms duration. Individual frames can fall below threshold.
4. Rank new blocks by integrated positive excess over their entry threshold.
   Keep at most three remote blocks across the entire layer. A head can own
   zero, one, or multiple blocks. Uniform attention produces no voices.
5. Retain a voice while its current support overlaps a region qualifying at
   the continuation threshold. Keep its identifier and source cursor. Its
   playback bounds accumulate the union of overlapping supported regions;
   shrinking support does not restart the phrase. This can replay historically
   attended samples outside the current support. Export both bounds explicitly.
6. End unsupported voices immediately and fill vacant slots with ranked new
   blocks. There is no dwell requirement or challenger preemption in version 2.

Remote voices play at 1x and loop inside their accumulated region. The scheduler
checks the ±100 ms guard on every source-to-output run, including loops. It can
seek or mute to preserve that guard. Each discontinuity has 5 ms nonoverlapping
boundary fades; there are no overlapping crossfades.

For each head, export current-neighborhood mass and mean relative to uniform
attention. Highlight qualifying nearby attention in amber. If any head exceeds
the entry threshold there, play one shared current-time monitor at 15% of a
remote voice's gain. This monitor is an explicit guard exception and does not
consume one of the three remote slots. It follows query time, rather than
adding duplicate current-time voices for multiple heads.

Export per-head stems, a quiet nearby stem, and a stereo mix. Remote voices use
fixed gains and head-index stereo positions. Record gains and all sample bounds
in the same trace consumed by audio and video rendering.

## Trace contract

Use `tapeheads/2` for sustained polyphonic traces; retain validation of legacy
`tapeheads/1` exports. JSON contains compact
selection events and provenance; optional dense tensors live in binary sidecars.
The contract must cover:

| Group | Required information |
| --- | --- |
| Source | Relative asset path, SHA-256, sample rate, channels, samples, duration |
| Model | Repository, immutable revision, adapter and dependency versions |
| Analysis | Layer index, head count, dtype, sample rate, preprocessing, context mode |
| Frames | Frame count, hop, anchor offset, boundary convention, padding mask |
| Selection | Threshold, exclusion radius, volume definition, hysteresis, dwell, tie-breaks, version |
| Events | Head, query bounds, key bounds, mass, excess volume, ranks, diagnostics |
| Playback | Output sample bounds, source sample bounds, seeks, loops, fades, gain |
| Rendering | Dimensions, frame rate, palette, font, audio assets |

Use integer sample indices for audio scheduling and half-open ranges everywhere.
Seconds are derived display values. Store analysis and original source sample
rates separately. Include the resampling ratio and rounding convention.
Events must identify whether bounds refer to model frames or source samples.
Every head remains present, including inactive heads. Do not serialize NaN or
infinity. A later implementation should provide a JSON Schema and a small
validated trace fixture; this document is not yet that schema.

## Visual direction

Use a warm near-white background, charcoal waveform, pale rules, and one muted
accent for the selected head. Prefer a neutral sans-serif face with tabular
numerals. Motion should explain a seek, hold, loop, or splice.

The shared source waveform spans the screen horizontally. Above it, arrange
12 or 16 slim lanes, one per head. Each lane uses the same source-time axis:
a bracket marks both ends of the selected span, showing its true duration.
A restrained cubic Bezier arm connects the head at query time to the bracket's
midpoint. A separate small marker shows the actual playback cursor. Keep the
arm within its lane to avoid crossings. Show the 200 ms current-neighborhood band faintly
around the query cursor. A rotating arm is an optional visual variant; its
endpoint must still identify the bracket precisely. A thin vertical query line moves independently across all
lanes and the waveform. Highlight the focused head; keep other lanes visible.
Avoid drawing every query-key edge in the overview.

The header contains the input name, model, layer, and time. The lower control
strip contains play, seek, source/mix, and head solo/mute. Show a magnified
waveform around the focused span when full-song scale hides detail. Keep
technical extraction details in an inspector or the trace rather than the
main performance view. A paused frame should still be legible.

Paik and Cage inform the behavior: independently moving tape heads, cuts,
repetition, held fragments, and silence. Avoid ornamental glitch, simulated
hardware, or literal imitation of a specific artwork.

## Pipeline and treemusic inspiration

Separate model adapters, selection, audio rendering, and video rendering.
A Python analysis pipeline writes the trace. A sample-accurate offline audio
renderer reads its playback schedule. An interactive viewer and deterministic
video renderer read the same events. Stream video frames to FFmpeg rather than
keeping a directory of uncompressed frames. Begin with 1080p at 30 fps.

The adjacent `treemusic` project provides useful precedents:

- `README.md` and `TreeJson.cpp`: one exported description feeds multiple views.
- `tools/render_video.py`: deterministic frames piped directly into FFmpeg.
- `viz/ikeda.html`: scrubbing, focus modes, and audio-synchronized inspection.
- `viz/websynth-worklet.js`: audio and animation share the same data.
- `HANDOFF.md`: distinguish structural labels from what actually executes.

Borrow those separations and trace discipline. Build a lighter visual surface
with readable head lanes. The existing C++ synthesis chain is not needed for
source-audio splicing. Inspect licenses before copying implementation code.

## Deeper layers

First extend the same per-head inspection to block 1. Keep native attention
separate from composed paths. Attention rollout, such as multiplying layer
matrices with a documented residual convention, is only an approximation:
value projections, nonlinearities, head mixing, and Conformer operations mean
it is not an exact attribution to original audio. Recursive tape playback is a
separate artistic mapping with its own versioned rules.

## Implementation milestones

1. Build short sine, chirp, impulse, silence, and repeated-burst fixtures.
   Confirm frame mapping, row normalization, padding exclusion, and head count.
2. Extract wav2vec block 0 on 10–30 seconds of audio. Compare blocked extraction
   with ordinary full attention within a stated numeric tolerance.
3. Export events and stems. Check stable spans, jumps, silence, loops, and
   crossfades using hand-constructed attention maps as well as real signals.
   Include dominant diagonal attention, second-choice blocks, exclusion-split
   spans, and held future spans that approach the moving guard interval.
4. Render one minimal video from the same trace. Confirm seeks and audio/video
   timing agree to within one video frame.
5. Add MuQ and repeat extraction checks. Profile full songs before scaling up.
6. Add layer 1 inspection and then an explicitly labeled recursive experiment.

The first prototype has exercised all four encoders and exports traces, stems,
and video. Thresholds remain experimental. Peak memory has not been profiled.
Implementation differences, including boundary fades instead of overlapping
crossfades, are documented in [Prototype usage](prototype.md).

## Recursive second-layer implementation

The `tapeheads/3` performance embeds the two native layer traces and references
independently verifiable layer assets. Wav2vec 2.0, HuBERT, and MuQ execute the
complete first native block before extracting block 1. The frontend, residual,
feedforward, and model-specific convolution operations remain native. Full-song
waveform attention retains all keys. Recursive CLAP is not enabled: its second
Swin block needs a validated shifted-window time projection.

The second layer selects at most three remote voices using the same average and
overlap policy. Each parent voice reads an intermediate first-layer timestamp
at 1x. It contributes two branches:

- Direct: original audio at that intermediate timestamp, weighted by 0.35.
- Recursive: all selected first-layer playback voices active at that timestamp,
  weighted together by 0.65 and divided by the first-layer voice budget.

The quiet first-layer monitor participates as a reduced-gain child. The second
layer also has its shared quiet current-time monitor. A composed child can
return close to output time even when both individual hops are remote. Such
paths receive another 0.15 gain factor within the ±100 ms output neighborhood.
This is an artistic recursive tape mapping, not exact model attribution.

Each flattened event stores parent and child voice/segment identifiers, output
bounds, intermediate start, original-source bounds, inherited fade envelopes,
branch, and final gain. The audio renderer multiplies the two fade envelopes;
it does not restart fades at every frame or intersection. Each branch is also
exported as a separate stereo stem. Fixed gains keep the worst-case summed
amplitude below 0.8 times the source peak, without a limiter or normalization.

The report-style video uses two aligned head planes, procedural dot fields, a
dotted waveform, directed paths, and source-position dots. Green marks recursive
paths, charcoal marks direct audio, and amber marks nearby paths. All geometry
is drawn in code; there are no external or generated image assets.

## Dark recursive view and longer-span comparisons

The dark view connects a layer-two query directly to each selected intermediate
first-layer timestamp with a Bezier curve, then connects that timestamp to the
audible input position. It no longer routes through a horizontal segment on the
second-layer plane. Intermediate brackets belong to the parent voice; input
brackets belong to the child voice. Both show accumulated playback bounds and
share the curve endpoint's clock, rather than mixing current-support brackets
with historical playback cursors.

A second-layer head attends to tokens whose representations already combine
first-layer heads through output projection, residuals, and feedforward
operations. There is no native head-to-head attention matrix here. Branching
curves expose all selected first-layer playback voices at the intermediate
timestamp as an artistic mapping. They do not assert separately measured
connections between native head channels.

Nearby monitor paths and composed current-output returns remain reduced in the
audio, but are omitted from the drawing. Amber previously conflated proximity
to the intermediate frame with proximity to the output cursor. The dark view
removes that category and the instructional text overlays.

The six dark comparisons use entry alpha 1.5, continuation alpha 1.1, a 350 ms
averaging window, and minimum blocks of 300 ms. These settings seek broader,
more stable support. A 300 ms selected block does not guarantee every audible
run lasts 300 ms: attention can leave it early, a source cursor can reach an
edge, and recursive intersections can be shorter. Short runs are not all loops.

### Interactive hosting

Inference and trace generation stay offline. A static browser application could
perform scrubbing, mute/solo, and vector rendering from compact precomputed
traces with Web Audio and Canvas or SVG. It would not need a Worker request per
frame. Cloudflare documents free static asset requests, a 10 ms free Worker CPU
limit, and a 25 MiB individual static asset limit. The current archival traces
can exceed that limit substantially; browser exports would need compact event
fields, timeline chunks, and suitable audio assets before deployment.

Sources checked September 21, 2026:
[Workers limits](https://developers.cloudflare.com/workers/platform/limits/)
and [Workers pricing](https://developers.cloudflare.com/workers/platform/pricing/).

## Eight-layer shared recursion

The deep experiment uses `tapeheads/4` with one compact native playback schedule
per layer and SHA-256 references to full native traces. Layers 0 through 7 run
with their real preceding model blocks. MuQ's first eight layers are compared
to native attention on a short signal fixture before the full examples.

Each layer supports six remote voices in the example configuration. Its audio
bus sums the direct source at attended timestamps (20%) and the complete
previous-layer bus read at those timestamps (80%). The first layer reads the
source only. Shared buses avoid exponential duplication while retaining every
selected audio branch. This is an artistic tape recursion, not exact attribution.

Each bus records its loudness correction: target RMS 0.1, gain capped at 4, and
peak capped at 0.8. The final mix progresses through depths 1–8 in equal output
time stages, with 80 ms depth crossfades. Full-duration audio for every fixed
depth is also exported. Source clocks remain absolute within the original input.
The quiet monitor and guard apply at each hop. Unlike the two-layer flattened
mapping, deep composed returns near the final output cursor do not get an
additional attenuation; this distinction is explicit in the trace.

The video uses eight narrow bands and head activity dots. It allocates at most
128 visible edges across complete paths from each active root toward the input,
favoring longer runs when relative gains tie. Quiet branches remain hidden.
This is a display subset: no audio paths are dropped. Endpoints and brackets
refer to the same native tape run and intermediate timestamp.

Synthetic annotations come from the original generator manifest. Music
boundaries are estimates from spectral novelty and similarity, labeled opening,
contrast, and return. They are not verified verse, chorus, or bridge labels.
Annotations can be corrected in the root JSON without redoing model inference.
The optional `annotations.lyrics` array accepts user-provided objects with
`start`, `end` (source sample indices), and `text`, drawn as light diagonal text.
No song lyrics have been retrieved or transcribed for these examples.

### Full-song collection extraction

The collection runner captures the first eight layers in one forward pass.
Native surrounding blocks remain intact; CPU scaled dot-product attention
avoids materializing the full preceding-layer attention matrices. Captured
Q/K tensors produce query blocks normalized over all input keys. Four-second
fixtures compare all eight layers with independent native attention for each
temporal encoder at absolute tolerance 0.00001.

Adapter `single-pass-sdpa-qk/2` evaluates the softmax in float64 before
exporting float32 probabilities. A sharply peaked long wav2vec row exceeded
the original mass tolerance with float32 softmax; the tolerance remains
unchanged. Existing validated version 1 traces preserve their provenance.

Native JSON traces can use deterministic gzip compression. Root graph traces
remain plain JSON, and validation checks compressed asset hashes before
reconstructing audio. See [Collection studies](collection-studies.md) for
outputs, source audits, and remaining diagnostic observations.

## Current-support playback

The `current-support/3` selector replaces historical union bounds for new CLI
runs. At each query it ranks qualifying current blocks across heads, using a
5% continuity bonus for an existing voice with at least 50% intersection over
union. Stronger candidates can displace a still-supported voice. Playback
bounds always equal current support; an old, weaker continuation threshold
cannot preserve an abandoned interval. Overlapping candidates sharing at least
50% intersection over union occupy one slot. Heads can remain silent.

The revised MuQ and wav2vec studies use three voices per layer, entry alpha
1.5, 120 ms averaging, and a 120 ms minimum block. A run can last longer than
15 seconds only while its source read remains within selected support at every
query. Shorter runs remain possible when attention moves or the guard cuts a
read. Validation checks actual scheduled samples against current support.

For one to three layers, `responsive-routes/1` composes absolute source reads
before mixing. Each intermediate frame contributes original audio at its own
timestamp with weight 0.35, plus all selected lower-layer reads with weight
0.65. The bottom layer reads original audio only. Each hop divides by its voice
budget. Stereo placement is applied at the audible head, not multiplied through
ancestor pans, which could otherwise silence opposite-side children.

Routes reaching exactly the same source offset at the same output time share
the mean of their gains. This avoids stacking identical reads while preserving
all selected branches. Composed returns within the output guard receive the
quiet-monitor gain. Each fixed-depth mix is normalized once; normalized buses
are not fed back into this recursive renderer.

One 3 ms decaying offset correction smooths actual mix splices. Contiguous
query cuts with unchanged reads and gains cause no new envelope. There are no
multiplied fades at recursive hops. A short fade precedes actual silence;
corrections never extend selected playback into silence. Video frame rate does
not control audio scheduling. Cadence spectra are diagnostics, not proof that
a recording contains or lacks audible hum.

The dark video adds muted curves from intermediate reads to their direct input
audio positions. Curves retain their measured timestamps, including legitimate
near-vertical connections. Progressive exports visit depths one, two, and three;
full-length fixed-depth WAV files permit comparison at the same query time.
Music exports omit guessed structural labels. Supplied timed lyrics can be
added to `annotations.lyrics`; synthetic labels remain generator-derived.

Use `--tracking-policy overlap_union` to reproduce the older CLI selection and
recursive rendering. The earlier sections describe those versioned experiments,
not the current-support policy. Library callers must select the new policy
explicitly. Standalone native layer stems retain their original boundary fades;
the revised recursive exports use the composed route renderer described here.

## Per-head continuation studies

The `per-head-support/4` policy permits zero to three blocks independently in
**each head**. The batch preset uses entry alpha 1.5, continuation alpha 1.3,
200 ms averaging, a 200 ms minimum block, and a 15% continuity bonus. Blocks
rank by integrated excess above entry, with the ranking contribution capped
at a five-second equivalent so growing spans do not win solely through length.

New reads acquire at most five seconds of source support. A continued voice
can extend its endpoint by the elapsed query time only inside freshly qualified
support. Its start can move forward as support shrinks. Neither current source
bounds nor playback bounds accumulate an abandoned historical interval.

`per-head-shared/1` mixes raw shared buses to handle up to 48 voices per layer
without expanding every recursive path. Intermediate audio contributes 35%;
the previous raw bus contributes 65%. Identical reads within a layer share mean
gains. This is local deduplication, not global original-source path deduplication.
Parent panning does not multiply child panning. Splice correction and loudness
normalization apply once to each final fixed-depth mix. Guarding and the quiet
monitor apply locally at each hop; composed returns have no separate guard gain.

The video marks every 15 seconds. Long recordings label a subset of those ticks
to avoid overlapping text. The display has a 384-edge detail budget; all root
voices are shown, but recursive descendants are a display subset. Audio includes
all selected branches. Full native traces record the independent per-head cap.

The batch uses gzip JSON and PCM24 FLAC; quantization is checked against the
existing 0.000001 reconstruction tolerance. Native head WAV files are omitted
in schedule-only exports. Identical source files across layers share storage.
Previous outputs are preserved. The batch refuses to start another study with
less than 2 GiB free, and records per-study failures before trying the next one.

Section references in `data/section_labels.json` provide approximate Espresso
and Bohemian Rhapsody labels, explicitly marked with a tilde. They are not yet
aligned by audition to these downloaded editions. Other songs remain unlabeled
rather than receiving invented verse/chorus assignments. No lyrics are included.
Synthetic section labels remain derived from the generator manifests.

## Three-head revision

`per-head-support/5` selects at most three active heads per layer, each with
up to three regions. Heads rank by their strongest region with a 40%
continuity bonus. The new preset uses 500 ms averaging and exit alpha 1.1;
entry alpha remains 1.5. An overlapping acquisition cannot replace an
already supported voice merely because its fresh five-second crop scores
higher. Disjoint regions still compete, and absent support stops playback
immediately. Initial spans remain capped at five seconds, with extensions
requiring fresh support at every query. These settings favor longer runs
without guaranteeing a duration for any recording.

The revised batch writes `*_threeheads3` directories and records progress in
`data/three-head-batch.json` and `docs/three-head-studies.md`.
