# Tapeheads

Tapeheads sonifies audio encoder attention as independently moving tape heads.
By default, up to three remote voices per layer select source audio more than
100 ms from query time, with a separate quiet nearby monitor. A saved JSON trace drives
audio stems and a waveform video with curved arms and span brackets. Recursive
mode follows deeper-layer selections through lower-layer playback to the input.
The latest batch studies use up to three active heads per layer, each with up
to three regions, across three layers. This is an artistic tape mapping, not
exact model attribution.

## Run

Use Python 3.11 for the analysis and development environment, with FFmpeg and
FFprobe available for audio preparation and video rendering. Create the CPU
environment:

```sh
bash tools/setup_analysis.sh
```

List encoders, generate a repeated-chirp fixture, and render a short example:

```sh
.venv/bin/python -m tapeheads encoders
.venv/bin/python -m tools.generate_chirp_sequence
.venv/bin/python -m tapeheads analyze data/audio/chirp_sequence/mono_48000.wav \
  --encoder wav2vec2 --duration 12 --output outputs/my_example --video
.venv/bin/python -m tapeheads verify outputs/my_example/trace.json
```

Choose `wav2vec2`, `hubert`, `clap`, or `muq`. Install MuQ's extra dependencies
with `bash tools/setup_muq.sh` before selecting it. Weights download into the
project's `.cache/huggingface` directory on first use.

Audio and rendered outputs are local assets excluded from Git. To prepare the
recordings used in the studies, follow [Audio collection](docs/audio-data.md),
or pass your own audio file to `analyze`.

Study indexes link to locally generated files under `outputs/`; those links
will not open on GitHub or in a fresh clone until the studies are reproduced.
Committed validation reports and batch statuses are experiment snapshots,
not live job status.

Omit `--duration` to analyze the complete input. Wav2vec, HuBERT, and MuQ use
full-input keys with blocked queries. CLAP uses independent 10-second contexts
and native local windows, explicitly labeled in every trace. It does not
provide full-song attention.

Add `--recursive` to compose the first two layers with `wav2vec2`, `hubert`,
or `muq`:

```sh
.venv/bin/python -m tapeheads analyze data/audio/chirp_sequence/mono_48000.wav \
  --encoder wav2vec2 --recursive --video --output outputs/my_recursive_study
.venv/bin/python -m tapeheads verify outputs/my_recursive_study/trace.json
```

Recursive exports include fixed-depth stereo WAV files, a progressive mix,
and a dark vector-and-dot-field video. Add `--layers 3` for three layers.
CLAP recursion is not yet enabled. New CLI runs follow current support; use
`--tracking-policy overlap_union` to reproduce older experiments.

Regenerate the revised Espresso, Bohemian Rhapsody, Charles Dodge, and chirp
studies with either encoder:

```sh
.venv/bin/python -m tools.run_responsive_studies --encoder muq
.venv/bin/python -m tools.run_responsive_studies --encoder wav2vec2
```

The [current-support studies](docs/responsive-studies.md) link the revised
videos and JSON traces. These studies audit every native playback run against
current attention support and validate recursive audio reconstruction. See
[current-support playback](docs/design.md#current-support-playback) for the
selection and mixing rules.

See [Prototype usage](docs/prototype.md) for outputs, validation, and limits;
[Audio collection](docs/audio-data.md) for the prepared recordings and test
signals; and [Design](docs/design.md) for the longer-term plan.

The [eight-layer collection](docs/collection-studies.md) compares full songs,
a nature montage, and repeated chirps across wav2vec 2.0, MuQ, and HuBERT.
It uses up to six voices per layer and includes video, JSON, and fixed-depth
audio links with validation results.

## Development

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
bash tools/check.sh
```

Project files follow the applicable [Google style guides](https://google.github.io/styleguide/).
`tools/check.sh` runs Pyink, Ruff, pytype, and unit tests. Their configuration is
in `pyproject.toml`; pytype checks the package, tools, and tests using Python
3.11. Use descriptive names for clocks, samples, and model parameters. Comments
should explain timing constraints or implementation choices rather than repeat
the code. Rendered labels use sentence case; keep acronyms such as DTMF and
units such as Hz intact.

Selection blocks, voice state, and playback segments have shared types in
`tapeheads/trace_types.py`. Versioned trace metadata uses `JsonObject`; JSON
Schema and runtime validation still check its contents, array shapes, and
sample bounds. Type checking does not replace those checks.

The analysis environment is recorded in `requirements-lock.txt`; install its
CPU Torch packages from the PyTorch CPU index, as the setup scripts do.

## Three-head batch

Run every prepared source in full with MuQ and wav2vec2, progressing through
depths one, two, and three. The `per-head-support/5` preset selects at most
three active heads per layer, each with up to three regions. It uses entry
alpha 1.5, continuation alpha 1.1, 500 ms averaging, a 200 ms minimum block,
and a 40% head continuity bonus. Initial spans are capped at five seconds;
extensions require fresh support at every query.

Install MuQ's dependencies first. Once both encoders' weights are cached, run:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python -m tools.run_per_head_studies
```

Omit `HF_HUB_OFFLINE=1` if the weights still need to download. This preset is
configured by the batch runner; the main `analyze` CLI continues to default to
`current_support` with three remote voices across the entire layer.

The runner resumes completed studies when the package code signature matches,
isolates model jobs in subprocesses, and updates
[the three-head batch index](docs/three-head-studies.md) and
[data/three-head-batch.json](data/three-head-batch.json). Per-file logs are under
`data/logs/three-head/`. Outputs end in `_threeheads3`; trace files use
`.json.gz` and audio uses PCM24 `.flac`. Native traces contain playback
schedules without individual head WAV stems; fixed-depth audio and the final
mix remain available. Verify an export with:

```sh
.venv/bin/python -m tapeheads verify \
  outputs/chirp_sequence_wav2vec2_threeheads3/trace.json.gz
```

The batch includes long speech and classical recordings, so CPU processing
can take many hours. It requires at least 2 GiB free before starting each
study. The [earlier per-head studies](docs/per-head-studies.md) remain available
with their original settings and `_perhead3` outputs.
