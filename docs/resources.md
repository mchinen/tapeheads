# Resource budget

Measured September 18, 2026. GiB means 2^30 bytes; GB means 10^9 bytes.
Checkpoint sizes below come from Hugging Face file metadata, without downloading
weights. Runtime estimates are planning allowances, not benchmarks.

## Reference environment

The initial experiments used a CPU environment with 62 GiB of system memory.
Runtime and storage observations below describe that environment, not minimum
requirements. Setup scripts keep model and installer caches under the project's
`.cache/` directory.

## Checkpoint sizes

Download one weight format per checkpoint. Repository totals can include
multiple copies of the weights for different frameworks or serialization
formats.

| Checkpoint | Selected file | Bytes | GiB | Heads per layer |
| --- | --- | ---: | ---: | ---: |
| [wav2vec 2.0 base](https://huggingface.co/facebook/wav2vec2-base/tree/main) | `pytorch_model.bin` | 380,267,417 | 0.354 | 12 |
| [MuQ](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter/tree/main) | `model.safetensors` | 1,333,825,096 | 1.242 | 16 |
| [wav2vec 2.0 ASR comparison](https://huggingface.co/facebook/wav2vec2-base-960h/tree/main) | `model.safetensors` | 377,607,901 | 0.352 | 12 |
| [HuBERT base](https://huggingface.co/facebook/hubert-base-ls960/tree/main) | `pytorch_model.bin` | 377,569,754 | 0.352 | 12 |
| [WavLM base plus](https://huggingface.co/microsoft/wavlm-base-plus/tree/main) | `pytorch_model.bin` | 377,617,425 | 0.352 | 12 |

The first two total **1.60 GiB**. All five total about **2.65 GiB**.
Read metadata through `https://huggingface.co/api/models/{repo}?blobs=true`.
The exact observed revisions are:

| Checkpoint | Revision |
| --- | --- |
| wav2vec 2.0 base | `0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8` |
| MuQ | `0562a57814f6f8bbd9fdea0a25921a2fce1a841a` |
| wav2vec 2.0 ASR | `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` |
| HuBERT | `dba3bb02fda4248b6e082697eee756de8fe8aa8a` |
| WavLM | `4c66d4806a428f2e922ccfa1a962776e232d487b` |

Wav2vec base uses 16 kHz input and roughly 50 frames/s. MuQ uses 24 kHz
input and roughly 25 frames/s. Its top-level `encoder_depth = 12` overrides
`w2v2_config.num_hidden_layers = 24` during construction; budgeting for 24
active layers from the nested config would be incorrect. Verify this against
the [MuQ config](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter/blob/main/config.json)
and [constructor](https://github.com/tencent-ailab/MuQ/blob/main/src/muq/muq/models/muq_model.py).

MuQ's authors recommend float32 inference because reduced precision can produce
NaNs. Its code is MIT-licensed; its weights are CC BY-NC 4.0. These are
openly available weights with a noncommercial restriction, which matters if
this project later becomes a commercial product. See the
[official repository](https://github.com/tencent-ailab/MuQ).

## Attention memory

For one batch item and one layer, a dense attention tensor occupies
`heads * frames * frames * bytes_per_value`. These estimates use float32 and
nominal frame rates; exact boundary frame counts vary slightly.

| Input duration | wav2vec: 12 heads, 50 Hz | MuQ: 16 heads, 25 Hz |
| --- | ---: | ---: |
| 10 seconds | 0.011 GiB | 0.004 GiB |
| 30 seconds | 0.101 GiB | 0.034 GiB |
| 3 minutes | 3.621 GiB | 1.207 GiB |
| 5 minutes | 10.058 GiB | 3.353 GiB |

These are matrix sizes, not peak process memory. Logits, probabilities,
intermediate activations, frontend features, and library workspaces add to the
peak. Materializing attention for every layer multiplies retained matrices by
12 for these two models. Five-minute wav2vec attention alone would then exceed
120 GiB. A Python list or JSON representation would be substantially larger.

For block 0, compute the frontend across the complete signal, then evaluate
query blocks against all keys and discard probabilities after reduction.
For example, 128 wav2vec query rows against 15,000 keys across 12 heads use
about 88 MiB for one float32 score block. Additional workspaces remain, and
MuQ's convolutional frontend also needs profiling. No GPU-memory or runtime
claim is justified until a short extraction is benchmarked.

## Working disk allowance

Reserve **15–25 GiB** for an initial environment and a few experiments:

| Item | Planning allowance |
| --- | ---: |
| Initial two checkpoints | 1.60 GiB |
| Python environment and installation caches | 4–10 GiB |
| Source fixtures, traces, audio, and videos | 2–5 GiB |
| Temporary downloads and working margin | 5–8 GiB |

Environment size depends on CPU versus accelerator packages. Use one shared
weight cache and avoid duplicate snapshots. Keep generated outputs and caches
out of Git when implementation begins.

A five-minute 48 kHz mono float32 stem is approximately 55 MiB; 16 such stems
are about 0.86 GiB. A five-minute video at 8–16 Mbit/s is approximately
0.28–0.56 GiB before audio and container overhead. These are chosen output
budgets, not required encoding rates. A five-minute 1080p30 RGB frame sequence
would consume about 52 GiB uncompressed, so stream frames into the encoder.

Keep the event JSON compact. Dense attention is opt-in and uses a binary
sidecar with its own budget. Event counts can still approach heads times
queries in a rapidly switching signal; measure trace size rather than assuming
all files remain tiny.

## Implementation snapshot

On September 19, 2026, the prototype environment and four checkpoints were
installed: wav2vec 2.0 base, HuBERT base, MuQ, and CLAP HTSAT unfused. The pinned
[CLAP checkpoint](https://huggingface.co/laion/clap-htsat-unfused/tree/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a)
adds a 614,525,833-byte weight file (0.572 GiB). The loader retains its audio
branch for attention extraction.

Observed usage after the first renders:

- Project cache, including weights and installer files: about 2.9 GiB.
- Python environment: about 1.8 GiB.
- Prototype outputs: about 353 MiB.

The complete 122.64-second Charles Dodge trace occupies
about 123 MiB as indented JSON. That prototype stored detailed per-query candidates. Later studies use
compressed native traces and compact recursive schedules; see
[Collection studies](collection-studies.md).
