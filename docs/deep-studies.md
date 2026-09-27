# Eight-layer MuQ studies

These links refer to local artifacts under `outputs/`, which are excluded
from Git. Reproduce the studies to open them locally. Recorded results and
batch statuses are snapshots of the original runs, not live job status.

These films increase recursive depth from one through eight over the course of
the input. Every layer can select up to six remote voices from sixteen native
heads. The dark field shows complete paths through a subset of the graph; the
audio includes all selected branches. No head-index matching is imposed.

| Input | Video | Graph trace | Progressive mix | Fixed depth eight |
| --- | --- | --- | --- | --- |
| Four chirps and test signals | [Watch](../outputs/chirp_sequence_muq_depth8/tapeheads.mp4) | [JSON](../outputs/chirp_sequence_muq_depth8/trace.json) | [Listen](../outputs/chirp_sequence_muq_depth8/mix.wav) | [Listen](../outputs/chirp_sequence_muq_depth8/depth_08.wav) |
| Full Blue Suede Shoes | [Watch](../outputs/blue_suede_shoes_full_muq_depth8/tapeheads.mp4) | [JSON](../outputs/blue_suede_shoes_full_muq_depth8/trace.json) | [Listen](../outputs/blue_suede_shoes_full_muq_depth8/mix.wav) | [Listen](../outputs/blue_suede_shoes_full_muq_depth8/depth_08.wav) |

Each folder also contains `depth_01.wav` through `depth_07.wav`, plus eight full
native attention-selection traces. Per-depth normalization gains, model
revisions, selection rules, and stage boundaries are recorded in the graph.

Synthetic event labels are exact. Musical labels are estimated opening,
contrast, and return regions derived from spectral changes and similarity;
they are not verified verse/chorus boundaries. No song lyrics are included.
The renderer supports timestamped text supplied in `annotations.lyrics`.

The guard and quiet monitor operate at each hop. Unlike the earlier two-layer
mix, multi-hop returns near final output time are not separately attenuated.
The progressive mix crossfades over 80 ms when changing depth. The video shows
the incoming depth during that short transition.

The short MuQ fixture has zero measured native-attention error at all eight
layers. The verification script reconstructs every depth bus from the saved
schedules and compares it with the exported WAV. Run:

```sh
.venv/bin/python -m tools.validate_deep_examples
```

Results are recorded in `data/deep-validation.json`. See
[Prototype usage](prototype.md#eight-layer-muq-experiment) for reproduction.
