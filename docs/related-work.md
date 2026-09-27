# Related work and references

Research checked September 18, 2026. This is a targeted survey, not an exhaustive
novelty claim.

## Closest precedents

| Work | Relevant idea | Relationship to Tapeheads |
| --- | --- | --- |
| [Visualizing Music Self-Attention](https://openreview.net/pdf?id=ryfxVNEajm) and the [Music Transformer viewer](https://magenta.github.io/music-transformer-visualization/performance.html) | Animated music attention, layer/head selection, threshold controls, and playback | Close visual precedent; uses symbolic music rather than waveform spans selected by an audio encoder |
| [MusicViz, ISMIR 2025](https://ismir2025program.ismir.net/lbd_479.html) | Weighted attention arcs over piano-roll and staff notation with time navigation | Useful for readable time-linked interaction; Tapeheads focuses on source-audio playback |
| [Understanding Self-Attention of Self-Supervised Audio Transformers, Interspeech 2020](https://www.isca-archive.org/interspeech_2020/yang20i_interspeech.html) | Categorizes audio attention and provides multi-head visualization and importance analysis | Strong precedent for inspecting audio heads; helps motivate distinguishing different attention patterns |
| [BertViz](https://github.com/jessevig/bertviz) | Head, model, and neuron views of transformer attention | Useful reference for focus and navigation; text-token views need adaptation to continuous audio |

These sources establish substantial prior work in attention visualization and
music playback. This survey did not identify an exact match for thresholded,
continuous source-audio tape playback across every first-layer head of wav2vec
and MuQ. That is a proposed emphasis, not evidence that the concept is new.

## Models and implementation references

- [wav2vec 2.0 paper](https://arxiv.org/abs/2006.11477): the speech representation
  learning baseline.
- [Transformers wav2vec documentation](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2):
  model configuration and attention outputs.
- [MuQ repository](https://github.com/tencent-ailab/MuQ): released music model,
  inference guidance, and licensing.
- [MuQ paper](https://arxiv.org/abs/2501.01108): music representation learning
  with mel residual vector quantization.
- [MuQ wrapper source](https://github.com/tencent-ailab/MuQ/blob/main/src/muq/muq/muq.py):
  the current public API exposes hidden states, requiring an attention adapter.
- [MuQ subsampling source](https://github.com/tencent-ailab/MuQ/blob/main/src/muq/muq/modules/conv.py):
  derive frame mapping from the actual frontend.

## Local inspiration

Read `../treemusic/README.md`, `HANDOFF.md`, `tools/render_video.py`, and
`viz/ikeda.html`. The strongest transferable idea is that one saved trace drives
multiple synchronized presentations. Its renderer also demonstrates that
streaming frames into FFmpeg avoids large intermediate image collections.

Tapeheads should preserve that relationship between data and presentation while
using a quieter layout: shared waveform, aligned head lanes, sparse labels,
and clear source selection. Paik and Cage remain user-supplied artistic
references; this pass did not research or assert a lineage to particular works.
