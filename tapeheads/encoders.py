"""Encoder registry and first-layer adapters with explicit time mappings."""

import dataclasses
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import transformers

from tapeheads import io
from tapeheads import trace_types


@dataclasses.dataclass(frozen=True)
class EncoderSpec:
    """An encoder's checkpoint, sample rate, and extraction family."""

    repository: str
    revision: str
    sample_rate: int
    family: str
    description: str


ENCODERS = {
    'wav2vec2': EncoderSpec(
        'facebook/wav2vec2-base',
        '0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8',
        16000,
        'waveform',
        'Pretrained speech encoder; full-context temporal attention.',
    ),
    'hubert': EncoderSpec(
        'facebook/hubert-base-ls960',
        'dba3bb02fda4248b6e082697eee756de8fe8aa8a',
        16000,
        'waveform',
        'HuBERT speech encoder; full-context temporal attention.',
    ),
    'muq': EncoderSpec(
        'OpenMuQ/MuQ-large-msd-iter',
        '0562a57814f6f8bbd9fdea0a25921a2fce1a841a',
        24000,
        'muq',
        'Music Conformer; requires optional MuQ dependencies.',
    ),
    'clap': EncoderSpec(
        'laion/clap-htsat-unfused',
        '8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a',
        48000,
        'clap',
        'HTSAT audio branch; 10-second contexts, windowed patch attention.',
    ),
}


@dataclasses.dataclass
class AttentionView:
    """Time-indexed attention shared by selection and playback.

    Attributes:
        sample_rate: Analysis samples per second.
        edges: Source time-bin edges in samples.
        queries: Query anchors in samples.
        heads: Number of native attention heads.
        metadata: Checkpoint and projection provenance.
        rows: Callable yielding query index, heads-by-keys probabilities, and
            the valid context-key mask. Probabilities are never renormalized
            after exclusion. For patch models, metadata explains projection.
    """

    sample_rate: int
    edges: np.ndarray
    queries: np.ndarray
    heads: int
    metadata: trace_types.JsonObject
    rows: Callable[[], Iterator[tuple[int, np.ndarray, np.ndarray]]]


class AttentionCaptureComplete(Exception):
    """Internal control flow for stopping immediately before attention."""


def capture_inputs(
    model: torch.nn.Module,
    module: torch.nn.Module,
    inputs: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Run the real frontend and stop before the requested attention module."""
    captured = {}

    def capture(
        unused_module: torch.nn.Module,
        arguments: tuple[Any, ...],
        keyword_arguments: dict[str, Any],
    ) -> None:
        captured['args'] = arguments
        captured['kwargs'] = keyword_arguments
        raise AttentionCaptureComplete()

    hook = module.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with torch.inference_mode():
            model(**inputs)
    except AttentionCaptureComplete:
        pass
    finally:
        hook.remove()
    if not captured:
        raise RuntimeError('Requested attention module was never reached')
    return captured['args'], captured['kwargs']


def metadata(
    encoder_spec: EncoderSpec, model, adapter: str
) -> trace_types.JsonObject:
    """Return immutable checkpoint and numerical implementation metadata."""
    revision = getattr(getattr(model, 'config', None), '_commit_hash', None)
    return {
        'repository': encoder_spec.repository,
        'revision': revision or encoder_spec.revision,
        'adapter': adapter,
        'layer_index': 0,
        'dtype': 'float32',
        'torch_version': torch.__version__,
        'transformers_version': transformers.__version__,
    }


def waveform_view(
    name: str,
    audio: np.ndarray,
    encoder_spec: EncoderSpec,
    block_size: int,
    verify: bool,
    layer_index: int = 0,
) -> AttentionView:
    """Extract wav2vec/HuBERT attention in query blocks against all keys."""
    classes = {
        'wav2vec2': transformers.Wav2Vec2Model,
        'hubert': transformers.HubertModel,
    }
    options = dict(
        revision=encoder_spec.revision,
        cache_dir=io.ROOT / '.cache' / 'huggingface',
    )
    extractor = transformers.Wav2Vec2FeatureExtractor.from_pretrained(
        encoder_spec.repository, **options
    )
    model, loading = classes[name].from_pretrained(
        encoder_spec.repository,
        attn_implementation='eager',
        output_loading_info=True,
        **options,
    )
    if loading['missing_keys'] or loading.get('mismatched_keys'):
        raise RuntimeError('Checkpoint did not fully initialize the encoder')
    model.eval()
    inputs = extractor(
        audio, sampling_rate=encoder_spec.sample_rate, return_tensors='pt'
    )
    module = model.encoder.layers[layer_index].attention
    arguments, keyword_arguments = capture_inputs(model, module, inputs)
    hidden = keyword_arguments.get(
        'hidden_states', arguments[0] if arguments else None
    )
    if hidden is None:
        raise ValueError('Attention hook did not receive hidden states')
    if model.config.do_stable_layer_norm:
        raise ValueError(
            'Stable-layer-norm checkpoints need adapter validation'
        )
    with torch.inference_mode():
        query = module.q_proj(hidden) * module.scaling
        key = module.k_proj(hidden)
        frame_count = hidden.shape[1]
        query = query.view(1, frame_count, module.num_heads, module.head_dim)
        key = key.view(1, frame_count, module.num_heads, module.head_dim)
        query = query[0].transpose(0, 1).contiguous()
        key = key[0].transpose(0, 1).contiguous()
    stride, receptive_field = 1, 1
    for kernel, step in zip(model.config.conv_kernel, model.config.conv_stride):
        receptive_field += (kernel - 1) * stride
        stride *= step
    queries = (
        np.arange(frame_count, dtype=np.int64) * stride + receptive_field // 2
    )
    edges = np.r_[queries - stride // 2, queries[-1] + stride // 2]
    edges = np.clip(edges, 0, len(audio))
    details = metadata(encoder_spec, model, 'waveform-qk/2')
    details['layer_index'] = layer_index
    details['preceding_layers'] = (
        'native eager forward, including residual and feedforward'
    )
    details.update(
        {
            'context': 'full_input',
            'checkpoint_missing_keys': loading['missing_keys'],
            'token_layout': 'temporal',
            'hop_samples': stride,
            'frontend_receptive_field_samples': receptive_field,
            'frame_anchor_samples': receptive_field // 2,
            'preprocessing': extractor.to_dict(),
            'query_block_size': block_size,
        }
    )
    if verify:
        if len(audio) > 5 * encoder_spec.sample_rate:
            raise ValueError('--verify-attention is limited to five seconds')
        with torch.inference_mode():
            reference = model(**inputs, output_attentions=True).attentions[
                layer_index
            ][0]
            error = 0.0
            for start in range(0, frame_count, block_size):
                actual = torch.softmax(
                    query[:, start : start + block_size]
                    @ key.transpose(-1, -2),
                    dim=-1,
                )
                error = max(
                    error,
                    float(
                        (actual - reference[:, start : start + block_size])
                        .abs()
                        .max()
                    ),
                )
        if error > 2e-6:
            raise AssertionError(f'Native attention mismatch: {error}')
        details['native_attention_max_error'] = error
        del reference
    del model, hidden, inputs, arguments, keyword_arguments

    def rows():
        valid = np.ones(frame_count, dtype=bool)
        with torch.inference_mode():
            for start in range(0, frame_count, block_size):
                weights = torch.softmax(
                    query[:, start : start + block_size]
                    @ key.transpose(-1, -2),
                    dim=-1,
                ).numpy()
                if not np.allclose(weights.sum(-1), 1, atol=2e-6):
                    raise AssertionError('Attention rows do not sum to one')
                for offset in range(weights.shape[1]):
                    yield start + offset, weights[:, offset], valid

    return AttentionView(
        encoder_spec.sample_rate,
        edges,
        queries,
        module.num_heads,
        details,
        rows,
    )


def create_view(
    name: str,
    audio: np.ndarray,
    block_size: int = 128,
    verify: bool = False,
    layer_index: int = 0,
) -> AttentionView:
    """Load the requested adapter without coupling downstream code to it."""
    if not 0 <= layer_index < 8:
        raise ValueError('Layer index must be between zero and seven')
    if name not in ENCODERS:
        raise ValueError(f'Unknown encoder: {name}')
    encoder_spec = ENCODERS[name]
    if encoder_spec.family == 'waveform':
        return waveform_view(
            name, audio, encoder_spec, block_size, verify, layer_index
        )
    if encoder_spec.family == 'clap':
        if layer_index:
            raise ValueError(
                'Recursive CLAP requires a validated shifted-window time mapping'
            )
        from tapeheads import clap_adapter  # Optional adapter dependencies.

        return clap_adapter.create_view(audio, encoder_spec, verify)
    if encoder_spec.family == 'muq':
        from tapeheads import muq_adapter  # Optional adapter dependencies.

        return muq_adapter.create_view(
            audio, encoder_spec, block_size, verify, layer_index
        )
    raise ValueError(f'No adapter for {encoder_spec.family}')
