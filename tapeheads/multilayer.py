"""Capture temporal encoder Q/K tensors in one bounded-memory forward pass."""

import types
from collections.abc import Callable
from collections.abc import Iterator

import numpy as np
import torch
import torch.nn.functional as functional
import transformers

from tapeheads import encoders
from tapeheads import io


def make_rows(
    query: torch.Tensor, key: torch.Tensor, block_size: int
) -> Callable[[], Iterator[tuple[int, np.ndarray, np.ndarray]]]:
    """Yield blocked query probabilities normalized over every input key."""

    def rows():
        valid = np.ones(query.shape[1], dtype=bool)
        with torch.inference_mode():
            for start in range(0, query.shape[1], block_size):
                weights = (
                    torch.softmax(
                        query[:, start : start + block_size]
                        @ key.transpose(-1, -2),
                        dim=-1,
                        dtype=torch.float64,
                    )
                    .float()
                    .numpy()
                )
                if not np.allclose(weights.sum(-1), 1, atol=2e-6):
                    error = float(np.max(np.abs(weights.sum(-1) - 1)))
                    raise AssertionError(
                        f'Attention row mass changed: maximum error {error}; '
                        f'finite={np.isfinite(weights).all()}; query={start}'
                    )
                for offset in range(weights.shape[1]):
                    yield start + offset, weights[:, offset], valid

    return rows


def create_views(
    name: str, audio: np.ndarray, layers: int = 8, block_size: int = 128
) -> list[encoders.AttentionView]:
    """Preserve native blocks, replacing attention products with CPU SDPA.

    Q/K capture is exact for the pinned temporal layouts. SDPA avoids retaining
    the full preceding-layer attention matrix; exported probabilities still
    use explicit softmax over all keys. Short native comparisons gate use.
    """
    if name not in ('wav2vec2', 'hubert', 'muq') or not 1 <= layers <= 8:
        raise ValueError('Use 1–8 layers of a supported temporal encoder')
    encoder_spec = encoders.ENCODERS[name]
    options = {
        'revision': encoder_spec.revision,
        'cache_dir': io.ROOT / '.cache/huggingface',
    }
    if name == 'muq':
        from muq import MuQ

        model = MuQ.from_pretrained(encoder_spec.repository, **options).eval()
        modules = [
            layer.self_attn for layer in model.model.conformer.layers[:layers]
        ]
        inputs = {'x': torch.from_numpy(audio).unsqueeze(0)}
        hop = model.config.hop_length * (100 // model.config.label_rate)
        anchor = 0
        preprocessing = 'native MuQ mel and stored mean/std'
    else:
        classes = {
            'wav2vec2': transformers.Wav2Vec2Model,
            'hubert': transformers.HubertModel,
        }
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
            raise RuntimeError(
                'Checkpoint did not fully initialize the encoder'
            )
        model.eval()
        modules = [layer.attention for layer in model.encoder.layers[:layers]]
        inputs = extractor(
            audio, sampling_rate=encoder_spec.sample_rate, return_tensors='pt'
        )
        hop, receptive = 1, 1
        for kernel, step in zip(
            model.config.conv_kernel, model.config.conv_stride
        ):
            receptive += (kernel - 1) * hop
            hop *= step
        anchor = receptive // 2
        preprocessing = extractor.to_dict()
    captures = []
    originals = [module.forward for module in modules]

    def replacement(index):
        def forward(module, hidden_states, *arguments, **keyword_arguments):
            if arguments:
                raise ValueError('Unexpected positional attention arguments')
            if (
                module.training
                or keyword_arguments.get('attention_mask') is not None
            ):
                raise ValueError(
                    'Only unpadded evaluation batches are supported'
                )
            size = module.head_size if name == 'muq' else module.head_dim
            heads = module.num_heads
            batch, frame_count, channels = hidden_states.shape
            if batch != 1:
                raise ValueError('Expected one complete input')
            if name == 'muq':
                if module.position_embeddings_type != 'rotary':
                    raise ValueError('Only pinned rotary MuQ is supported')
                states = module._apply_rotary_embedding(
                    hidden_states,
                    keyword_arguments['relative_position_embeddings'],
                )
                query, key = module.linear_q(states), module.linear_k(states)
                value = module.linear_v(hidden_states)
                projection = module.linear_out
            else:
                if (
                    keyword_arguments.get('key_value_states') is not None
                    or keyword_arguments.get('past_key_value') is not None
                    or keyword_arguments.get('layer_head_mask') is not None
                ):
                    raise ValueError('Unsupported cross attention or head mask')
                query, key = module.q_proj(hidden_states), module.k_proj(
                    hidden_states
                )
                value = module.v_proj(hidden_states)
                projection = module.out_proj
            query, key, value = [
                tensor.view(batch, frame_count, heads, size)
                .transpose(1, 2)
                .contiguous()
                for tensor in (query, key, value)
            ]
            captures.append((query[0] * size**-0.5, key[0]))
            if index == layers - 1:
                raise encoders.AttentionCaptureComplete()
            attended = functional.scaled_dot_product_attention(
                query, key, value, dropout_p=0, is_causal=False
            )
            result = projection(
                attended.transpose(1, 2).reshape(batch, frame_count, channels)
            )
            return (result, None) if name == 'muq' else (result, None, None)

        return forward

    for index, module in enumerate(modules):
        module.forward = types.MethodType(replacement(index), module)
    try:
        with torch.inference_mode():
            model(**inputs)
    except encoders.AttentionCaptureComplete:
        pass
    finally:
        for module, forward in zip(modules, originals):
            module.forward = forward
    if len(captures) != layers:
        raise AssertionError('Not all requested layers were captured')
    frame_count = captures[0][0].shape[1]
    queries = np.arange(frame_count, dtype=np.int64) * hop + anchor
    if name == 'muq':
        edges = np.r_[0, queries[1:] - hop // 2, queries[-1] + hop // 2]
    else:
        edges = np.r_[queries - hop // 2, queries[-1] + hop // 2]
    edges = np.clip(edges, 0, len(audio))
    views = []
    for index, (query, key) in enumerate(captures):
        details = encoders.metadata(
            encoder_spec, model, 'single-pass-sdpa-qk/2'
        )
        details.update(
            {
                'layer_index': index,
                'context': 'full_input',
                'token_layout': 'temporal',
                'hop_samples': hop,
                'frame_anchor_samples': anchor,
                'preprocessing': preprocessing,
                'query_block_size': block_size,
                'preceding_layers': 'native blocks with evaluation SDPA attention products',
                'probabilities': 'explicit blocked softmax, all input keys',
                'softmax_dtype': 'float64, exported as float32',
            }
        )
        views.append(
            encoders.AttentionView(
                encoder_spec.sample_rate,
                edges,
                queries,
                query.shape[0],
                details,
                make_rows(query, key, block_size),
            )
        )
    return views
