"""First-layer MuQ adapter preserving rotary/relative positional attention."""

import numpy as np
import torch

from tapeheads import encoders
from tapeheads import io


def create_view(
    audio: np.ndarray,
    encoder_spec: encoders.EncoderSpec,
    block_size: int,
    verify: bool = False,
    layer_index: int = 0,
) -> encoders.AttentionView:
    """Capture MuQ's real first attention input and compute row blocks."""
    try:
        from muq import MuQ  # MuQ is an optional encoder dependency.
    except ImportError as error:
        raise ImportError('Install MuQ with tools/setup_muq.sh') from error
    model = MuQ.from_pretrained(
        encoder_spec.repository,
        revision=encoder_spec.revision,
        cache_dir=io.ROOT / '.cache' / 'huggingface',
    ).eval()
    module = model.model.conformer.layers[layer_index].self_attn
    arguments, keyword_arguments = encoders.capture_inputs(
        model, module, {'x': torch.from_numpy(audio).unsqueeze(0)}
    )
    hidden = keyword_arguments['hidden_states']
    relative = keyword_arguments['relative_position_embeddings']
    frame_count = hidden.shape[1]
    if module.position_embeddings_type == 'relative':
        raise ValueError(
            'Relative MuQ embeddings need a separate blocked adapter'
        )
    with torch.inference_mode():
        query_key = hidden
        if module.position_embeddings_type == 'rotary':
            query_key = module._apply_rotary_embedding(hidden, relative)
        query = (
            module.linear_q(query_key)
            .view(1, frame_count, module.num_heads, module.head_size)[0]
            .transpose(0, 1)
        )
        key = (
            module.linear_k(query_key)
            .view(1, frame_count, module.num_heads, module.head_size)[0]
            .transpose(0, 1)
        )
    hop = model.config.hop_length * (100 // model.config.label_rate)
    queries = np.arange(frame_count, dtype=np.int64) * hop
    edges = np.r_[0, queries[1:] - hop // 2, queries[-1] + hop // 2]
    edges = np.clip(edges, 0, len(audio))
    details = encoders.metadata(encoder_spec, model, 'muq-rotary-qk/2')
    details['layer_index'] = layer_index
    details['preceding_layers'] = (
        'native Conformer forward including convolutions'
    )
    details.update(
        {
            'context': 'full_input',
            'token_layout': 'temporal',
            'hop_samples': hop,
            'frame_anchor_samples': 0,
            'mapping': 'centered STFT, padded conv stride; midpoint time bins',
            'preprocessing': 'MuQ native mel and stored training mean/std',
            'position_embeddings': module.position_embeddings_type,
            'query_block_size': block_size,
        }
    )
    if verify:
        if len(audio) > 5 * encoder_spec.sample_rate:
            raise ValueError('--verify-attention is limited to five seconds')
        with torch.inference_mode():
            reference = module(*arguments, **keyword_arguments)[1][0]
            actual = torch.softmax(
                (query @ key.transpose(-1, -2)) / module.head_size**0.5, dim=-1
            )
            error = float((reference - actual).abs().max())
        if error > 2e-6:
            raise AssertionError(f'MuQ attention mismatch: {error}')
        details['native_attention_max_error'] = error
        del reference, actual
    head_size, heads = module.head_size, module.num_heads
    del model, hidden, query_key, arguments, keyword_arguments, relative

    def rows():
        valid = np.ones(frame_count, dtype=bool)
        with torch.inference_mode():
            for start in range(0, frame_count, block_size):
                weights = torch.softmax(
                    (
                        query[:, start : start + block_size]
                        @ key.transpose(-1, -2)
                    )
                    / head_size**0.5,
                    dim=-1,
                ).numpy()
                if not np.allclose(weights.sum(-1), 1, atol=2e-6):
                    raise AssertionError('MuQ attention rows do not sum to one')
                for offset in range(weights.shape[1]):
                    yield start + offset, weights[:, offset], valid

    return encoders.AttentionView(
        encoder_spec.sample_rate, edges, queries, heads, details, rows
    )
