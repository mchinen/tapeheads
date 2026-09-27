"""Project CLAP's first Swin block from frequency patches onto source time."""

import numpy as np
import torch
import transformers

from tapeheads import encoders
from tapeheads import io


def project_windows(
    probabilities: np.ndarray, grid_size: int = 64, window_size: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Sum key-frequency mass and average query-frequency rows.

    CLAP unfolds four consecutive time stripes into the image's vertical axis.
    A 64-by-64 patch image therefore maps to 256 time bins and 16 frequency
    bins. Local window membership remains a hard context mask.

    Args:
        probabilities: Native windows-by-heads-by-queries-by-keys weights.
        grid_size: Number of patch rows/columns for the pinned checkpoint.
        window_size: Native first-block window size in patches.

    Returns:
        Heads-by-time-by-time weights and a time-by-time context mask.
    """
    if grid_size != 64 or window_size != 8:
        raise ValueError('Only the pinned CLAP patch layout is supported')
    heads = probabilities.shape[1]
    projected = np.zeros((heads, 256, 256), dtype=np.float32)
    valid = np.zeros((256, 256), dtype=bool)
    query_counts = np.zeros(256, dtype=np.int64)
    index = 0
    for top in range(0, grid_size, window_size):
        for left in range(0, grid_size, window_size):
            times = np.array(
                [
                    (row // 16) * 64 + column
                    for row in range(top, top + window_size)
                    for column in range(left, left + window_size)
                ]
            )
            weights = probabilities[index]
            for query_index, query_time in enumerate(times):
                query_counts[query_time] += 1
                for key_time in np.unique(times):
                    projected[:, query_time, key_time] += weights[
                        :, query_index, times == key_time
                    ].sum(axis=-1)
                    valid[query_time, key_time] = True
            index += 1
    projected /= query_counts[None, :, None]
    if not np.allclose(projected.sum(-1), 1, atol=2e-6):
        raise AssertionError('Frequency projection lost attention mass')
    return projected, valid


def create_view(
    audio: np.ndarray, encoder_spec: encoders.EncoderSpec, verify: bool = False
) -> encoders.AttentionView:
    """Load the audio branch and expose first-block, local-window attention."""
    options = dict(
        revision=encoder_spec.revision,
        cache_dir=io.ROOT / '.cache' / 'huggingface',
    )
    extractor = transformers.ClapFeatureExtractor.from_pretrained(
        encoder_spec.repository, **options
    )
    full_model, loading = transformers.ClapModel.from_pretrained(
        encoder_spec.repository, output_loading_info=True, **options
    )
    if loading['missing_keys'] or loading.get('mismatched_keys'):
        raise RuntimeError('CLAP checkpoint did not fully initialize the model')
    model = full_model.audio_model.eval()
    del full_model
    encoder = model.audio_encoder
    block = encoder.layers[0].blocks[0]
    module = block.attention.self
    if (
        encoder.enable_fusion
        or block.shift_size != 0
        or tuple(encoder.patch_embed.grid_size) != (64, 64)
    ):
        raise ValueError('CLAP checkpoint has an unsupported patch layout')
    chunk_samples = extractor.nb_max_samples
    # 1001 centered STFT frames are interpolated to 1024 time coordinates.
    feature_frames = chunk_samples // extractor.hop_length + 1
    centers = (
        (np.arange(256) * 4 + 1.5)
        * (feature_frames - 1)
        / 1023
        * extractor.hop_length
    )
    local_edges = np.r_[
        0, np.rint((centers[:-1] + centers[1:]) / 2), chunk_samples
    ].astype(np.int64)
    plans = []
    all_edges = [0]
    all_queries = []
    for start in range(0, len(audio), chunk_samples):
        length = min(chunk_samples, len(audio) - start)
        count = int(np.sum(centers < length))
        if count == 0:
            # Ignore trailing audio shorter than one patch center.
            continue
        edges = local_edges[: count + 1].copy()
        edges[-1] = length
        queries = np.rint(centers[:count]).astype(np.int64) + start
        plans.append((start, length, count, len(all_queries)))
        all_edges.extend((edges[1:] + start).tolist())
        all_queries.extend(queries.tolist())
    details = encoders.metadata(
        encoder_spec, model, 'clap-first-window-time-projection/1'
    )
    details.update(
        {
            'context': 'independent_10_second_chunks_with_local_swin_windows',
            'checkpoint_missing_keys': loading['missing_keys'],
            'chunk_samples': chunk_samples,
            'padding': 'zeros, no repeat padding, no random crop',
            'token_layout': 'time_frequency_patches',
            'native_layer': 'audio_encoder.layers.0.blocks.0.attention.self',
            'projection': 'sum key frequency, mean query frequency; no renormalizing',
            'mapping': 'patch-center Voronoi time bins after mel interpolation',
            'native_window_patches': [8, 8],
            'native_patch_grid': [64, 64],
            'preprocessing': extractor.to_dict(),
            'context_key_count': 'local time keys only, before temporal exclusion',
            'validation': 'native first-block probabilities, projected mass checked',
        }
    )

    def rows():
        total = len(all_queries)
        with torch.inference_mode():
            for start, length, count, offset in plans:
                chunk = np.zeros(chunk_samples, dtype=np.float32)
                chunk[:length] = audio[start : start + length]
                inputs = extractor(
                    chunk,
                    sampling_rate=encoder_spec.sample_rate,
                    return_tensors='pt',
                    padding='pad',
                    truncation='rand_trunc',
                )
                arguments, keyword_arguments = encoders.capture_inputs(
                    model, module, inputs
                )
                if len(arguments) >= 4:
                    arguments = (*arguments[:3], True, *arguments[4:])
                else:
                    keyword_arguments['output_attentions'] = True
                native = module(*arguments, **keyword_arguments)[1].numpy()
                projected, visible = project_windows(native)
                for query in range(count):
                    probabilities = np.zeros(
                        (module.num_attention_heads, total), dtype=np.float32
                    )
                    valid = np.zeros(total, dtype=bool)
                    probabilities[:, offset : offset + count] = projected[
                        :, query, :count
                    ]
                    valid[offset : offset + count] = visible[query, :count]
                    yield offset + query, probabilities, valid

    return encoders.AttentionView(
        encoder_spec.sample_rate,
        np.array(all_edges),
        np.array(all_queries),
        module.num_attention_heads,
        details,
        rows,
    )
