"""Command-line entry point for encoder selection, extraction, and rendering."""

import argparse

from tapeheads import deep
from tapeheads import encoders
from tapeheads import pipeline
from tapeheads import recursive
from tapeheads import render
from tapeheads import selection
from tapeheads import validation


def main() -> None:
    """Dispatch an encoder-independent pipeline command."""
    parser = argparse.ArgumentParser(
        description='Sonify audio encoder attention.'
    )
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('encoders', help='List supported encoder adapters.')
    verify = commands.add_parser(
        'verify', help='Validate a trace and its assets.'
    )
    verify.add_argument('trace')
    analyze = commands.add_parser(
        'analyze', help='Save trace, source, and stems.'
    )
    analyze.add_argument('input')
    analyze.add_argument(
        '--encoder', choices=encoders.ENCODERS, default='wav2vec2'
    )
    analyze.add_argument('--output', required=True)
    analyze.add_argument('--start', type=float, default=0)
    analyze.add_argument('--duration', type=float)
    analyze.add_argument('--alpha', type=float)
    analyze.add_argument('--exit-alpha', type=float)
    analyze.add_argument('--guard-ms', type=float, default=100)
    analyze.add_argument('--max-layer-blocks', type=int, default=3)
    analyze.add_argument('--average-window-ms', type=float)
    analyze.add_argument('--min-block-ms', type=float)
    analyze.add_argument('--near-gain', type=float, default=0.15)
    analyze.add_argument(
        '--tracking-policy',
        choices=('current_support', 'overlap_union'),
        default='current_support',
        help='Follow current support, or reproduce historical union playback.',
    )
    analyze.add_argument('--query-block-size', type=int, default=128)
    analyze.add_argument('--threads', type=int, default=4)
    analyze.add_argument('--verify-attention', action='store_true')
    analyze.add_argument('--video', action='store_true')
    analyze.add_argument('--layers', type=int, default=2)
    analyze.add_argument(
        '--recursive',
        action='store_true',
        help='Compose native layers; current-support playback supports up to three.',
    )
    video = commands.add_parser(
        'render', help='Render video from an existing trace.'
    )
    video.add_argument('trace')
    video.add_argument('--width', type=int, default=1920)
    video.add_argument('--height', type=int, default=1080)
    video.add_argument('--fps', type=int, default=30)
    arguments = parser.parse_args()
    if arguments.command == 'encoders':
        for name, encoder_spec in encoders.ENCODERS.items():
            print(
                f'{name:10} {encoder_spec.sample_rate:5} Hz  {encoder_spec.description}'
            )
    elif arguments.command == 'verify':
        print(validation.validate(arguments.trace))
    elif arguments.command == 'analyze':
        if arguments.layers != 2 and not arguments.recursive:
            parser.error('--layers requires --recursive')
        deep_mode = arguments.recursive and arguments.layers != 2
        responsive = arguments.tracking_policy == 'current_support'
        if responsive and arguments.recursive and arguments.layers > 3:
            parser.error('current_support supports at most three layers')
        alpha = (
            arguments.alpha
            if arguments.alpha is not None
            else (1.5 if deep_mode or responsive else 2.0)
        )
        exit_alpha = (
            arguments.exit_alpha
            if arguments.exit_alpha is not None
            else (alpha if responsive else (1.1 if deep_mode else 1.25))
        )
        average_ms = (
            arguments.average_window_ms
            if arguments.average_window_ms is not None
            else (120 if responsive else (350 if deep_mode else 200))
        )
        minimum_ms = (
            arguments.min_block_ms
            if arguments.min_block_ms is not None
            else (120 if responsive else (300 if deep_mode else 120))
        )
        config = selection.SelectionConfig(
            entry_alpha=alpha,
            exit_alpha=min(exit_alpha, alpha),
            guard_seconds=arguments.guard_ms / 1000,
            max_layer_blocks=arguments.max_layer_blocks,
            average_window_seconds=average_ms / 1000,
            minimum_block_seconds=minimum_ms / 1000,
            near_gain=arguments.near_gain,
            tracking_policy=arguments.tracking_policy,
            switch_ratio=1.05 if responsive else 1.2,
        )
        if arguments.recursive:
            if arguments.start:
                parser.error('--recursive currently requires --start 0')
            if arguments.layers != 2 or responsive:
                deep.analyze(
                    arguments.input,
                    arguments.output,
                    arguments.encoder,
                    arguments.layers,
                    arguments.max_layer_blocks,
                    arguments.duration,
                    arguments.verify_attention,
                    config=config,
                    threads=arguments.threads,
                    block_size=arguments.query_block_size,
                )
            else:
                recursive.analyze(
                    arguments.input,
                    arguments.output,
                    arguments.encoder,
                    arguments.duration,
                    config,
                    arguments.threads,
                    arguments.verify_attention,
                )
        else:
            pipeline.analyze(
                arguments.input,
                arguments.output,
                arguments.encoder,
                arguments.start,
                arguments.duration,
                config,
                arguments.query_block_size,
                arguments.threads,
                arguments.verify_attention,
            )
        if arguments.video:
            render.render(f'{arguments.output}/trace.json')
    elif arguments.command == 'render':
        render.render(
            arguments.trace, arguments.width, arguments.height, arguments.fps
        )


if __name__ == '__main__':
    main()
