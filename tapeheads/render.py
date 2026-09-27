"""Render the shared trace as a restrained waveform-and-tape-head film."""

import bisect
import math
import pathlib
import subprocess

import numpy as np
import soundfile
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from tapeheads import io
from tapeheads import trace_types

BACKGROUND = '#f5f4ef'
INK = '#202a2a'
MUTED = '#77817d'
RULE = '#dcded6'
ACCENT = '#287a6b'


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a reproducible local sans-serif font."""
    for path in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    ):
        if pathlib.Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


class Plate:
    """Draw source bins, native head lanes, and the actual playback positions."""

    def __init__(
        self,
        trace: trace_types.JsonObject,
        source: np.ndarray,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        self.trace = trace
        self.width = width
        self.height = height
        self.scale = width / 1920
        self.source = source
        self.sample_rate = trace['source']['sample_rate_hz']
        self.samples = len(source)
        self.left = 180
        self.right = 1770
        self.top = 280
        self.lane_height = 560 / len(trace['heads'])
        self.wave_y = 940
        self.font_small = font(round(17 * self.scale))
        self.font_label = font(round(20 * self.scale))
        self.font_title = font(round(42 * self.scale))
        self.font_brand = font(round(19 * self.scale))
        self.query_times = [
            [item['query_sample'] for item in head['selections']]
            for head in trace['heads']
        ]
        self.playback_times = [
            [item['output_start'] for item in head['playback']]
            for head in trace['heads']
        ]
        edges = np.linspace(0, len(source), 1101).astype(int)
        self.peaks = np.array(
            [
                np.max(np.abs(source[start : max(start + 1, end)]))
                for start, end in zip(edges[:-1], edges[1:])
            ]
        )
        maximum = self.peaks.max()
        if maximum:
            self.peaks /= maximum
        self.base = self._background()

    def point(self, x: float, y: float) -> tuple[int, int]:
        """Scale a point from the 1920-by-1080 reference layout."""
        return round(x * self.scale), round(y * self.height / 1080)

    def x(self, sample: float) -> float:
        """Map an audio sample to the shared source axis."""
        return self.left + sample / self.samples * (self.right - self.left)

    def line(
        self,
        draw: ImageDraw.ImageDraw,
        points: list[tuple[float, float]],
        color: str = RULE,
        width: int = 1,
    ) -> None:
        """Draw a scaled polyline."""
        draw.line(
            [self.point(x, y) for x, y in points],
            fill=color,
            width=max(1, round(width * self.scale)),
        )

    def text(
        self,
        draw: ImageDraw.ImageDraw,
        xy: tuple[float, float],
        text: str,
        color: str = MUTED,
        face: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None,
    ) -> None:
        """Draw a label in reference-layout coordinates."""
        draw.text(
            self.point(*xy), text, font=face or self.font_small, fill=color
        )

    def _background(self) -> Image.Image:
        canvas = Image.new('RGB', (self.width, self.height), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        self.text(draw, (72, 48), 'Tapeheads', INK, self.font_brand)
        self.text(
            draw,
            (72, 113),
            self.trace['source'].get('title', 'Audio study'),
            INK,
            self.font_title,
        )
        model = self.trace['encoder']
        self.text(
            draw,
            (74, 182),
            f'{model}  /  layer 01  /  {len(self.trace["heads"]):02d} heads',
        )
        context = self.trace['model']['context']
        context_label = (
            'Local windows · 10 s contexts'
            if 'chunks' in context
            else 'Full input'
        )
        self.text(draw, (1330, 56), context_label)
        guard = self.trace['selection']['guard_seconds'] * 1000
        self.text(
            draw,
            (1330, 88),
            (
                f'NEAR ±{guard:.0f} ms · QUIET {self.trace["near_monitor"]["relative_gain"]:.0%}'
                if 'near_monitor' in self.trace
                else f'Exclude ±{guard:.0f} ms · 1× playback'
            ),
        )
        self.line(draw, [(72, 227), (1848, 227)])
        self.text(draw, (74, 250), 'HEAD')
        self.text(draw, (self.left, 250), 'Source position')
        for index in range(len(self.trace['heads'])):
            y = self.top + index * self.lane_height + 28
            self.text(
                draw, (82, y - 8), f'{index + 1:02d}', INK, self.font_label
            )
            self.line(draw, [(self.left, y + 15), (self.right, y + 15)])
        self.text(draw, (74, 882), 'Source')
        self.line(draw, [(self.left, self.wave_y), (self.right, self.wave_y)])
        for index, peak in enumerate(self.peaks):
            x = self.left + index / len(self.peaks) * (self.right - self.left)
            self.line(
                draw,
                [(x, self.wave_y - 32 * peak), (x, self.wave_y + 32 * peak)],
                '#a9b2a9',
            )
        duration = self.samples / self.sample_rate
        for tick in range(7):
            seconds = duration * tick / 6
            x = self.left + tick / 6 * (self.right - self.left)
            self.text(draw, (x - 12, 985), f'{seconds:.1f}s')
        self.text(draw, (74, 1040), 'Attention / source audio')
        self.text(
            draw, (1135, 1040), 'Bracket: selected span     Dot: playback'
        )
        return canvas

    def frame(self, seconds: float) -> Image.Image:
        """Draw one deterministic frame using the saved source/playback clocks."""
        canvas = self.base.copy()
        draw = ImageDraw.Draw(canvas)
        sample = min(self.samples - 1, round(seconds * self.sample_rate))
        query_x = self.x(sample)
        guard = self.trace['selection']['guard_samples']
        guard_left = self.x(max(0, sample - guard))
        guard_right = self.x(min(self.samples, sample + guard))
        overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            (
                self.point(guard_left, self.top - 3),
                self.point(guard_right, 978),
            ),
            fill=(70, 85, 75, 13),
        )
        canvas = Image.alpha_composite(canvas.convert('RGBA'), overlay).convert(
            'RGB'
        )
        draw = ImageDraw.Draw(canvas)
        self.line(draw, [(query_x, self.top - 3), (query_x, 978)], '#98a99d')
        for index, head in enumerate(self.trace['heads']):
            position = bisect.bisect_right(self.query_times[index], sample) - 1
            if position < 0:
                continue
            item = head['selections'][position]
            y = self.top + index * self.lane_height + 28
            near = item.get('near_attention')
            if near and near['active']:
                self.line(
                    draw,
                    [
                        (guard_left, y + 6),
                        (guard_left, y + 15),
                        (guard_right, y + 15),
                        (guard_right, y + 6),
                    ],
                    '#bd914e',
                    3,
                )
            blocks = item.get(
                'selected_blocks',
                [item['selected']] if item['selected'] else [],
            )
            for selected in blocks:
                if 'playback_start' in selected:
                    self.line(
                        draw,
                        [
                            (self.x(selected['playback_start']), y + 18),
                            (self.x(selected['playback_end']), y + 18),
                        ],
                        '#bdc9c0',
                        2,
                    )
                left = self.x(selected['source_start'])
                right = self.x(selected['source_end'])
                middle = (left + right) / 2
                color = ACCENT if index == 0 else '#70897d'
                self.line(
                    draw,
                    [
                        (left, y + 6),
                        (left, y + 15),
                        (right, y + 15),
                        (right, y + 6),
                    ],
                    color,
                    2,
                )
                points = []
                for step in range(33):
                    fraction = step / 32
                    # Cubic arm remains inside its head's lane.
                    x = (
                        (1 - fraction) ** 3 * query_x
                        + 3 * (1 - fraction) ** 2 * fraction * query_x
                        + 3 * (1 - fraction) * fraction**2 * middle
                        + fraction**3 * middle
                    )
                    curve_y = (
                        (1 - fraction) ** 3 * (y - 5)
                        + 3 * (1 - fraction) ** 2 * fraction * (y - 18)
                        + 3 * (1 - fraction) * fraction**2 * (y - 18)
                        + fraction**3 * (y + 15)
                    )
                    points.append((x, curve_y))
                self.line(draw, points, color, 2)
                draw.ellipse(
                    [
                        self.point(query_x - 3, y - 8),
                        self.point(query_x + 3, y - 2),
                    ],
                    fill=color,
                )
            for voice in head.get('voices', [{'playback': head['playback']}]):
                segments = voice['playback']
                starts = [segment['output_start'] for segment in segments]
                play_index = bisect.bisect_right(starts, sample) - 1
                if play_index < 0:
                    continue
                segment = segments[play_index]
                if sample < segment['output_end']:
                    source_sample = (
                        segment['source_start']
                        + sample
                        - segment['output_start']
                    )
                    x = self.x(source_sample)
                    draw.ellipse(
                        [self.point(x - 4, y + 11), self.point(x + 4, y + 19)],
                        fill=INK,
                    )
        self.text(
            draw,
            (1510, 177),
            f'{seconds:06.2f} / {self.samples / self.sample_rate:06.2f}s',
            INK,
            self.font_label,
        )
        return canvas


def render(
    trace_path: trace_types.PathLike,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
) -> pathlib.Path:
    """Stream frames into FFmpeg and save a still for visual inspection."""
    if width < 640 or height < 360 or width % 2 or height % 2 or fps < 1:
        raise ValueError('Use even dimensions >= 640×360 and a positive FPS')
    trace_path = pathlib.Path(trace_path)
    trace = io.read_json(trace_path)
    source, sample_rate = soundfile.read(
        trace_path.parent / trace['source']['path'], dtype='float32'
    )
    if trace['format'] == 'tapeheads/4':
        from tapeheads import deep_render

        plate = deep_render.DeepPlate(trace, source, width, height)
    elif trace['format'] == 'tapeheads/3':
        from tapeheads import recursive_render

        plate = recursive_render.RecursivePlate(trace, source, width, height)
    else:
        plate = Plate(trace, source, width, height)
    duration = len(source) / sample_rate
    if trace['format'] == 'tapeheads/4':
        preview_time = duration * 0.94
    elif trace['format'] == 'tapeheads/3':
        preview_time = min(4.0, duration / 2)
    else:
        queries = trace['frames']['query_samples']
        activity = [
            sum(
                head['selections'][index]['selected'] is not None
                for head in trace['heads']
            )
            for index in range(len(queries))
        ]
        preview_index = max(
            range(len(queries)),
            key=lambda index: (
                activity[index],
                -abs(queries[index] / sample_rate - 4),
            ),
        )
        preview_time = queries[preview_index] / sample_rate
    plate.frame(preview_time).save(trace_path.parent / 'preview.png')
    output = trace_path.parent / 'tapeheads.mp4'
    command = [
        io.executable('ffmpeg'),
        '-v',
        'error',
        '-y',
        '-f',
        'rawvideo',
        '-pixel_format',
        'rgb24',
        '-video_size',
        f'{width}x{height}',
        '-framerate',
        str(fps),
        '-i',
        '-',
        '-i',
        str(trace_path.parent / trace['audio']['mix']),
        '-c:v',
        'libx264',
        '-preset',
        'fast',
        '-crf',
        '20',
        '-pix_fmt',
        'yuv420p',
        '-c:a',
        'aac',
        '-b:a',
        '192k',
        '-t',
        str(duration),
        '-movflags',
        '+faststart',
        str(output),
    ]
    with (trace_path.parent / 'render.log').open('w') as render_log:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=render_log
        )
        try:
            for index in range(math.ceil(duration * fps)):
                process.stdin.write(plate.frame(index / fps).tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('FFmpeg failed; inspect render.log')
        except BaseException:
            process.kill()
            process.wait()
            raise
    trace['rendering'] = {
        **trace['rendering'],
        'width': width,
        'height': height,
        'fps': fps,
        'video': output.name,
        'preview': 'preview.png',
        'preview_seconds': preview_time,
        'font_family': list(plate.font_small.getname())
        if isinstance(plate.font_small, ImageFont.FreeTypeFont)
        else ['Pillow default', 'Regular'],
        'style': 'deep-dark-field/2'
        if trace['format'] == 'tapeheads/4'
        else (
            'recursive-dark-field/3'
            if trace['format'] == 'tapeheads/3'
            else 'head-lanes/2'
        ),
    }
    io.write_json(trace_path, trace)
    print(f'Saved {output}', flush=True)
    return output
