"""Draw up to eight recursive planes with source-clock section labels."""

import bisect
from typing import NamedTuple

import numpy as np
from PIL import Image
from PIL import ImageDraw

from tapeheads import annotations
from tapeheads import recursive
from tapeheads import recursive_render
from tapeheads import render
from tapeheads import trace_types


class ActiveVoice(NamedTuple):
    """A voice at one layer's playback clock."""

    head_index: int
    voice_id: int
    source_sample: int
    segment: trace_types.PlaybackSegment
    relative_gain: float


class DeepPlate(recursive_render.RecursivePlate):
    """Render a bounded graph view while audio retains every recursive path."""

    def __init__(
        self,
        trace: trace_types.JsonObject,
        source: np.ndarray,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        layer = trace['layers'][min(1, len(trace['layers']) - 1)]
        native = {
            **layer,
            'source': trace['source'],
            'encoder': trace['encoder'],
            'heads': [
                {'selections': [], 'playback': []} for _ in layer['heads']
            ],
        }
        render.Plate.__init__(self, native, source, width, height)
        self.trace = trace
        self.trace['rendering'][
            'visual_policy'
        ] = 'bounded complete routes per root; longer runs win ties; quiet branches hidden; audio includes every branch'
        self.left, self.right = 220, 1780
        self.wave_y = 950
        self.levels = len(trace['layers'])
        self.stage_ends = [stage['end'] for stage in trace['stages']]
        self.bucket_samples = max(1, (self.samples + 1023) // 1024)
        self.lookups = []
        for layer in trace['layers']:
            buckets = {}
            for head, voice, relative_gain, segments in recursive.tracks(layer):
                if head >= 0:
                    for segment in segments:
                        for bucket in range(
                            segment['output_start'] // self.bucket_samples,
                            (segment['output_end'] - 1) // self.bucket_samples
                            + 1,
                        ):
                            buckets.setdefault(bucket, []).append(
                                (head, voice, relative_gain, segment)
                            )
            self.lookups.append(buckets)
        self.base = self.background()

    def lane_y(self, layer: int, head: int) -> float:
        """Give each layer its own band and each native head a small offset."""
        count = len(self.trace['layers'][layer]['heads'])
        band = 560 / self.levels
        return (
            285
            + (self.levels - 1 - layer) * band
            + (head + 0.5) * (band - 16) / count
        )

    def active(self, layer: int, sample: int) -> list[ActiveVoice]:
        """Find all remote voices active at a particular intermediate clock."""
        result: list[ActiveVoice] = []
        for head, voice, relative_gain, segment in self.lookups[layer].get(
            sample // self.bucket_samples, []
        ):
            if segment['output_start'] <= sample < segment['output_end']:
                source = (
                    segment['source_start'] + sample - segment['output_start']
                )
                result.append(
                    ActiveVoice(head, voice, source, segment, relative_gain)
                )
        return result

    def background(self) -> Image.Image:
        """Create dark planes and faint diagonal labels without image assets."""
        canvas = Image.new(
            'RGB', (self.width, self.height), recursive_render.PAPER
        )
        draw = ImageDraw.Draw(canvas)
        self.text(
            draw,
            (64, 42),
            'Tapeheads',
            recursive_render.INK,
            self.font_brand,
        )
        self.text(
            draw,
            (64, 101),
            self.trace['source']['title'],
            recursive_render.INK,
            self.font_title,
        )
        self.text(
            draw,
            (66, 168),
            f'{self.trace["encoder"]} / {self.levels:02d} layers · full input',
            '#91a58e',
        )
        self.line(draw, [(64, 225), (1856, 225)], '#2e3c31')
        band = 560 / self.levels
        for layer in range(self.levels):
            y = 285 + (self.levels - 1 - layer) * band
            self.text(
                draw,
                (66, y),
                f'{layer + 1:02d}',
                recursive_render.INK,
                self.font_label,
            )
            count = len(self.trace['layers'][layer]['heads'])
            for head in range(count):
                self.dot(
                    draw,
                    126 + head % 8 * 10,
                    y + 8 + head // 8 * 10,
                    1.5,
                    '#46583e',
                )
            for x in range(220, 1781, 22):
                for offset in range(0, int(band - 12), 16):
                    self.dot(draw, x, y + offset, 0.6, recursive_render.FAINT)
            self.line(
                draw, [(220, y + band - 10), (1780, y + band - 10)], '#1e2921'
            )
        self.text(
            draw, (66, 897), '00  Input', recursive_render.INK, self.font_label
        )
        for index, peak in enumerate(self.peaks):
            x = self.left + index / len(self.peaks) * (self.right - self.left)
            self.line(
                draw,
                [(x, self.wave_y - peak * 26), (x, self.wave_y + peak * 26)],
                '#576d52',
            )
        for seconds in range(0, int(self.samples / self.sample_rate) + 1, 15):
            sample = seconds * self.sample_rate
            self.line(
                draw, [(self.x(sample), 978), (self.x(sample), 985)], '#7c8a79'
            )
            label_stride = max(
                1,
                int(
                    np.ceil(self.samples / self.sample_rate * 60 / (1560 * 15))
                ),
            )
            if seconds // 15 % label_stride:
                continue
            self.text(
                draw,
                (self.x(sample) - 12, 992),
                f'{seconds // 60}:{seconds % 60:02d}',
                '#7c8a79',
            )
        # Only long sections get static diagonal text; individual beeps appear
        # in the source-clock caption as the cursor passes them.
        for section in self.trace['annotations']['sections']:
            if section['end'] - section['start'] < self.sample_rate:
                continue
            label = annotations.display_label(section['label'])
            tile = Image.new(
                'RGBA', (round(500 * self.scale), round(55 * self.scale))
            )
            ImageDraw.Draw(tile).text(
                (0, 0), label, font=self.font_small, fill=(170, 195, 151, 105)
            )
            tile = tile.rotate(
                24, resample=Image.Resampling.BICUBIC, expand=True
            )
            label_position = self.point(self.x(section['start']), 851)
            canvas.paste(
                tile,
                (label_position[0], label_position[1] - tile.height // 2),
                tile,
            )
        for lyric in self.trace['annotations'].get('lyrics', []):
            tile = Image.new(
                'RGBA', (round(700 * self.scale), round(100 * self.scale))
            )
            ImageDraw.Draw(tile).multiline_text(
                (0, 0),
                lyric['text'],
                font=self.font_small,
                fill=(175, 191, 162, 85),
            )
            tile = tile.rotate(
                18, resample=Image.Resampling.BICUBIC, expand=True
            )
            label_position = self.point(self.x(lyric['start']), 854)
            canvas.paste(
                tile,
                (label_position[0], label_position[1] - tile.height // 2),
                tile,
            )
        return canvas

    def frame(self, seconds: float) -> Image.Image:
        """Show bounded complete routes through the current recursive depth."""
        canvas = self.base.copy()
        draw = ImageDraw.Draw(canvas)
        sample = min(self.samples - 1, round(seconds * self.sample_rate))
        stage_index = min(
            self.levels - 1, bisect.bisect_right(self.stage_ends, sample)
        )
        depth = self.trace['stages'][stage_index]['depth']
        self.line(
            draw, [(self.x(sample), 278), (self.x(sample), 978)], '#3c4b39'
        )
        edge_limit = self.trace['rendering']['max_visual_edges']
        roots = self.active(depth - 1, sample)
        edge_count = 0

        def walk(layer: int, clock: int, node: ActiveVoice, budget: int) -> int:
            head, voice, source, segment, relative_gain = node
            start = (self.x(clock), self.lane_y(layer, head))
            self.dot(draw, *start, 2.5, recursive_render.GREEN)
            y = 285 + (self.levels - 1 - layer) * 560 / self.levels
            self.dot(
                draw,
                126 + head % 8 * 10,
                y + 8 + head // 8 * 10,
                2.5,
                recursive_render.GREEN,
            )
            children = self.active(layer - 1, source) if layer else []
            if children and self.trace['rendering'].get(
                'show_intermediate_audio'
            ):
                # Direct audio at this intermediate clock is an audible branch.
                self.arrow(
                    draw, start, (self.x(source), 915), '#849b80', bend=20
                )
                self.dot(draw, self.x(source), 915, 2, '#849b80')
            if not children:
                end = (self.x(source), 915)
                self.arrow(draw, start, end, recursive_render.CHILD, bend=20)
                self.bracket(
                    draw,
                    self.x(segment['source_start']),
                    self.x(segment['source_end']),
                    912,
                    '#577b5e',
                    1,
                )
                self.dot(draw, *end, 2.5, recursive_render.CHILD)
                return 1
            # Reserve enough edges for complete paths down to the input.
            # Audio remains unpruned. Ties favor longer current tape runs.
            children.sort(
                key=lambda child: (
                    -child.relative_gain,
                    -(
                        child.segment['output_end']
                        - child.segment['output_start']
                    ),
                    child.head_index,
                )
            )
            take = min(len(children), max(1, budget // (layer + 1)))
            used = int(
                bool(self.trace['rendering'].get('show_intermediate_audio'))
            )
            for child in children[:take]:
                end = (self.x(source), self.lane_y(layer - 1, child.head_index))
                self.arrow(
                    draw,
                    start,
                    end,
                    recursive_render.GREEN
                    if layer == depth - 1
                    else recursive_render.CHILD,
                    bend=12,
                )
                self.bracket(
                    draw,
                    self.x(segment['source_start']),
                    self.x(segment['source_end']),
                    end[1] - 3,
                    '#46623c',
                    1,
                )
                used += 1 + walk(
                    layer - 1, source, child, max(layer, budget // take - 1)
                )
            return used

        for root in roots:
            edge_count += walk(
                depth - 1, sample, root, edge_limit // max(1, len(roots))
            )
        self.text(
            draw,
            (1400, 164),
            f'Depth {depth:02d} / {self.levels:02d}   {seconds:06.2f}s',
            recursive_render.GREEN,
            self.font_label,
        )
        label = next(
            (
                annotations.display_label(section['label'])
                for section in self.trace['annotations']['sections']
                if section['start'] <= sample < section['end']
            ),
            '',
        )
        estimated = self.trace['annotations']['method'].startswith(
            'spectral novelty'
        )
        self.text(
            draw,
            (66, 1044),
            ('Estimated: ' if estimated else '') + label,
            recursive_render.INK,
        )
        self.text(
            draw,
            (1300, 1044),
            f'{edge_count:03d} connections',
            '#7c8a79',
        )
        return canvas
