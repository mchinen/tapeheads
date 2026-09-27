"""Draw recursive tape paths with vector marks and a restrained dot field."""

import bisect
import math

import numpy as np
from PIL import Image
from PIL import ImageDraw

from tapeheads import render
from tapeheads import trace_types

PAPER = '#101513'
INK = '#e5eee5'
GREEN = '#b7f568'
CHILD = '#73bca0'
FAINT = '#27342b'


class RecursivePlate(render.Plate):
    """Show output, intermediate, and source positions on two head planes."""

    def __init__(
        self,
        trace: trace_types.JsonObject,
        source: np.ndarray,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        layer = trace['layers'][1]
        shell = {**layer, 'source': trace['source']}
        super().__init__(shell, source, width, height)
        self.trace = trace
        self.left = 220
        self.right = 1780
        self.wave_y = 946
        self.starts = np.array(
            [event['output_start'] for event in trace['events']]
        )
        self.ends = np.array([event['output_end'] for event in trace['events']])
        self.queries = trace['layers'][0]['frames']['query_samples']
        self.base = self.background()

    def lane_y(self, layer: int, head: int) -> float:
        """Locate a native head on one of the two aligned planes."""
        count = len(self.trace['layers'][layer]['heads'])
        return (586 if layer == 0 else 300) + (head + 0.5) * 206 / count

    def dot(
        self,
        draw: ImageDraw.ImageDraw,
        x: float,
        y: float,
        radius: float,
        color: str,
    ) -> None:
        """Draw a circle in layout coordinates."""
        draw.ellipse(
            [
                self.point(x - radius, y - radius),
                self.point(x + radius, y + radius),
            ],
            fill=color,
        )

    def arrow(
        self,
        draw: ImageDraw.ImageDraw,
        start: tuple[float, float],
        end: tuple[float, float],
        color: str = GREEN,
        bend: float = 0,
        width: int = 1,
    ) -> None:
        """Draw a cubic vector with a directional arrowhead."""
        start_x, start_y = start
        end_x, end_y = end
        points = []
        for index in range(25):
            fraction = index / 24
            x = (
                (1 - fraction) ** 3 * start_x
                + 3 * (1 - fraction) ** 2 * fraction * (start_x + bend)
                + 3 * (1 - fraction) * fraction**2 * (end_x - bend)
                + fraction**3 * end_x
            )
            y = (
                (1 - fraction) ** 3 * start_y
                + 3 * (1 - fraction) ** 2 * fraction * (start_y - 12)
                + 3 * (1 - fraction) * fraction**2 * (end_y - 12)
                + fraction**3 * end_y
            )
            points.append((x, y))
        self.line(draw, points, color, width)
        angle = math.atan2(end_y - points[-2][1], end_x - points[-2][0])
        self.line(
            draw,
            [
                (
                    end_x - 7 * math.cos(angle - 0.5),
                    end_y - 7 * math.sin(angle - 0.5),
                ),
                (end_x, end_y),
                (
                    end_x - 7 * math.cos(angle + 0.5),
                    end_y - 7 * math.sin(angle + 0.5),
                ),
            ],
            color,
            width,
        )

    def bracket(
        self,
        draw: ImageDraw.ImageDraw,
        left: float,
        right: float,
        y: float,
        color: str,
        width: int = 2,
    ) -> None:
        """Mark true source bounds without inflating short selections."""
        self.line(
            draw,
            [(left, y - 4), (left, y + 3), (right, y + 3), (right, y - 4)],
            color,
            width,
        )

    def background(self) -> Image.Image:
        """Build the static report typography, dot texture, and waveform."""
        canvas = Image.new('RGB', (self.width, self.height), PAPER)
        draw = ImageDraw.Draw(canvas)
        self.text(
            draw,
            (64, 42),
            'Tapeheads',
            INK,
            self.font_brand,
        )
        self.text(draw, (1490, 42), 'Two layers', GREEN)
        self.text(
            draw, (64, 101), self.trace['source']['title'], INK, self.font_title
        )
        self.text(
            draw,
            (66, 167),
            f'{self.trace["encoder"]}   /   2 layers · full input',
            '#91a58e',
        )
        self.line(draw, [(64, 216), (1856, 216)], '#2e3c31')
        for layer, title, top in (
            (1, '02   Layer 2', 256),
            (0, '01   Layer 1', 542),
        ):
            self.text(draw, (66, top), title, INK, self.font_label)
            for x in range(220, 1781, 18):
                for y in range(top + 44, top + 247, 18):
                    self.dot(draw, x, y, 0.7, FAINT)
            for head in range(len(self.trace['layers'][layer]['heads'])):
                y = self.lane_y(layer, head)
                self.text(draw, (157, y - 8), f'{head + 1:02d}', '#8e9c8c')
                self.dot(draw, 198, y, 2, '#34483a')
        self.line(draw, [(64, 832), (1856, 832)], '#2e3c31')
        self.text(draw, (66, 857), '00   Input', INK, self.font_label)
        for index, peak in enumerate(self.peaks):
            x = self.left + index / len(self.peaks) * (self.right - self.left)
            for offset in range(0, max(1, round(peak * 35)), 5):
                self.dot(draw, x, self.wave_y - offset, 1, '#637961')
                if offset:
                    self.dot(draw, x, self.wave_y + offset, 1, '#637961')
        for tick in range(7):
            sample = tick * self.samples / 6
            self.text(
                draw,
                (self.x(sample) - 12, 994),
                f'{sample / self.sample_rate:.1f}s',
                '#7c8a79',
            )
        self.text(draw, (66, 1044), 'Output → layer 1 → source', INK)
        self.text(
            draw,
            (1050, 1044),
            'Layer 2 → layer 1 → input',
            '#71806d',
        )
        return canvas

    def selected_voice(
        self, layer: int, head: int, voice: int, sample: int
    ) -> trace_types.SelectedVoice | None:
        """Find a voice's actual accumulated playback bracket at its clock."""
        queries = self.trace['layers'][layer]['frames']['query_samples']
        index = bisect.bisect_right(queries, sample) - 1
        if index < 0 or head < 0:
            return None
        selections = self.trace['layers'][layer]['heads'][head]['selections']
        return next(
            (
                block
                for block in selections[index]['selected_blocks']
                if block['voice_id'] == voice
            ),
            None,
        )

    def frame(self, seconds: float) -> Image.Image:
        """Connect each query directly to its attended intermediate position."""
        canvas = self.base.copy()
        draw = ImageDraw.Draw(canvas)
        sample = min(self.samples - 1, round(seconds * self.sample_rate))
        x = self.x(sample)
        self.line(draw, [(x, 295), (x, 807)], '#445343')
        self.dot(draw, x, 289, 3, INK)
        active = [
            self.trace['events'][int(index)]
            for index in np.flatnonzero(
                (self.starts <= sample) & (self.ends > sample)
            )
        ]
        # Hide quiet monitor paths; they are still included in the audio.
        visible = [
            event
            for event in active
            if event['parent_head'] >= 0
            and not event['near_output']
            and event['gain'] > 0
        ]
        paths = [
            event
            for event in visible
            if event['branch'] == 'recursive' and event['child_head'] >= 0
        ]
        parent_keys = set()
        brackets = set()
        for event in paths:
            parent, child = event['parent_head'], event['child_head']
            parent_y, child_y = self.lane_y(1, parent), self.lane_y(0, child)
            middle = (
                event['intermediate_start'] + sample - event['output_start']
            )
            source = event['source_start'] + sample - event['output_start']
            intermediate_x, source_x = self.x(middle), self.x(source)
            key = (parent, event['parent_voice'])
            parent_keys.add(key)
            self.dot(draw, 198, parent_y, 3, GREEN)
            self.dot(draw, x, parent_y, 3, GREEN)
            self.dot(draw, 198, child_y, 3, CHILD)
            block = self.selected_voice(
                1, parent, event['parent_voice'], sample
            )
            bracket_key = (*key, child)
            if block and bracket_key not in brackets:
                self.bracket(
                    draw,
                    self.x(block['playback_start']),
                    self.x(block['playback_end']),
                    child_y,
                    '#61834c',
                    1,
                )
                brackets.add(bracket_key)
            # Both endpoints use the event clocks, not bracket midpoints.
            self.arrow(
                draw,
                (x, parent_y),
                (intermediate_x, child_y + 3),
                GREEN,
                bend=25,
                width=1,
            )
            self.dot(draw, intermediate_x, child_y + 3, 3, GREEN)
            wave_y = 897 + (event['parent_voice'] % 3) * 6
            child_block = self.selected_voice(
                0, child, event['child_voice'], middle
            )
            if child_block:
                self.bracket(
                    draw,
                    self.x(child_block['playback_start']),
                    self.x(child_block['playback_end']),
                    wave_y,
                    CHILD,
                    1,
                )
            self.arrow(
                draw,
                (intermediate_x, child_y + 3),
                (source_x, wave_y + 3),
                CHILD,
                bend=25,
            )
            self.dot(draw, source_x, wave_y + 3, 3, CHILD)
            self.line(draw, [(source_x, 919), (source_x, 975)], '#4c715b')
        for event in visible:
            if event['branch'] != 'direct':
                continue
            parent = event['parent_head']
            parent_y = self.lane_y(1, parent)
            source = event['source_start'] + sample - event['output_start']
            source_x = self.x(source)
            key = (parent, event['parent_voice'])
            if key not in parent_keys:
                self.arrow(
                    draw, (x, parent_y), (source_x, 809), '#72856c', bend=25
                )
                self.dot(draw, source_x, 809, 3, INK)
            parent_keys.add(key)
            self.dot(draw, source_x, 984, 3, INK)
        self.text(
            draw,
            (1410, 159),
            f'{seconds:06.2f} / {self.samples / self.sample_rate:06.2f}s',
            INK,
            self.font_label,
        )
        self.text(draw, (66, 282), f'{len(parent_keys):02d} voices', GREEN)
        self.text(draw, (66, 568), f'{len(paths):02d} paths', CHILD)
        return canvas
