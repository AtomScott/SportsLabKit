from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Box is of shape (1,2xdim), e.g. for dim=2 [xmin, ymin, width, height] format is accepted
Box = np.ndarray # deprecated

@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @property
    def int_xy_tuple(self) -> tuple[int, int]:
        return int(self.x), int(self.y)

@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def min_x(self) -> float:
        return self.x

    @property
    def min_y(self) -> float:
        return self.y

    @property
    def max_x(self) -> float:
        return self.x + self.width

    @property
    def max_y(self) -> float:
        return self.y + self.height

    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)

    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    @property
    def bottom_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height)

    @property
    def top_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y)

    @property
    def center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height / 2)

    def pad(self, padding: float) -> 'Rect':
        return Rect(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2*padding,
            height=self.height + 2*padding
        )

    def contains_point(self, point: Point) -> bool:
        return self.min_x < point.x < self.max_x and self.min_y < point.y < self.max_y


_COLOR_NAME_TO_RGB: dict[str, tuple[int, int, int]] = {
    "navy": (0, 38, 63),
    "blue": (0, 120, 210),
    "aqua": (115, 221, 252),
    "teal": (15, 205, 202),
    "olive": (52, 153, 114),
    "green": (0, 204, 84),
    "lime": (1, 255, 127),
    "yellow": (255, 216, 70),
    "orange": (255, 125, 57),
    "red": (255, 47, 65),
    "maroon": (135, 13, 75),
    "fuchsia": (246, 0, 184),
    "purple": (179, 17, 193),
    "black": (24, 24, 24),
    "gray": (168, 168, 168),
    "silver": (220, 220, 220),
    "brown": (139, 69, 19),
    "gold": (255, 215, 0)
}

_COLOR_NAMES = list(_COLOR_NAME_TO_RGB.keys())
_COLORS = list(_COLOR_NAME_TO_RGB.values())

@dataclass(frozen=True)
class Color:
    r: int
    g: int
    b: int

    @property
    def bgr_tuple(self) -> tuple[int, int, int]:
        return self.b, self.g, self.r

    @classmethod
    def from_hex_string(cls, hex_string: str) -> 'Color':
        r, g, b = tuple(int(hex_string[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
        return cls(r=r, g=g, b=b)

    @classmethod
    def from_name(cls, color_name: str) -> 'Color':
        rgb = _COLOR_NAME_TO_RGB.get(color_name.lower())
        if rgb is None:
            raise ValueError(f"Color name {color_name} not recognized.")
        return cls(r=rgb[0], g=rgb[1], b=rgb[2])


# Vector is of shape (1, N)
Vector = np.ndarray

_pathlike = str | Path

# numpy/opencv image alias
NpImage = np.ndarray
