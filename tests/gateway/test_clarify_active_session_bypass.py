"""Regression tests for clarify replies while a gateway session is busy."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


class _ClarifyBypassAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


def _event(text="custom answer"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="private",
            user_id="user1",
        ),
        message_id="msg1",
    )


def _clear_clarify_state():
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


@pytest.mark.asyncio
async def test_active_session_routes_typed_choice_clarify_reply_to_runner_not_busy_queue():
    """Typed text must resolve a pending choice clarify even while the agent is busy.

    Telegram button clarifies keep the adapter session active while the agent
    thread blocks on ``wait_for_response``.  If the adapter only bypasses for
    entries already marked ``awaiting_text``, typed replies to the visible
    multi-choice prompt are handled as busy follow-ups and the clarify wait is
    never resolved.
    """
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _ClarifyBypassAdapter()
    adapter._message_handler = AsyncMock(return_value="")
    adapter._busy_session_handler = AsyncMock(return_value=True)
    event = _event("None of those are valid options")
    session_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    adapter._active_sessions[session_key] = asyncio.Event()
    cm.register("clarify-1", session_key, "Pick one", ["A", "B"])

    await adapter.handle_message(event)

    adapter._message_handler.assert_awaited_once_with(event)
    adapter._busy_session_handler.assert_not_awaited()
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_profile_routed_active_session_finds_profile_clarify_key():
    """A routed topic reply must use the profile's active session key."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _ClarifyBypassAdapter()
    adapter._message_handler = AsyncMock(return_value="")
    adapter._busy_session_handler = AsyncMock(return_value=True)

    event = MessageEvent(
        text="screenshot attached",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100123",
            chat_type="group",
            user_id="user-1",
            thread_id="42",
            profile="pilot",
        ),
        message_id="msg-profile",
    )
    session_key = build_session_key(event.source, profile=event.source.profile)
    adapter._active_sessions[session_key] = asyncio.Event()
    cm.register("clarify-profile", session_key, "Send a screenshot", None)

    with patch.object(adapter, "_start_session_processing") as start_processing:
        await adapter.handle_message(event)

    adapter._message_handler.assert_awaited_once_with(event)
    adapter._busy_session_handler.assert_not_awaited()
    start_processing.assert_not_called()
    assert adapter._pending_messages == {}
    _clear_clarify_state()


@pytest.mark.asyncio
async def test_prequeued_profile_image_resolves_new_open_ended_clarify():
    """A screenshot queued before clarify registration must unblock that clarify."""
    _clear_clarify_state()
    from gateway.run import _resolve_prequeued_image_clarify
    from tools import clarify_gateway as cm

    adapter = _ClarifyBypassAdapter()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100123",
        chat_type="group",
        user_id="user-1",
        thread_id="42",
        profile="pilot",
    )
    session_key = build_session_key(source, profile=source.profile)
    image_event = MessageEvent(
        text="",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/screenshot.jpg"],
        media_types=["image/jpeg"],
        message_id="image-before-clarify",
    )
    adapter._pending_messages[session_key] = image_event
    entry = cm.register(
        "clarify-after-image",
        session_key,
        "Send a screenshot",
        None,
    )

    resolved = await _resolve_prequeued_image_clarify(
        adapter,
        session_key,
        entry.clarify_id,
        image_event,
    )

    assert resolved is True
    assert entry.event.is_set()
    assert entry.response == "[User sent an image: /tmp/screenshot.jpg]"
    assert adapter.get_pending_message(session_key) is None
    _clear_clarify_state()


@pytest.mark.asyncio
async def test_prequeued_clarify_does_not_consume_replacement_event():
    """Reconciliation must stay pinned to the image captured at registration."""
    _clear_clarify_state()
    from gateway.run import _resolve_prequeued_image_clarify
    from tools import clarify_gateway as cm

    adapter = _ClarifyBypassAdapter()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100123",
        chat_type="group",
        user_id="user-1",
        thread_id="42",
        profile="pilot",
    )
    session_key = build_session_key(source, profile=source.profile)
    original_image = MessageEvent(
        text="",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/original-screenshot.jpg"],
        media_types=["image/jpeg"],
        message_id="original-image",
    )
    replacement_image = MessageEvent(
        text="later image",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/replacement-screenshot.jpg"],
        media_types=["image/jpeg"],
        message_id="replacement-image",
    )
    adapter._pending_messages[session_key] = original_image
    entry = cm.register(
        "clarify-after-original-image",
        session_key,
        "Send a screenshot",
        None,
    )

    # Model the event-loop delay called out by the independent review: the
    # exact object captured when clarify opened no longer owns the pending slot.
    adapter._pending_messages[session_key] = replacement_image
    resolved = await _resolve_prequeued_image_clarify(
        adapter,
        session_key,
        entry.clarify_id,
        original_image,
    )

    assert resolved is False
    assert not entry.event.is_set()
    assert entry.response is None
    assert adapter._pending_messages[session_key] is replacement_image
    _clear_clarify_state()


