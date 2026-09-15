"""Behavioral regression coverage for the bounded PR #97438 SSE backport.

Provenance: PR base ``3ca096de5f8183cb2e0ec23673f294d5978656a3``;
upstream commits ``2a29e735aefd71866428e54cb8622cd89fa1fd4b`` and
``4d8ca34403ba3c4dbe9652b751e750410a1edad0``.

The agent/provider boundary is synthetic.  The real ``APIServerAdapter``,
``/v1/chat/completions`` handler, async queue, and SSE writer are exercised.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import make_mocked_request

from gateway.config import PlatformConfig
from gateway.platforms import api_server as api_server_module
from gateway.platforms.api_server import APIServerAdapter, ThreadSafeAsyncQueue


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


class _CaptureWriter:
    """Minimal aiohttp payload writer that captures the real SSE bytes."""

    buffer_size = 0
    output_size = 0
    length = None

    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    async def write(self, chunk: bytes) -> None:
        self.chunks.append(bytes(chunk))

    async def write_eof(self, chunk: bytes = b"") -> None:
        if chunk:
            self.chunks.append(bytes(chunk))

    async def drain(self) -> None:
        return None

    def enable_compression(self, encoding: str = "deflate", strategy=None) -> None:
        return None

    def enable_chunking(self) -> None:
        return None

    async def write_headers(self, status_line: str, headers) -> None:
        return None

    def send_headers(self) -> None:
        return None


def _request_and_writer(**body_overrides):
    writer = _CaptureWriter()
    request = make_mocked_request(
        "POST",
        "/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        writer=writer,
    )
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": True,
        **body_overrides,
    }
    request._read_bytes = json.dumps(body).encode()
    return request, writer


async def _call_route(adapter: APIServerAdapter, fake_run_agent, **body_overrides):
    request, writer = _request_and_writer(**body_overrides)
    with patch.object(adapter, "_run_agent", side_effect=fake_run_agent):
        response = await adapter._handle_chat_completions(request)
    return response, b"".join(writer.chunks).decode()


async def _run_stream(fake_run_agent, **body_overrides):
    adapter = _make_adapter()
    try:
        return await _call_route(adapter, fake_run_agent, **body_overrides)
    finally:
        await adapter.disconnect()


def _json_events(body: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def _content(events: list[dict]) -> list[object]:
    return [
        choice["delta"]["content"]
        for event in events
        for choice in event.get("choices", [])
        if "content" in choice.get("delta", {})
    ]


def _finish_events(events: list[dict]) -> list[dict]:
    return [
        event
        for event in events
        if event.get("choices", [{}])[0].get("finish_reason") is not None
    ]


@pytest.mark.asyncio
async def test_final_response_is_streamed_when_provider_emits_no_deltas():
    async def fake_run_agent(**kwargs):
        assert kwargs["stream_delta_callback"] is not None
        return (
            {"final_response": "pong", "messages": [], "api_calls": 1},
            {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )

    response, body = await _run_stream(fake_run_agent)

    events = _json_events(body)
    finishes = _finish_events(events)
    assert response.status == 200
    assert _content(events) == ["pong"]
    assert [event["choices"][0]["finish_reason"] for event in finishes] == ["stop"]
    assert [event["usage"] for event in finishes] == [
        {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    ]
    assert body.count("data: [DONE]\n\n") == 1
    assert body.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_existing_text_delta_is_not_duplicated_by_final_response():
    async def fake_run_agent(**kwargs):
        kwargs["stream_delta_callback"]("same answer")
        return (
            {"final_response": "same answer", "messages": [], "api_calls": 1},
            {},
        )

    _, body = await _run_stream(fake_run_agent)

    events = _json_events(body)
    assert _content(events) == ["same answer"]
    assert len(_finish_events(events)) == 1
    assert body.count("data: [DONE]\n\n") == 1


@pytest.mark.parametrize(
    ("delta", "expected_content"),
    [
        ("", ["", "fallback"]),
        (" \n", [" \n"]),
    ],
    ids=("empty-delta-allows-fallback", "whitespace-delta-is-text"),
)
@pytest.mark.asyncio
async def test_delta_text_boundary_controls_fallback(delta, expected_content):
    async def fake_run_agent(**kwargs):
        kwargs["stream_delta_callback"](delta)
        return ({"final_response": "fallback"}, {})

    _, body = await _run_stream(fake_run_agent)

    assert _content(_json_events(body)) == expected_content


@pytest.mark.asyncio
async def test_tool_progress_only_does_not_suppress_final_response():
    async def fake_run_agent(**kwargs):
        kwargs["tool_start_callback"](
            "call_terminal_1", "terminal", {"command": "printf synthetic"}
        )
        return (
            {"final_response": "tool result", "messages": [], "api_calls": 1},
            {},
        )

    _, body = await _run_stream(fake_run_agent)

    assert "event: hermes.tool.progress" in body
    assert _content(_json_events(body)) == ["tool result"]


@pytest.mark.parametrize(
    ("final_response", "expected_content"),
    [
        ("", []),
        (" \n", [" \n"]),
        (None, []),
        (7, []),
        (["not", "text"], []),
    ],
    ids=("empty", "whitespace", "none", "number", "list"),
)
@pytest.mark.asyncio
async def test_final_response_boundary_types(final_response, expected_content):
    async def fake_run_agent(**kwargs):
        return ({"final_response": final_response}, {})

    _, body = await _run_stream(fake_run_agent)

    assert _content(_json_events(body)) == expected_content
    assert body.count("data: [DONE]\n\n") == 1


@pytest.mark.parametrize(
    ("result", "finish_reason"),
    [
        (
            {"failed": True, "error": "provider failed"},
            "error",
        ),
        (
            {"partial": True},
            "error",
        ),
        (
            {"partial": True, "completed": False, "error": "output truncated"},
            "length",
        ),
        (
            {"completed": False},
            "error",
        ),
        (
            {"error": "provider returned an error"},
            "error",
        ),
    ],
    ids=("failed", "partial", "truncation", "incomplete", "error-field"),
)
@pytest.mark.asyncio
async def test_abnormal_result_never_gains_success_fallback(result, finish_reason):
    marker = "must-not-be-streamed-as-success"

    async def fake_run_agent(**kwargs):
        return ({"final_response": marker, **result}, {})

    _, body = await _run_stream(fake_run_agent)

    events = _json_events(body)
    assert _content(events) == []
    assert [event["choices"][0]["finish_reason"] for event in _finish_events(events)] == [
        finish_reason
    ]
    assert body.count("data: [DONE]\n\n") == 1


@pytest.mark.parametrize("failure_kind", ["result", "task"])
@pytest.mark.asyncio
async def test_stream_errors_are_secret_safe_and_terminate_once(failure_kind, caplog):
    secret = "sk-issue125-must-not-cross-sse-boundary"

    async def fake_run_agent(**kwargs):
        message = f"auth failed OPENAI_API_KEY={secret}"
        if failure_kind == "task":
            raise RuntimeError(message)
        return ({"failed": True, "error": message}, {})

    _, body = await _run_stream(fake_run_agent)

    events = _json_events(body)
    assert secret not in body
    assert secret not in caplog.text
    assert "OPENAI_API_KEY=" in body
    assert [event["choices"][0]["finish_reason"] for event in _finish_events(events)] == [
        "error"
    ]
    assert body.count("data: [DONE]\n\n") == 1


@pytest.mark.asyncio
async def test_queue_is_drained_when_agent_finishes_before_writer_catches_up():
    async def fake_run_agent(**kwargs):
        callback = kwargs["stream_delta_callback"]
        callback("queued-before-")
        callback("task-completion")
        return ({"final_response": "queued-before-task-completion"}, {})

    _, body = await _run_stream(fake_run_agent)

    assert _content(_json_events(body)) == ["queued-before-", "task-completion"]
    assert body.count("data: [DONE]\n\n") == 1


@pytest.mark.asyncio
async def test_client_disconnect_interrupts_and_cancels_agent_task():
    class FakeAgent:
        def __init__(self) -> None:
            self.interrupt_messages: list[str] = []

        def interrupt(self, message=None) -> None:
            self.interrupt_messages.append(message)

    class DisconnectingResponse:
        def __init__(self) -> None:
            self.write_count = 0

        async def prepare(self, request) -> None:
            return None

        async def write(self, payload) -> None:
            self.write_count += 1
            if self.write_count == 2:
                raise ConnectionResetError("synthetic disconnect")

    async def unfinished_agent():
        await asyncio.Event().wait()

    adapter = _make_adapter()
    stream_q = ThreadSafeAsyncQueue()
    stream_q.put_nowait("first delta")
    agent = FakeAgent()
    task = asyncio.create_task(unfinished_agent())
    request = MagicMock(headers={})
    fake_response = DisconnectingResponse()
    try:
        with patch.object(
            api_server_module.web, "StreamResponse", return_value=fake_response
        ):
            returned = await adapter._write_sse_chat_completion(
                request=request,
                completion_id="chatcmpl-disconnect",
                model="test",
                created=1,
                stream_q=stream_q,
                agent_task=task,
                agent_ref=[agent],
            )
    finally:
        await adapter.disconnect()

    assert returned is fake_response
    assert agent.interrupt_messages == ["SSE client disconnected"]
    assert task.cancelled()


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_cross_stream_content():
    adapter = _make_adapter()
    request_a, writer_a = _request_and_writer(
        messages=[{"role": "user", "content": "request-a"}]
    )
    request_b, writer_b = _request_and_writer(
        messages=[{"role": "user", "content": "request-b"}]
    )

    async def fake_run_agent(**kwargs):
        label = kwargs["user_message"]
        callback = kwargs["stream_delta_callback"]
        callback(f"{label}:first")
        await asyncio.sleep(0)
        callback(f"{label}:second")
        return ({"final_response": f"{label}:first{label}:second"}, {})

    try:
        with patch.object(adapter, "_run_agent", side_effect=fake_run_agent):
            response_a, response_b = await asyncio.gather(
                adapter._handle_chat_completions(request_a),
                adapter._handle_chat_completions(request_b),
            )
    finally:
        await adapter.disconnect()

    body_a = b"".join(writer_a.chunks).decode()
    body_b = b"".join(writer_b.chunks).decode()
    assert response_a.status == response_b.status == 200
    assert _content(_json_events(body_a)) == ["request-a:first", "request-a:second"]
    assert _content(_json_events(body_b)) == ["request-b:first", "request-b:second"]
    assert "request-b" not in body_a
    assert "request-a" not in body_b
