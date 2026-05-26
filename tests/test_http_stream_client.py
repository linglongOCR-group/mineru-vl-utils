import pytest
import httpx

from mineru_vl_utils.vlm_client.base_client import RequestError, ServerError
from mineru_vl_utils.vlm_client.http_client import HttpVlmClient


def _stream_client(transport: httpx.BaseTransport) -> HttpVlmClient:
    client = object.__new__(HttpVlmClient)
    client.server_url = "http://server"
    client.model_name = "mineru-vlm"
    client.system_prompt = "system"
    client.sampling_params = None
    client.text_before_image = False
    client.allow_truncated_content = False
    client.debug = False
    client._client = httpx.Client(transport=transport)
    return client


def _sse(*events: str) -> bytes:
    return "".join(f"data: {event}\n\n" for event in events).encode("utf-8")


def test_stream_predict_raises_on_non_200_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server failed")

    client = _stream_client(httpx.MockTransport(handler))

    with pytest.raises(ServerError, match="Unexpected status code"):
        list(client.stream_predict(None, "prompt"))


def test_stream_predict_raises_on_length_finish_reason_after_chunks():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                '{"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}',
                '{"choices":[{"delta":{},"finish_reason":"length"}]}',
                "[DONE]",
            ),
        )

    client = _stream_client(httpx.MockTransport(handler))
    chunks = []

    with pytest.raises(RequestError, match="truncated"):
        for chunk in client.stream_predict(None, "prompt"):
            chunks.append(chunk)

    assert chunks == ["partial"]
