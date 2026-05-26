import asyncio
import json

from PIL import Image

from mineru_vl_utils import mineru_client as mineru_client_module
from mineru_vl_utils.dissection import DissectionRecorder
from mineru_vl_utils.structs import ContentBlock


class FakeLayoutClient:
    server_url = "http://layout"
    chat_url = "http://layout/v1/chat/completions"
    model_name = "layout-model"

    async def aio_predict(self, image, prompt, sampling_params=None, priority=None):
        return "layout-output"


class FakeRecognitionClient:
    server_url = "http://recognition"
    chat_url = "http://recognition/v1/chat/completions"
    model_name = "recognition-model"

    async def aio_predict(self, image, prompt, sampling_params=None, priority=None):
        if prompt == "bad-prompt":
            raise RuntimeError("recognition failed")
        return f"recognized:{prompt}"

    async def async_stream_predict(self, image, prompt, sampling_params=None, priority=None):
        if prompt == "partial-prompt":
            yield "part"
            yield "ial"
            raise TimeoutError("stream timeout")
        if prompt == "empty-fail-prompt":
            raise RuntimeError("stream failed before tokens")
        yield f"recognized:{prompt}"


def test_aio_two_step_extract_records_partial_recognition_failure(monkeypatch, tmp_path):
    def fake_new_vlm_client(**kwargs):
        if kwargs.get("server_url") == "http://layout":
            return FakeLayoutClient()
        return FakeRecognitionClient()

    monkeypatch.setattr(mineru_client_module, "new_vlm_client", fake_new_vlm_client)
    client = mineru_client_module.MinerUClient(
        backend="http-client",
        layout_server_url="http://layout",
        recognition_server_url="http://recognition",
        use_tqdm=False,
    )
    async def fake_aio_prepare_for_layout(executor, image):
        return b"layout-image"

    client.helper.prepare_for_layout = lambda image: b"layout-image"
    client.helper.parse_layout_output = lambda output: [
        ContentBlock("text", [0.0, 0.0, 0.5, 1.0]),
        ContentBlock("text", [0.5, 0.0, 1.0, 1.0]),
    ]
    async def fake_aio_parse_layout_output(executor, output):
        return client.helper.parse_layout_output(output)

    async def fake_aio_prepare_for_extract(
        executor,
        image,
        blocks,
        not_extract_list=None,
        image_analysis=None,
    ):
        return (
            [b"crop-0", b"crop-1"],
            ["good-prompt", "bad-prompt"],
            [None, None],
            [0, 1],
        )

    async def fake_aio_post_process(executor, blocks):
        return blocks

    client.helper.aio_parse_layout_output = fake_aio_parse_layout_output
    client.helper.aio_prepare_for_layout = fake_aio_prepare_for_layout
    client.helper.aio_prepare_for_extract = fake_aio_prepare_for_extract
    client.helper.aio_post_process = fake_aio_post_process

    recorder = DissectionRecorder(tmp_path / "dissection", document_stem="doc")
    result = asyncio.run(
        client.aio_two_step_extract(
            Image.new("RGB", (20, 10), (255, 255, 255)),
            dissection_recorder=recorder,
            page_idx=0,
        )
    )
    recorder.finalize()

    assert result[0].content == "recognized:good-prompt"
    assert result[1].content is None

    first = json.loads(
        (tmp_path / "dissection" / "recognition" / "page_000_bbox_000.json").read_text(encoding="utf-8")
    )
    second = json.loads(
        (tmp_path / "dissection" / "recognition" / "page_000_bbox_001.json").read_text(encoding="utf-8")
    )
    assert first["status"] == "recognized"
    assert second["status"] == "failed"
    assert second["error"] == "recognition failed"

    manifest = json.loads((tmp_path / "dissection" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status_counts"] == {"recognized": 1, "failed": 1}


def test_aio_two_step_extract_records_stream_partial_content(monkeypatch, tmp_path):
    def fake_new_vlm_client(**kwargs):
        if kwargs.get("server_url") == "http://layout":
            return FakeLayoutClient()
        return FakeRecognitionClient()

    monkeypatch.setattr(mineru_client_module, "new_vlm_client", fake_new_vlm_client)
    client = mineru_client_module.MinerUClient(
        backend="http-client",
        layout_server_url="http://layout",
        recognition_server_url="http://recognition",
        use_tqdm=False,
    )

    async def fake_aio_prepare_for_layout(executor, image):
        return b"layout-image"

    client.helper.parse_layout_output = lambda output: [
        ContentBlock("text", [0.0, 0.0, 0.5, 1.0]),
        ContentBlock("text", [0.5, 0.0, 1.0, 1.0]),
    ]

    async def fake_aio_parse_layout_output(executor, output):
        return client.helper.parse_layout_output(output)

    async def fake_aio_prepare_for_extract(
        executor,
        image,
        blocks,
        not_extract_list=None,
        image_analysis=None,
    ):
        return (
            [b"crop-0", b"crop-1"],
            ["good-prompt", "partial-prompt"],
            [None, None],
            [0, 1],
        )

    async def fake_aio_post_process(executor, blocks):
        return blocks

    client.helper.aio_prepare_for_layout = fake_aio_prepare_for_layout
    client.helper.aio_parse_layout_output = fake_aio_parse_layout_output
    client.helper.aio_prepare_for_extract = fake_aio_prepare_for_extract
    client.helper.aio_post_process = fake_aio_post_process

    recorder = DissectionRecorder(tmp_path / "dissection", document_stem="doc")
    result = asyncio.run(
        client.aio_two_step_extract(
            Image.new("RGB", (20, 10), (255, 255, 255)),
            dissection_recorder=recorder,
            dissection_stream=True,
            page_idx=0,
        )
    )
    recorder.finalize()

    assert result[0].content == "recognized:good-prompt"
    assert result[1].content is None

    first = json.loads(
        (tmp_path / "dissection" / "recognition" / "page_000_bbox_000.json").read_text(encoding="utf-8")
    )
    second = json.loads(
        (tmp_path / "dissection" / "recognition" / "page_000_bbox_001.json").read_text(encoding="utf-8")
    )
    assert first["status"] == "recognized"
    assert second["status"] == "partial"
    assert second["partial_content"] == "partial"
    assert second["exception_type"] == "TimeoutError"

    manifest = json.loads((tmp_path / "dissection" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status_counts"] == {"recognized": 1, "partial": 1}


def test_aio_two_step_extract_stream_failure_without_tokens_is_failed(monkeypatch, tmp_path):
    def fake_new_vlm_client(**kwargs):
        if kwargs.get("server_url") == "http://layout":
            return FakeLayoutClient()
        return FakeRecognitionClient()

    monkeypatch.setattr(mineru_client_module, "new_vlm_client", fake_new_vlm_client)
    client = mineru_client_module.MinerUClient(
        backend="http-client",
        layout_server_url="http://layout",
        recognition_server_url="http://recognition",
        use_tqdm=False,
    )

    async def fake_aio_prepare_for_layout(executor, image):
        return b"layout-image"

    client.helper.parse_layout_output = lambda output: [ContentBlock("text", [0.0, 0.0, 1.0, 1.0])]

    async def fake_aio_parse_layout_output(executor, output):
        return client.helper.parse_layout_output(output)

    async def fake_aio_prepare_for_extract(
        executor,
        image,
        blocks,
        not_extract_list=None,
        image_analysis=None,
    ):
        return ([b"crop-0"], ["empty-fail-prompt"], [None], [0])

    async def fake_aio_post_process(executor, blocks):
        return blocks

    client.helper.aio_prepare_for_layout = fake_aio_prepare_for_layout
    client.helper.aio_parse_layout_output = fake_aio_parse_layout_output
    client.helper.aio_prepare_for_extract = fake_aio_prepare_for_extract
    client.helper.aio_post_process = fake_aio_post_process

    recorder = DissectionRecorder(tmp_path / "dissection", document_stem="doc")
    result = asyncio.run(
        client.aio_two_step_extract(
            Image.new("RGB", (20, 10), (255, 255, 255)),
            dissection_recorder=recorder,
            dissection_stream=True,
            page_idx=0,
        )
    )
    recorder.finalize()

    assert result[0].content is None
    record = json.loads(
        (tmp_path / "dissection" / "recognition" / "page_000_bbox_000.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "failed"
    assert "partial_content" not in record
