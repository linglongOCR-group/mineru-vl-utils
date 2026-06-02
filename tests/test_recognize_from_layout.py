import asyncio

from PIL import Image

from mineru_vl_utils import mineru_client as mineru_client_module
from mineru_vl_utils.structs import ContentBlock, ExtractResult


class FakeLayoutClient:
    server_url = "http://layout"
    chat_url = "http://layout/v1/chat/completions"
    model_name = "layout-model"

    async def aio_predict(self, image, prompt, sampling_params=None, priority=None):
        return "layout-output"

    def predict(self, image, prompt, sampling_params=None, priority=None):
        return "layout-output"


class FakeRecognitionClient:
    server_url = "http://recognition"
    chat_url = "http://recognition/v1/chat/completions"
    model_name = "recognition-model"

    async def aio_predict(self, image, prompt, sampling_params=None, priority=None):
        return f"recognized:{prompt}"

    def predict(self, image, prompt, sampling_params=None, priority=None):
        return f"recognized:{prompt}"

    def batch_predict(self, images, prompts, params=None, priority=None):
        return [f"recognized:{p}" if not isinstance(prompts, str) else f"recognized:{prompts}" for p in (prompts if not isinstance(prompts, str) else [prompts] * len(images))]

    async def aio_batch_predict(self, images, prompts, params=None, priority=None, *, semaphore=None, use_tqdm=False, tqdm_desc=None):
        if isinstance(prompts, str):
            return [f"recognized:{prompts}"] * len(images)
        return [f"recognized:{p}" for p in prompts]


def _make_client(monkeypatch):
    """Create a MinerUClient with fake layout and recognition backends."""
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
    return client


def _install_sync_fakes(client):
    """Install sync fake helpers for layout + recognition stages."""
    client.helper.prepare_for_layout = lambda image: b"layout-image"
    client.helper.parse_layout_output = lambda output: [
        ContentBlock("text", [0.0, 0.0, 0.5, 1.0]),
        ContentBlock("text", [0.5, 0.0, 1.0, 1.0]),
    ]
    client.helper.prepare_for_extract = lambda image, blocks, nel=None, ia=None: (
        [b"crop-0", b"crop-1"],
        ["\nText Recognition:", "\nText Recognition:"],
        [None, None],
        [0, 1],
    )
    client.helper.post_process = lambda blocks: blocks


def _install_async_fakes(client):
    """Install async fake helpers for layout + recognition stages."""
    async def fake_aio_prepare_for_layout(executor, image):
        return b"layout-image"

    async def fake_aio_parse_layout_output(executor, output):
        return client.helper.parse_layout_output(output)

    async def fake_aio_prepare_for_extract(executor, image, blocks, nel=None, ia=None):
        return (
            [b"crop-0", b"crop-1"],
            ["\nText Recognition:", "\nText Recognition:"],
            [None, None],
            [0, 1],
        )

    async def fake_aio_post_process(executor, blocks):
        return blocks

    _install_sync_fakes(client)
    client.helper.aio_prepare_for_layout = fake_aio_prepare_for_layout
    client.helper.aio_parse_layout_output = fake_aio_parse_layout_output
    client.helper.aio_prepare_for_extract = fake_aio_prepare_for_extract
    client.helper.aio_post_process = fake_aio_post_process


# ----------------------------------------------------------------------
# Test 1: recognize_from_layout produces same content as two_step_extract
# ----------------------------------------------------------------------
def test_recognize_from_layout_matches_two_step_extract(monkeypatch):
    client = _make_client(monkeypatch)
    _install_sync_fakes(client)
    image = Image.new("RGB", (20, 10), (255, 255, 255))

    full_result = client.two_step_extract(image)

    layout_result = client.layout_detect(image)
    recognize_result = client.recognize_from_layout(image, layout_result)

    assert len(full_result) == len(recognize_result)
    for full_block, recog_block in zip(full_result, recognize_result):
        assert full_block.content == recog_block.content
        assert full_block.type == recog_block.type
        assert full_block.bbox == recog_block.bbox


# ----------------------------------------------------------------------
# Test 2: recognize_from_layout accepts plain list[ContentBlock]
# ----------------------------------------------------------------------
def test_recognize_from_layout_accepts_plain_list(monkeypatch):
    client = _make_client(monkeypatch)
    _install_sync_fakes(client)
    image = Image.new("RGB", (20, 10), (255, 255, 255))

    plain_blocks = [
        ContentBlock("text", [0.0, 0.0, 0.5, 1.0]),
        ContentBlock("text", [0.5, 0.0, 1.0, 1.0]),
    ]

    result = client.recognize_from_layout(image, plain_blocks)

    assert isinstance(result, ExtractResult)
    assert len(result) == 2
    assert result[0].content is not None
    assert result[1].content is not None
    assert result.layout_scored is None


# ----------------------------------------------------------------------
# Test 3: aio_batch_recognize_from_layout processes all images independently
# ----------------------------------------------------------------------
def test_aio_batch_recognize_from_layout_processes_all_images(monkeypatch):
    client = _make_client(monkeypatch)
    _install_async_fakes(client)

    image1 = Image.new("RGB", (20, 10), (255, 255, 255))
    image2 = Image.new("RGB", (30, 15), (255, 255, 255))

    layout1 = client.layout_detect(image1)
    layout2 = client.layout_detect(image2)

    results = asyncio.run(
        client.aio_batch_recognize_from_layout(
            [image1, image2],
            [layout1, layout2],
        )
    )

    assert len(results) == 2
    for result in results:
        assert isinstance(result, ExtractResult)
        assert len(result) == 2
        assert result[0].content is not None
        assert result[1].content is not None


# ----------------------------------------------------------------------
# Test 4: aio_recognize_from_layout matches aio_two_step_extract
# ----------------------------------------------------------------------
def test_aio_recognize_from_layout_matches_aio_two_step_extract(monkeypatch):
    client = _make_client(monkeypatch)
    _install_async_fakes(client)
    image = Image.new("RGB", (20, 10), (255, 255, 255))

    full_result = asyncio.run(client.aio_two_step_extract(image))

    layout_result = asyncio.run(client.aio_layout_detect(image))
    recognize_result = asyncio.run(
        client.aio_recognize_from_layout(image, layout_result)
    )

    assert len(full_result) == len(recognize_result)
    for full_block, recog_block in zip(full_result, recognize_result):
        assert full_block.content == recog_block.content


# ----------------------------------------------------------------------
# Test 5: batch_recognize_from_layout processes all images
# ----------------------------------------------------------------------
def test_batch_recognize_from_layout_processes_all_images(monkeypatch):
    client = _make_client(monkeypatch)
    _install_sync_fakes(client)

    image1 = Image.new("RGB", (20, 10), (255, 255, 255))
    image2 = Image.new("RGB", (30, 15), (255, 255, 255))

    layout1 = client.layout_detect(image1)
    layout2 = client.layout_detect(image2)

    results = client.batch_recognize_from_layout(
        [image1, image2],
        [layout1, layout2],
    )

    assert len(results) == 2
    for result in results:
        assert isinstance(result, ExtractResult)
        assert len(result) == 2
        assert result[0].content is not None
        assert result[1].content is not None
