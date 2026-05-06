from mineru_vl_utils import mineru_client as mineru_client_module
from mineru_vl_utils.structs import ContentBlock


class FakeVlmClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.predict_calls = []
        self.batch_predict_calls = []

    def predict(self, image, prompt, params=None, priority=None):
        self.predict_calls.append((image, prompt, params, priority))
        return f"{self.server_url}:layout"

    def batch_predict(self, images, prompts, params=None, priority=None):
        self.batch_predict_calls.append((images, prompts, params, priority))
        return [f"{self.server_url}:recognition" for _ in images]


def _build_client(monkeypatch, **kwargs):
    created = []

    def fake_new_vlm_client(**client_kwargs):
        client = FakeVlmClient(client_kwargs.get("server_url"))
        created.append(client)
        return client

    monkeypatch.setattr(mineru_client_module, "new_vlm_client", fake_new_vlm_client)
    client = mineru_client_module.MinerUClient(
        use_tqdm=False,
        **kwargs,
    )
    return client, created


def test_http_client_routes_layout_and_recognition_to_stage_urls(monkeypatch):
    client, created = _build_client(
        monkeypatch,
        backend="http-client",
        layout_server_url="http://layout",
        recognition_server_url="http://recognition",
    )
    assert [item.server_url for item in created] == ["http://recognition", "http://layout"]

    client.helper.prepare_for_layout = lambda image: "layout-image"
    client.helper.parse_layout_output = lambda output: [ContentBlock("text", [0.0, 0.0, 1.0, 1.0])]
    client.helper.prepare_for_extract = lambda image, blocks, not_extract_list=None: (
        ["recognition-image"],
        ["recognition-prompt"],
        [None],
        [0],
    )
    client.helper.post_process = lambda blocks: blocks

    result = client.two_step_extract(image="page")

    assert result[0].content == "http://recognition:recognition"
    assert client.layout_client.predict_calls == [("layout-image", "\nLayout Detection:", None, None)]
    assert client.client.batch_predict_calls == [
        (["recognition-image"], ["recognition-prompt"], [None], None)
    ]


def test_http_client_reuses_shared_server_url_for_both_stages(monkeypatch):
    client, created = _build_client(
        monkeypatch,
        backend="http-client",
        server_url="http://shared",
    )

    assert [item.server_url for item in created] == ["http://shared"]
    assert client.layout_client is client.client


def test_http_client_uses_stage_environment_fallbacks(monkeypatch):
    monkeypatch.setenv("MINERU_VL_LAYOUT_SERVER", "http://layout-env")
    monkeypatch.setenv("MINERU_VL_RECOGNITION_SERVER", "http://recognition-env")

    client, created = _build_client(monkeypatch, backend="http-client")

    assert [item.server_url for item in created] == ["http://recognition-env", "http://layout-env"]
    assert client.client.server_url == "http://recognition-env"
    assert client.layout_client.server_url == "http://layout-env"


def test_non_http_backend_uses_one_shared_client(monkeypatch):
    client, created = _build_client(
        monkeypatch,
        backend="transformers",
        model=object(),
        processor=object(),
        layout_server_url="http://layout",
        recognition_server_url="http://recognition",
    )

    assert len(created) == 1
    assert client.layout_client is client.client
