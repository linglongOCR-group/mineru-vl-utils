import json
from pathlib import Path

from PIL import Image

from mineru_vl_utils.dissection import DissectionRecorder
from mineru_vl_utils.structs import ContentBlock


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_dissection_recorder_writes_layout_crops_recognition_and_manifest(tmp_path):
    recorder = DissectionRecorder(tmp_path / "dissection", document_stem="doc")
    image = Image.new("RGB", (20, 10), (255, 255, 255))
    blocks = [
        ContentBlock("text", [0.0, 0.0, 0.5, 1.0]),
        ContentBlock("table", [0.5, 0.0, 1.0, 1.0]),
    ]

    ids = recorder.record_layout(page_idx=0, blocks=blocks, image_size=image.size)
    recorder.save_layout_crops(page_idx=0, image=image, blocks=blocks, bbox_ids=ids)
    recorder.record_recognition_requested(
        ids[0],
        prompt="Text Recognition:",
        endpoint="http://recognition/v1/chat/completions",
        model_name="mineru-vlm",
    )
    recorder.record_recognized(ids[0], content="hello", duration_seconds=0.25)
    recorder.record_failed(
        ids[1],
        stage="recognition",
        error=RuntimeError("bad response"),
        crop_path="crops/page_000_bbox_001.png",
        http_status_code=502,
    )
    recorder.finalize()

    assert (tmp_path / "dissection" / "layout.json").exists()
    assert (tmp_path / "dissection" / "crops" / "page_000_bbox_000.png").exists()
    assert (tmp_path / "dissection" / "crops" / "page_000_bbox_001.png").exists()
    assert (tmp_path / "dissection" / "recognition" / "page_000_bbox_000.json").exists()
    assert (tmp_path / "dissection" / "recognition" / "page_000_bbox_001.json").exists()

    layout = _read_json(tmp_path / "dissection" / "layout.json")
    assert layout["document_stem"] == "doc"
    assert layout["pages"][0]["blocks"][0]["bbox_id"] == "page_000_bbox_000"

    recognized = _read_json(tmp_path / "dissection" / "recognition" / "page_000_bbox_000.json")
    assert recognized["status"] == "recognized"
    assert recognized["content"] == "hello"

    failed = _read_json(tmp_path / "dissection" / "recognition" / "page_000_bbox_001.json")
    assert failed["status"] == "failed"
    assert failed["exception_type"] == "RuntimeError"
    assert failed["http_status_code"] == 502

    errors = (tmp_path / "dissection" / "errors.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(errors) == 1
    assert json.loads(errors[0])["bbox_id"] == "page_000_bbox_001"

    manifest = _read_json(tmp_path / "dissection" / "manifest.json")
    assert manifest["status_counts"] == {"recognized": 1, "failed": 1}
    assert manifest["records"][0]["transitions"] == [
        "detected",
        "cropped",
        "recognition_requested",
        "recognized",
    ]
