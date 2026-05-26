from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from .structs import ContentBlock
from .vlm_client.utils import get_rgb_image


def _json_default(value: Any):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_status_code(error: BaseException) -> int | None:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    status_code = getattr(error, "status_code", None)
    return status_code if isinstance(status_code, int) else None


class DissectionRecorder:
    def __init__(self, root_dir: str | Path, document_stem: str | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.document_stem = document_stem
        self.crops_dir = self.root_dir / "crops"
        self.recognition_dir = self.root_dir / "recognition"
        self.layout_path = self.root_dir / "layout.json"
        self.errors_path = self.root_dir / "errors.jsonl"
        self.manifest_path = self.root_dir / "manifest.json"
        self.created_at = _utc_timestamp()
        self.pages: dict[int, dict[str, Any]] = {}
        self.records: dict[str, dict[str, Any]] = {}
        self._request_starts: dict[str, float] = {}
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.recognition_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def bbox_id(page_idx: int, bbox_idx: int) -> str:
        return f"page_{page_idx:03d}_bbox_{bbox_idx:03d}"

    @staticmethod
    def _relative(path: Path) -> str:
        return path.as_posix()

    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

    def _record_transition(self, record: dict[str, Any], status: str) -> None:
        transitions = record.setdefault("transitions", [])
        if not transitions or transitions[-1] != status:
            transitions.append(status)
        record["status"] = status
        record["updated_at"] = _utc_timestamp()

    def _block_payload(self, block: ContentBlock, bbox_id: str, page_idx: int, bbox_idx: int) -> dict[str, Any]:
        payload = {
            "bbox_id": bbox_id,
            "page_idx": page_idx,
            "bbox_index": bbox_idx,
            "type": block.get("type"),
            "bbox": block.get("bbox"),
            "angle": block.get("angle"),
            "index": block.get("index", bbox_idx),
        }
        if "merge_prev" in block:
            payload["merge_prev"] = block["merge_prev"]
        if block.get("scored") is not None:
            payload["scored"] = block["scored"]
        return payload

    def record_layout(
        self,
        page_idx: int,
        blocks: Sequence[ContentBlock],
        image_size: tuple[int, int] | None = None,
    ) -> dict[int, str]:
        bbox_ids: dict[int, str] = {}
        page_blocks = []
        for bbox_idx, block in enumerate(blocks):
            bbox_id = self.bbox_id(page_idx, bbox_idx)
            bbox_ids[bbox_idx] = bbox_id
            payload = self._block_payload(block, bbox_id, page_idx, bbox_idx)
            page_blocks.append(payload)
            record = self.records.setdefault(bbox_id, dict(payload))
            record.update(payload)
            self._record_transition(record, "detected")

        self.pages[page_idx] = {
            "page_idx": page_idx,
            "image_size": list(image_size) if image_size else None,
            "blocks": page_blocks,
        }
        self._flush_layout()
        self._flush_manifest()
        return bbox_ids

    def _flush_layout(self) -> None:
        self._write_json(
            self.layout_path,
            {
                "document_stem": self.document_stem,
                "created_at": self.created_at,
                "updated_at": _utc_timestamp(),
                "pages": [self.pages[key] for key in sorted(self.pages)],
            },
        )

    def _crop_path(self, bbox_id: str) -> Path:
        return self.crops_dir / f"{bbox_id}.png"

    def _recognition_path(self, bbox_id: str) -> Path:
        return self.recognition_dir / f"{bbox_id}.json"

    def _save_pil_crop(self, image: Image.Image, bbox: Sequence[float], path: Path) -> None:
        rgb = get_rgb_image(image)
        width, height = rgb.size
        x1, y1, x2, y2 = bbox
        crop = rgb.crop((x1 * width, y1 * height, x2 * width, y2 * height))
        crop.save(path, format="PNG")

    def save_layout_crops(
        self,
        page_idx: int,
        image: Image.Image,
        blocks: Sequence[ContentBlock],
        bbox_ids: dict[int, str],
    ) -> None:
        for bbox_idx, block in enumerate(blocks):
            bbox_id = bbox_ids.get(bbox_idx)
            if not bbox_id:
                continue
            path = self._crop_path(bbox_id)
            try:
                self._save_pil_crop(image, block["bbox"], path)
                record = self.records[bbox_id]
                record["crop_path"] = self._relative(path.relative_to(self.root_dir))
                self._record_transition(record, "cropped")
            except Exception as exc:
                self.record_failed(
                    bbox_id,
                    stage="cropping",
                    error=exc,
                    crop_path=self._relative(path.relative_to(self.root_dir)),
                )
        self._flush_manifest()

    def save_prepared_crop(self, bbox_id: str, image: Image.Image | bytes) -> None:
        path = self._crop_path(bbox_id)
        if isinstance(image, bytes):
            path.write_bytes(image)
        else:
            get_rgb_image(image).save(path, format="PNG")
        record = self.records[bbox_id]
        record["crop_path"] = self._relative(path.relative_to(self.root_dir))
        self._record_transition(record, "cropped")
        self._flush_manifest()

    def record_skipped(self, bbox_id: str, reason: str) -> None:
        record = self.records[bbox_id]
        record["skip_reason"] = reason
        self._record_transition(record, "skipped")
        self._flush_manifest()

    def record_recognition_requested(
        self,
        bbox_id: str,
        prompt: str,
        endpoint: str | None = None,
        model_name: str | None = None,
    ) -> None:
        now = time.monotonic()
        self._request_starts[bbox_id] = now
        record = self.records[bbox_id]
        record.update(
            {
                "prompt": prompt,
                "endpoint": endpoint,
                "model_name": model_name,
                "request_start": _utc_timestamp(),
            }
        )
        self._record_transition(record, "recognition_requested")
        self._write_recognition_record(bbox_id, record)
        self._flush_manifest()

    def record_recognized(
        self,
        bbox_id: str,
        content: str,
        duration_seconds: float | None = None,
    ) -> None:
        record = self.records[bbox_id]
        record["request_end"] = _utc_timestamp()
        if duration_seconds is None:
            start = self._request_starts.pop(bbox_id, None)
            duration_seconds = time.monotonic() - start if start is not None else None
        if duration_seconds is not None:
            record["duration_seconds"] = round(duration_seconds, 6)
        record["content"] = content
        self._record_transition(record, "recognized")
        self._write_recognition_record(bbox_id, record)
        self._flush_manifest()

    def record_partial(
        self,
        bbox_id: str,
        partial_content: str,
        error: BaseException,
        http_status_code: int | None = None,
        retry_count: int | None = None,
        retry_backoff_factor: float | None = None,
    ) -> None:
        self._record_error_status(
            bbox_id,
            status="partial",
            stage="recognition",
            error=error,
            http_status_code=http_status_code,
            retry_count=retry_count,
            retry_backoff_factor=retry_backoff_factor,
            extra_payload={"partial_content": partial_content},
        )

    def record_failed(
        self,
        bbox_id: str,
        stage: str,
        error: BaseException,
        crop_path: str | None = None,
        http_status_code: int | None = None,
        retry_count: int | None = None,
        retry_backoff_factor: float | None = None,
    ) -> None:
        self._record_error_status(
            bbox_id,
            status="failed",
            stage=stage,
            error=error,
            crop_path=crop_path,
            http_status_code=http_status_code,
            retry_count=retry_count,
            retry_backoff_factor=retry_backoff_factor,
        )

    def _record_error_status(
        self,
        bbox_id: str,
        status: str,
        stage: str,
        error: BaseException,
        crop_path: str | None = None,
        http_status_code: int | None = None,
        retry_count: int | None = None,
        retry_backoff_factor: float | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        record = self.records.setdefault(
            bbox_id,
            {"bbox_id": bbox_id, "transitions": []},
        )
        record["request_end"] = _utc_timestamp()
        start = self._request_starts.pop(bbox_id, None)
        if start is not None:
            record["duration_seconds"] = round(time.monotonic() - start, 6)
        if retry_count is not None:
            record["retry_count"] = retry_count
        if retry_backoff_factor is not None:
            record["retry_backoff_factor"] = retry_backoff_factor
        if crop_path is not None:
            record["crop_path"] = crop_path
        status_code = http_status_code if http_status_code is not None else _safe_status_code(error)
        error_payload = {
            "bbox_id": bbox_id,
            "stage": stage,
            "crop_path": record.get("crop_path"),
            "exception_type": type(error).__name__,
            "error": str(error),
            "message": str(error),
            "http_status_code": status_code,
            "traceback": "".join(traceback.format_exception_only(type(error), error)).strip(),
        }
        if extra_payload:
            error_payload.update(extra_payload)
        record.update(error_payload)
        self._record_transition(record, status)
        self._write_recognition_record(bbox_id, record)
        with self.errors_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(error_payload, ensure_ascii=False, default=_json_default) + "\n")
        self._flush_manifest()

    def _write_recognition_record(self, bbox_id: str, record: dict[str, Any]) -> None:
        payload = dict(record)
        payload["recognition_path"] = self._relative(self._recognition_path(bbox_id).relative_to(self.root_dir))
        self._write_json(self._recognition_path(bbox_id), payload)

    def _flush_manifest(self) -> None:
        records = [self.records[key] for key in sorted(self.records)]
        status_counts: dict[str, int] = {}
        for record in records:
            status = record.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        self._write_json(
            self.manifest_path,
            {
                "document_stem": self.document_stem,
                "created_at": self.created_at,
                "updated_at": _utc_timestamp(),
                "total_blocks": len(records),
                "status_counts": status_counts,
                "records": records,
            },
        )

    def finalize(self) -> None:
        if self.pages:
            self._flush_layout()
        self._flush_manifest()
