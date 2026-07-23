"""Browser-isolated persistent history for the V31 research website."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any


class HistoryStoreError(ValueError):
    """Raised when a history owner or batch identifier is invalid."""


_BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,96}$")


class HistoryStore:
    """Persist complete batches below a one-way browser-owner namespace.

    The raw browser token is never written to disk or exposed in filenames.
    This is browser-level privacy isolation rather than authenticated user
    accounts, which is appropriate for the temporary local research demo.
    """

    def __init__(self, root_dir: Path, *, max_batches_per_owner: int = 50) -> None:
        if max_batches_per_owner < 1:
            raise ValueError("max_batches_per_owner must be positive")
        self.root_dir = Path(root_dir).resolve()
        self.max_batches_per_owner = int(max_batches_per_owner)

    @staticmethod
    def _owner_hash(owner_token: str) -> str:
        if not isinstance(owner_token, str) or not owner_token.strip():
            raise HistoryStoreError("浏览器历史标识尚未建立，请刷新页面后重试")
        if len(owner_token) > 256:
            raise HistoryStoreError("浏览器历史标识无效")
        return hashlib.sha256(owner_token.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _batch_id(batch_id: str) -> str:
        if not isinstance(batch_id, str) or not _BATCH_ID_PATTERN.fullmatch(batch_id):
            raise HistoryStoreError("历史批次编号无效")
        return batch_id

    def _owner_dir(self, owner_token: str) -> Path:
        return self.root_dir / self._owner_hash(owner_token)

    def _path(self, owner_token: str, batch_id: str) -> Path:
        return self._owner_dir(owner_token) / f"{self._batch_id(batch_id)}.json"

    @staticmethod
    def _metadata(batch_state: dict[str, Any]) -> dict[str, Any]:
        jobs = list(batch_state.get("jobs", []))
        results = dict(batch_state.get("results", {}))
        completed = sum(job.get("status") == "已完成" for job in jobs)
        failed = sum(job.get("status") == "失败" for job in jobs)
        names = [str(job.get("display_name", "未命名序列")) for job in jobs]
        return {
            "batch_id": batch_state["batch_id"],
            "created_at": batch_state.get("created_at"),
            "completed_at": batch_state.get("completed_at"),
            "sequence_count": len(jobs),
            "completed_count": completed,
            "failed_count": failed,
            "rejected_count": len(batch_state.get("rejected_inputs", [])),
            "prediction_count": sum(
                int(result.get("endpoint_record_count", len(result.get("endpoints", []))))
                for result in results.values()
            ),
            "sequence_names": names,
        }

    def save(self, owner_token: str, batch_state: dict[str, Any]) -> Path:
        """Atomically save a complete batch and enforce per-browser retention."""

        batch_id = self._batch_id(str(batch_state.get("batch_id", "")))
        owner_dir = self._owner_dir(owner_token)
        owner_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "peptide_omnipanel_v31_web_history_v1",
            "metadata": self._metadata(batch_state),
            "batch_state": batch_state,
        }
        target = owner_dir / f"{batch_id}.json"
        temporary = owner_dir / f".{batch_id}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        self._prune(owner_token)
        return target

    def _prune(self, owner_token: str) -> None:
        owner_dir = self._owner_dir(owner_token)
        files = sorted(
            owner_dir.glob("*.json"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
        for stale in files[self.max_batches_per_owner :]:
            stale.unlink(missing_ok=True)

    def list(self, owner_token: str) -> list[dict[str, Any]]:
        """Return newest-first batch metadata for one browser only."""

        owner_dir = self._owner_dir(owner_token)
        if not owner_dir.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for path in owner_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                metadata = dict(payload["metadata"])
                self._batch_id(str(metadata["batch_id"]))
                metadata["stored_bytes"] = path.stat().st_size
                rows.append(metadata)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
        return sorted(
            rows,
            key=lambda row: (str(row.get("created_at") or ""), str(row["batch_id"])),
            reverse=True,
        )

    def load(self, owner_token: str, batch_id: str) -> dict[str, Any]:
        path = self._path(owner_token, batch_id)
        if not path.is_file():
            raise HistoryStoreError("未找到该浏览器下的历史批次")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            batch_state = dict(payload["batch_state"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise HistoryStoreError("历史批次文件损坏或不完整") from error
        if batch_state.get("batch_id") != batch_id:
            raise HistoryStoreError("历史批次编号不一致")
        return batch_state

    def delete(self, owner_token: str, batch_id: str) -> bool:
        path = self._path(owner_token, batch_id)
        if not path.is_file():
            return False
        path.unlink()
        return True
