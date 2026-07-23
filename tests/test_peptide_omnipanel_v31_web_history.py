from __future__ import annotations

from pathlib import Path

import pytest

from peptide_omnipanel.v31_web_history import HistoryStore, HistoryStoreError


def _state(batch_id: str, name: str = "候选肽") -> dict:
    return {
        "schema_version": "peptide_omnipanel_v31_web_batch_state_v1",
        "batch_id": batch_id,
        "created_at": f"2026-07-23T00:00:0{batch_id[-1]}+00:00",
        "completed_at": f"2026-07-23T00:00:1{batch_id[-1]}+00:00",
        "jobs": [
            {
                "job_id": f"任务-{batch_id}",
                "display_name": name,
                "sequence": "ACDEFGHIK",
                "source": "文本输入",
                "status": "已完成",
            }
        ],
        "rejected_inputs": [],
        "results": {
            f"任务-{batch_id}": {
                "endpoint_record_count": 18,
                "endpoints": [{"endpoint_id": "BBB"}],
            }
        },
    }


def test_history_is_atomic_persistent_and_browser_isolated(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path)
    target = store.save("owner-a", _state("batch-1"))
    assert target.is_file()
    assert "owner-a" not in str(target)
    assert not list(target.parent.glob("*.tmp"))
    assert store.list("owner-a")[0]["sequence_names"] == ["候选肽"]
    assert store.load("owner-a", "batch-1")["jobs"][0]["sequence"] == "ACDEFGHIK"
    assert store.list("owner-b") == []
    with pytest.raises(HistoryStoreError, match="未找到"):
        store.load("owner-b", "batch-1")


def test_history_is_newest_first_and_retention_is_bounded(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path, max_batches_per_owner=2)
    store.save("owner", _state("batch-1"))
    store.save("owner", _state("batch-2"))
    store.save("owner", _state("batch-3"))
    assert [row["batch_id"] for row in store.list("owner")] == ["batch-3", "batch-2"]
    assert store.delete("owner", "batch-2") is True
    assert store.delete("owner", "batch-2") is False


def test_history_rejects_unsafe_identifiers(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path)
    with pytest.raises(HistoryStoreError):
        store.list("")
    with pytest.raises(HistoryStoreError):
        store.load("owner", "../other")
