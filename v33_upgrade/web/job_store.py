"""Persistent client-scoped async job queue & task center storage for Peptide OmniPanel.

Supports:
  - Background chunked batch execution (4 molecules/chunk) with real-time progress updates & ETA
  - Graceful mid-run cancellation preserving already-completed results
  - Single-molecule prediction history logging into the same unified task center
  - Client-level isolation via `X-Client-Id` (browser UUID) stored in lightweight SQLite
  - Summary CSV, Full 46-Endpoint Wide CSV, and JSON export generators
"""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .adapter import OmniPanelAdapter
from .schemas import AssayConditions, BatchMoleculeResult, BatchParsedRow, PredictResponse

DEFAULT_JOBS_DIR = Path(
    os.environ.get(
        "OMNIPANEL_JOBS_DIR",
        os.environ.get("V33_JOBS_DIR", str(Path(__file__).resolve().parent.parent / "jobs_store")),
    )
)
MAX_JOBS_PER_CLIENT = 100
CHUNK_SIZE = 4


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_job_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M")
    suffix = uuid.uuid4().hex[:4].upper()
    return f"T-{stamp}-{suffix}"


def _compute_batch_analytics(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in results if r.get("status") == "ok"]
    n_ok = len(ok_rows)
    n_anchor = sum(1 for r in ok_rows if (r.get("mechanism_anchor") or {}).get("detected"))
    n_cyclic = sum(
        1
        for r in ok_rows
        if any("macrocycl" in m.lower() or "cyclic" in m.lower() or "disulfide" in m.lower() for m in (r.get("modifications") or []))
    )
    n_low_hem = 0
    n_high_hem = 0
    n_high_perm = 0
    n_long_hl = 0
    mws: list[float] = []

    for r in ok_rows:
        if r.get("mw") is not None:
            mws.append(float(r["mw"]))
        km = r.get("key_metrics") or {}
        hval = km.get("hemolysis_risk_val")
        if hval is not None:
            if float(hval) < 0.25:
                n_low_hem += 1
            elif float(hval) >= 0.50:
                n_high_hem += 1
        pval = km.get("pampa_val")
        cval = km.get("caco2_val")
        if (pval is not None and float(pval) >= 1e-6) or (cval is not None and float(cval) >= 1e-6):
            n_high_perm += 1
        hl = km.get("half_life_val")
        if hl is not None and float(hl) >= 12.0:
            n_long_hl += 1

    mws.sort()
    median_mw = round(mws[len(mws) // 2], 1) if mws else None
    ep_count = len(ok_rows[0].get("predictions") or []) if ok_rows else 0
    return {
        "n_ok": n_ok,
        "n_acylation_anchors": n_anchor,
        "n_cyclic_or_disulfide": n_cyclic,
        "n_low_hemolysis": n_low_hem,
        "n_high_hemolysis": n_high_hem,
        "n_high_permeability": n_high_perm,
        "n_long_half_life": n_long_hl,
        "median_mw": median_mw,
        "endpoint_count": ep_count,
    }


class JobManager:
    """SQLite-backed thread-safe async task manager for single and batch peptide predictions."""

    def __init__(self, adapter: OmniPanelAdapter, db_dir: Optional[Path] = None) -> None:
        self.adapter = adapter
        self.db_dir = Path(db_dir) if db_dir is not None else DEFAULT_JOBS_DIR
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_dir / "tasks.sqlite"
        self._lock = threading.Lock()
        self._cancel_flags: dict[str, threading.Event] = {}
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="omnipanel-job")
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._lock, self._connect() as con:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    include_tier2 INTEGER NOT NULL DEFAULT 1,
                    conditions_json TEXT NOT NULL,
                    total INTEGER NOT NULL DEFAULT 0,
                    completed INTEGER NOT NULL DEFAULT 0,
                    succeeded INTEGER NOT NULL DEFAULT 0,
                    failed INTEGER NOT NULL DEFAULT 0,
                    current_item TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    runtime_ms REAL DEFAULT 0.0,
                    error TEXT,
                    summary_json TEXT,
                    results_json TEXT
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_jobs_client_created ON jobs(client_id, created_at DESC)")
            # Mark any stale running/queued jobs from an unexpected process restart as interrupted
            con.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    error = COALESCE(error, 'Service restarted while task was running'),
                    finished_at = COALESCE(finished_at, ?)
                WHERE status IN ('queued', 'running')
                """,
                (_utc_iso(),),
            )
            con.commit()

    def _prune_old_jobs(self, con: sqlite3.Connection, client_id: str) -> None:
        rows = con.execute(
            "SELECT job_id FROM jobs WHERE client_id = ? ORDER BY created_at DESC LIMIT -1 OFFSET ?",
            (client_id, MAX_JOBS_PER_CLIENT),
        ).fetchall()
        if rows:
            con.executemany("DELETE FROM jobs WHERE job_id = ?", [(r["job_id"],) for r in rows])

    def submit_batch_job(
        self,
        client_id: str,
        parsed_rows: list[BatchParsedRow],
        include_tier2: bool,
        conditions: AssayConditions,
        title: Optional[str] = None,
    ) -> dict[str, Any]:
        job_id = _new_job_id()
        clean_title = (title or "").strip() or f"Batch Screening ({len(parsed_rows)})"
        now = _utc_iso()
        cond_json = json.dumps(conditions.model_dump(), ensure_ascii=False)

        cancel_evt = threading.Event()
        with self._lock:
            self._cancel_flags[job_id] = cancel_evt
            with self._connect() as con:
                con.execute(
                    """
                    INSERT INTO jobs (
                        job_id, client_id, job_type, title, status, include_tier2,
                        conditions_json, total, completed, succeeded, failed,
                        current_item, created_at, summary_json, results_json
                    ) VALUES (?, ?, 'batch', ?, 'queued', ?, ?, ?, 0, 0, 0, ?, ?, '{}', '[]')
                    """,
                    (
                        job_id,
                        client_id,
                        clean_title,
                        1 if include_tier2 else 0,
                        cond_json,
                        len(parsed_rows),
                        parsed_rows[0].name if parsed_rows else None,
                        now,
                    ),
                )
                self._prune_old_jobs(con, client_id)
                con.commit()

        self._executor.submit(
            self._run_batch_job,
            job_id,
            parsed_rows,
            include_tier2,
            conditions,
            cancel_evt,
        )
        return self.get_job_meta(job_id, client_id) or {}

    def _run_batch_job(
        self,
        job_id: str,
        parsed_rows: list[BatchParsedRow],
        include_tier2: bool,
        conditions: AssayConditions,
        cancel_evt: threading.Event,
    ) -> None:
        t0 = time.perf_counter()
        self._update_job(job_id, status="running", started_at=_utc_iso())
        results_dicts: list[dict[str, Any]] = []
        succeeded = 0
        failed = 0

        try:
            for start in range(0, len(parsed_rows), CHUNK_SIZE):
                if cancel_evt.is_set():
                    elapsed = round((time.perf_counter() - t0) * 1000.0, 1)
                    self._update_job(
                        job_id,
                        status="cancelled",
                        completed=len(results_dicts),
                        succeeded=succeeded,
                        failed=failed,
                        current_item=None,
                        finished_at=_utc_iso(),
                        runtime_ms=elapsed,
                        summary_json=json.dumps(_compute_batch_analytics(results_dicts), ensure_ascii=False),
                        results_json=json.dumps(results_dicts, ensure_ascii=False),
                    )
                    return

                chunk = parsed_rows[start : start + CHUNK_SIZE]
                self._update_job(job_id, current_item=chunk[0].name)

                chunk_res: list[BatchMoleculeResult] = self.adapter.predict_batch(
                    chunk,
                    include_tier2=include_tier2,
                    conditions=conditions,
                )
                for item in chunk_res:
                    d = item.model_dump()
                    results_dicts.append(d)
                    if item.status == "ok":
                        succeeded += 1
                    else:
                        failed += 1

                elapsed = round((time.perf_counter() - t0) * 1000.0, 1)
                self._update_job(
                    job_id,
                    completed=len(results_dicts),
                    succeeded=succeeded,
                    failed=failed,
                    runtime_ms=elapsed,
                    summary_json=json.dumps(_compute_batch_analytics(results_dicts), ensure_ascii=False),
                    results_json=json.dumps(results_dicts, ensure_ascii=False),
                )

            elapsed = round((time.perf_counter() - t0) * 1000.0, 1)
            final_status = "completed" if failed == 0 else ("partial" if succeeded > 0 else "failed")
            self._update_job(
                job_id,
                status=final_status,
                completed=len(results_dicts),
                succeeded=succeeded,
                failed=failed,
                current_item=None,
                finished_at=_utc_iso(),
                runtime_ms=elapsed,
                summary_json=json.dumps(_compute_batch_analytics(results_dicts), ensure_ascii=False),
                results_json=json.dumps(results_dicts, ensure_ascii=False),
            )
        except Exception as exc:
            elapsed = round((time.perf_counter() - t0) * 1000.0, 1)
            self._update_job(
                job_id,
                status="failed",
                error=str(exc),
                current_item=None,
                finished_at=_utc_iso(),
                runtime_ms=elapsed,
                completed=len(results_dicts),
                succeeded=succeeded,
                failed=failed,
                summary_json=json.dumps(_compute_batch_analytics(results_dicts), ensure_ascii=False),
                results_json=json.dumps(results_dicts, ensure_ascii=False),
            )
        finally:
            with self._lock:
                self._cancel_flags.pop(job_id, None)

    def record_single_prediction(
        self,
        client_id: str,
        pred: PredictResponse,
        include_tier2: bool,
        conditions: AssayConditions,
    ) -> str:
        """Record a completed single-molecule prediction in the client's task history."""
        job_id = _new_job_id()
        row = BatchParsedRow(
            index=1,
            name=pred.query_name,
            detected_type=pred.resolved_type,
            sequence=pred.sequence,
            smiles=pred.smiles,
            valid=True,
        )
        mol_res = self.adapter._to_batch_molecule_result(row, pred).model_dump()
        results_list = [mol_res]
        now = _utc_iso()
        with self._lock, self._connect() as con:
            con.execute(
                """
                INSERT INTO jobs (
                    job_id, client_id, job_type, title, status, include_tier2,
                    conditions_json, total, completed, succeeded, failed,
                    current_item, created_at, started_at, finished_at, runtime_ms,
                    summary_json, results_json
                ) VALUES (?, ?, 'single', ?, 'completed', ?, ?, 1, 1, 1, 0, NULL, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    client_id,
                    pred.query_name,
                    1 if include_tier2 else 0,
                    json.dumps(conditions.model_dump(), ensure_ascii=False),
                    now,
                    now,
                    now,
                    float(pred.runtime_ms),
                    json.dumps(_compute_batch_analytics(results_list), ensure_ascii=False),
                    json.dumps(results_list, ensure_ascii=False),
                ),
            )
            self._prune_old_jobs(con, client_id)
            con.commit()
        return job_id

    def _update_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields.keys())
        vals = list(fields.values()) + [job_id]
        with self._lock, self._connect() as con:
            con.execute(f"UPDATE jobs SET {cols} WHERE job_id = ?", vals)
            con.commit()

    @staticmethod
    def _row_to_meta(row: sqlite3.Row) -> dict[str, Any]:
        total = int(row["total"] or 0)
        completed = int(row["completed"] or 0)
        pct = round((completed / total) * 100.0, 1) if total > 0 else 0.0
        if row["status"] in ("completed", "partial") and total > 0:
            pct = 100.0
        eta_s: Optional[float] = None
        if row["status"] == "running" and completed > 0 and total > completed and row["runtime_ms"]:
            per_item_s = (float(row["runtime_ms"]) / 1000.0) / completed
            eta_s = round(per_item_s * (total - completed), 1)
        return {
            "job_id": row["job_id"],
            "job_type": row["job_type"],
            "title": row["title"],
            "status": row["status"],
            "include_tier2": bool(row["include_tier2"]),
            "conditions": json.loads(row["conditions_json"] or "{}"),
            "total": total,
            "completed": completed,
            "succeeded": int(row["succeeded"] or 0),
            "failed": int(row["failed"] or 0),
            "progress_pct": pct,
            "current_item": row["current_item"],
            "eta_seconds": eta_s,
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "runtime_ms": round(float(row["runtime_ms"] or 0.0), 1),
            "error": row["error"],
            "summary": json.loads(row["summary_json"] or "{}"),
        }

    def list_jobs(self, client_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, self._connect() as con:
            rows = con.execute(
                """
                SELECT job_id, job_type, title, status, include_tier2, conditions_json,
                       total, completed, succeeded, failed, current_item,
                       created_at, started_at, finished_at, runtime_ms, error, summary_json
                FROM jobs
                WHERE client_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (client_id, limit),
            ).fetchall()
        return [self._row_to_meta(r) for r in rows]

    def get_job_meta(self, job_id: str, client_id: str) -> Optional[dict[str, Any]]:
        with self._lock, self._connect() as con:
            row = con.execute(
                """
                SELECT job_id, job_type, title, status, include_tier2, conditions_json,
                       total, completed, succeeded, failed, current_item,
                       created_at, started_at, finished_at, runtime_ms, error, summary_json
                FROM jobs WHERE job_id = ? AND client_id = ?
                """,
                (job_id, client_id),
            ).fetchone()
        return self._row_to_meta(row) if row else None

    def get_job_detail(self, job_id: str, client_id: str) -> Optional[dict[str, Any]]:
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT * FROM jobs WHERE job_id = ? AND client_id = ?",
                (job_id, client_id),
            ).fetchone()
        if not row:
            return None
        meta = self._row_to_meta(row)
        meta["results"] = json.loads(row["results_json"] or "[]")
        return meta

    def cancel_job(self, job_id: str, client_id: str) -> Optional[dict[str, Any]]:
        meta = self.get_job_meta(job_id, client_id)
        if not meta:
            return None
        if meta["status"] in ("queued", "running"):
            with self._lock:
                evt = self._cancel_flags.get(job_id)
                if evt is not None:
                    evt.set()
            if meta["status"] == "queued":
                self._update_job(job_id, status="cancelled", finished_at=_utc_iso(), current_item=None)
        return self.get_job_meta(job_id, client_id)

    def delete_job(self, job_id: str, client_id: str) -> bool:
        with self._lock:
            evt = self._cancel_flags.get(job_id)
            if evt is not None:
                evt.set()
            with self._connect() as con:
                cur = con.execute("DELETE FROM jobs WHERE job_id = ? AND client_id = ?", (job_id, client_id))
                con.commit()
                return cur.rowcount > 0

    def export_job_csv(self, job_detail: dict[str, Any], full_wide: bool = False) -> str:
        results: list[dict[str, Any]] = job_detail.get("results") or []
        buf = io.StringIO()
        writer = csv.writer(buf)

        def _row_source(r: dict[str, Any]) -> str:
            if r.get("prediction_source"):
                return str(r["prediction_source"])
            preds = r.get("predictions") or []
            if preds:
                srcs = sorted({str(p.get("prediction_source") or "unknown") for p in preds})
                return "|".join(srcs)
            return "unknown"

        if not full_wide:
            writer.writerow(
                [
                    "index",
                    "name",
                    "status",
                    "prediction_source",
                    "mw",
                    "logp",
                    "net_charge_ph74",
                    "half_life",
                    "ppb",
                    "pampa",
                    "caco2",
                    "efflux_ratio",
                    "hemolysis_risk",
                    "hc50",
                    "cytotoxicity_cc50",
                    "logd",
                    "solubility",
                ]
            )
            for r in results:
                km = r.get("key_metrics") or {}
                writer.writerow(
                    [
                        r.get("index", ""),
                        r.get("name", ""),
                        r.get("status", ""),
                        _row_source(r),
                        r.get("mw", ""),
                        r.get("logp", ""),
                        r.get("net_charge_ph74", ""),
                        km.get("half_life", ""),
                        km.get("ppb", ""),
                        km.get("pampa", ""),
                        km.get("caco2", ""),
                        km.get("efflux_ratio", ""),
                        km.get("hemolysis_risk", ""),
                        km.get("hc50", ""),
                        km.get("cytotoxicity_cc50", ""),
                        km.get("logd", ""),
                        km.get("solubility", ""),
                    ]
                )
            return buf.getvalue()

        # Full wide matrix: every predicted endpoint value + 80% interval + AD + UQ reliability across all molecules
        ep_order: list[str] = []
        seen: set[str] = set()
        for r in results:
            for p in r.get("predictions") or []:
                ep = p.get("endpoint")
                if ep and ep not in seen:
                    seen.add(ep)
                    ep_order.append(ep)

        header = ["index", "name", "status", "prediction_source", "mw", "logp", "net_charge_ph74", "acylation_anchor"]
        for ep in ep_order:
            header.extend(
                [
                    f"{ep}_value",
                    f"{ep}_display",
                    f"{ep}_ci80_low",
                    f"{ep}_ci80_high",
                    f"{ep}_ad",
                    f"{ep}_max_tc",
                    f"{ep}_uq_reliability",
                ]
            )
        writer.writerow(header)

        for r in results:
            anch = r.get("mechanism_anchor") or {}
            anch_str = f"{anch['range_h'][0]}-{anch['range_h'][1]}h" if anch.get("detected") and anch.get("range_h") else ""
            by_ep = {p.get("endpoint"): p for p in (r.get("predictions") or [])}
            row = [
                r.get("index", ""),
                r.get("name", ""),
                r.get("status", ""),
                _row_source(r),
                r.get("mw", ""),
                r.get("logp", ""),
                r.get("net_charge_ph74", ""),
                anch_str,
            ]
            for ep in ep_order:
                p = by_ep.get(ep) or {}
                row.extend(
                    [
                        p.get("value", ""),
                        p.get("value_display", ""),
                        p.get("ci80_low", ""),
                        p.get("ci80_high", ""),
                        p.get("ad_status", ""),
                        p.get("max_tanimoto", ""),
                        p.get("uq_reliability", ""),
                    ]
                )
            writer.writerow(row)
        return buf.getvalue()
