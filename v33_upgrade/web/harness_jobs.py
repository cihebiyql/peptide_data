"""Dedicated asynchronous job manager for Peptide-Harness multi-round optimization.

Key safety & production concurrency guarantees (Red Line 6):
  - Isolated `ThreadPoolExecutor(max_workers=1)` so background multi-round optimization
    never competes with Task Center batch workers or saturates CPU inference.
  - Interactive yield gate: before every remote chunk (<=4 items), waits until ALL
    interactive inference slots are idle and no interactive caller is queued, and
    caps per-chunk HTTP timeout to remaining `wall_clock_budget_s`.
  - Strict client identity (`X-Client-Id` required) + source IP concurrency limit (1 active
    Harness job per `client_id` AND per `client_ip`, global queue cap 8) checked BEFORE
    expensive RDKit preflight to prevent abuse and forged-client-ID queue flooding.
  - Per-client retention partitioning + protection grace period so one caller cannot
    evict another user's completed results by spamming submit+cancel cycles.
  - Strict ownership isolation on `GET` and `DELETE` (cross-client returns `None` -> HTTP 404).
  - Cooperative mid-round/mid-chunk cancellation (`DELETE`) preserving already-evaluated candidates.
  - Zero SSRF: LLM proposer hook is strictly disabled by default (`V35_HARNESS_LLM_ENABLED=0`)
    and never accepts client-supplied URLs or keys.
"""

from __future__ import annotations

import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .adapter import EngineBusyError, EngineUnavailableError, OmniPanelAdapter
from .peptide_harness import (
    HarnessConfig,
    preflight_validate_harness_request,
    run_peptide_harness,
)
from .schemas import AssayConditions, BatchMoleculeResult, BatchParsedRow

_INTERNAL_URL_RE = re.compile(r"https?://(?:127\.0\.0\.1|localhost|\[::1\]|10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+)(?::\d+)?\S*", re.IGNORECASE)


def _sanitize_error_msg(msg: str) -> str:
    """Strip internal loopback/private URLs from user-facing job error messages."""
    return _INTERNAL_URL_RE.sub("[internal-engine]", str(msg))[:400]


def is_harness_llm_enabled() -> bool:
    """Strictly read-only server-controlled feature flag (always False unless explicitly enabled in env)."""
    return os.environ.get("V35_HARNESS_LLM_ENABLED", "0").strip().lower() in ("1", "true", "yes")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_harness_job_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M")
    return f"H-{stamp}-{uuid.uuid4().hex[:8].upper()}"


class HarnessQueueFullError(RuntimeError):
    """Raised when global active Harness queue depth reaches capacity (HTTP 429)."""


class HarnessClientBusyError(RuntimeError):
    """Raised when the same client_id or source IP already has an active Harness job (HTTP 429)."""


@dataclass
class _HarnessJobRecord:
    job_id: str
    client_id: str
    client_ip: str
    status: str  # queued | running | completed | partial | cancelled | failed
    phase: str
    current_round: int
    total_rounds: int
    progress_pct: float
    created_at: str
    created_mono: float
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    finished_mono: Optional[float] = None
    runtime_ms: float = 0.0
    cancel_requested: bool = False
    error: Optional[str] = None
    preflight: dict[str, Any] = field(default_factory=dict)
    request_summary: dict[str, Any] = field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: Optional[Future] = None

    def to_meta_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "phase": self.phase,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "progress_pct": self.progress_pct,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "runtime_ms": self.runtime_ms,
            "cancel_requested": self.cancel_requested,
            "error": self.error,
            "preflight": dict(self.preflight),
            "request_summary": dict(self.request_summary),
        }

    def to_detail_dict(self) -> dict[str, Any]:
        d = self.to_meta_dict()
        d["result"] = self.result
        return d


class HarnessJobManager:
    """Thread-safe dedicated single-worker async manager for Peptide-Harness jobs."""

    def __init__(
        self,
        adapter: OmniPanelAdapter,
        *,
        max_queue_depth: int = 8,
        max_retained: int = 128,
        max_retained_per_client: int = 8,
        ttl_seconds: float = 3600.0,
        min_protected_ttl_s: float = 300.0,
        interactive_yield_wait_s: float = 15.0,
        eval_batch_fn: Optional[Callable[[list[BatchParsedRow], bool, AssayConditions], list[BatchMoleculeResult]]] = None,
    ) -> None:
        self.adapter = adapter
        self.max_queue_depth = max(1, int(max_queue_depth))
        self.max_retained = max(8, int(max_retained))
        self.max_retained_per_client = max(2, int(max_retained_per_client))
        self.ttl_seconds = max(10.0, float(ttl_seconds))
        self.min_protected_ttl_s = max(0.0, min(self.ttl_seconds, float(min_protected_ttl_s)))
        self.interactive_yield_wait_s = max(0.5, float(interactive_yield_wait_s))
        self._custom_eval_batch_fn = eval_batch_fn

        self._lock = threading.Lock()
        self._jobs: OrderedDict[str, _HarnessJobRecord] = OrderedDict()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omnipanel-harness")

    def _prune_locked(self) -> None:
        """Evict expired terminal jobs, then prune per-client excess, protecting other clients' recent results."""
        now_m = time.monotonic()
        # 1. Evict strictly TTL-expired terminal jobs
        expired_ids = [
            jid
            for jid, rec in self._jobs.items()
            if rec.status not in ("queued", "running")
            and rec.finished_mono is not None
            and (now_m - rec.finished_mono) > self.ttl_seconds
        ]
        for jid in expired_ids:
            self._jobs.pop(jid, None)

        # 2. Enforce per-client retention cap first (prevents one client from spamming & evicting victims!)
        by_client: dict[str, list[str]] = {}
        for jid, rec in self._jobs.items():
            if rec.status not in ("queued", "running"):
                by_client.setdefault(rec.client_id, []).append(jid)
        for jids in by_client.values():
            while len(jids) > self.max_retained_per_client:
                oldest = jids.pop(0)
                self._jobs.pop(oldest, None)

        # 3. Enforce global retention cap ONLY on terminal jobs past the protection grace window
        evictable = [
            jid
            for jid, rec in self._jobs.items()
            if rec.status not in ("queued", "running")
            and rec.finished_mono is not None
            and (now_m - rec.finished_mono) >= self.min_protected_ttl_s
        ]
        terminal_total = sum(1 for rec in self._jobs.values() if rec.status not in ("queued", "running"))
        while terminal_total > self.max_retained and evictable:
            oldest = evictable.pop(0)
            self._jobs.pop(oldest, None)
            terminal_total -= 1

    def _check_quota_locked(self, client_id: str, client_ip: str) -> None:
        active = [r for r in self._jobs.values() if r.status in ("queued", "running")]
        if len(active) >= self.max_queue_depth:
            raise HarnessQueueFullError(
                f"Peptide-Harness queue is full ({len(active)}/{self.max_queue_depth} active jobs). Please retry shortly."
            )
        for r in active:
            if r.client_id == client_id:
                raise HarnessClientBusyError(
                    f"Client already has an active Peptide-Harness job ({r.job_id}). Wait for completion or cancel it first."
                )
            if client_ip and r.client_ip == client_ip:
                raise HarnessClientBusyError(
                    "Source IP already has an active Peptide-Harness job. Only 1 concurrent optimization job per IP is permitted."
                )

    def has_active_for(self, client_id: str, client_ip: str) -> bool:
        """Check whether `client_id` or `client_ip` already has an active async Harness job (used by sync route)."""
        with self._lock:
            self._prune_locked()
            for r in self._jobs.values():
                if r.status in ("queued", "running"):
                    if r.client_id == client_id or (client_ip and r.client_ip == client_ip):
                        return True
            return False

    def submit_job(
        self,
        *,
        client_id: str,
        client_ip: str,
        query: Optional[str],
        sequence: Optional[str],
        smiles: Optional[str],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        """Enforce cheap quota FIRST, then preflight-validate and enqueue a background Peptide-Harness job."""
        # 1. Cheap quota check BEFORE expensive RDKit preflight (P2-4)
        with self._lock:
            self._prune_locked()
            self._check_quota_locked(client_id, client_ip)

        # 2. Fail-fast preflight BEFORE taking a queue slot (returns HTTP 400 / 503 synchronously)
        preflight = preflight_validate_harness_request(
            self.adapter,
            query=query,
            sequence=sequence,
            smiles=smiles,
            config=config,
        )

        # 3. Re-verify quota under lock and atomically insert + submit (rolling back on submit error)
        with self._lock:
            self._prune_locked()
            self._check_quota_locked(client_id, client_ip)

            job_id = _new_harness_job_id()
            n_rounds = max(1, min(4, int(config.rounds)))
            rec = _HarnessJobRecord(
                job_id=job_id,
                client_id=client_id,
                client_ip=client_ip,
                status="queued",
                phase="queued",
                current_round=0,
                total_rounds=n_rounds,
                progress_pct=0.0,
                created_at=_utc_iso(),
                created_mono=time.monotonic(),
                preflight=preflight,
                request_summary={
                    "query_name": preflight["query_name"],
                    "objective_x": config.objective_x,
                    "objective_y": config.objective_y,
                    "functional_endpoint": config.functional_endpoint,
                    "rounds": n_rounds,
                    "population_size": max(2, min(8, int(config.population_size))),
                    "max_total_evals": max(4, min(60, int(config.max_total_evals))),
                    "strategies": list(config.strategies),
                    "harness_mode": preflight["harness_mode"],
                },
            )
            self._jobs[job_id] = rec
            try:
                fut = self._executor.submit(self._run_job, job_id, query, sequence, smiles, config)
                rec.future = fut
            except Exception:
                self._jobs.pop(job_id, None)
                raise
            return rec.to_meta_dict()

    def _run_job(
        self,
        job_id: str,
        query: Optional[str],
        sequence: Optional[str],
        smiles: Optional[str],
        config: HarnessConfig,
    ) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return
            if rec.cancel_event.is_set() or rec.status == "cancelled":
                rec.status = "cancelled"
                rec.phase = "cancelled"
                rec.finished_at = rec.finished_at or _utc_iso()
                rec.finished_mono = rec.finished_mono or time.monotonic()
                return
            rec.status = "running"
            rec.phase = "starting"
            rec.started_at = _utc_iso()
            cancel_evt = rec.cancel_event

        t0 = time.perf_counter()
        wall_budget_s = max(5.0, float(config.wall_clock_budget_s))

        def _on_progress(snap: dict[str, Any]) -> None:
            with self._lock:
                r = self._jobs.get(job_id)
                if r is None:
                    return
                if "phase" in snap:
                    r.phase = str(snap["phase"])
                if "current_round" in snap:
                    r.current_round = int(snap["current_round"])
                if "total_rounds" in snap:
                    r.total_rounds = int(snap["total_rounds"])
                if "progress_pct" in snap:
                    r.progress_pct = float(snap["progress_pct"])
                r.runtime_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        def _yielding_eval_batch(
            rows: list[BatchParsedRow],
            include_tier2: bool,
            conditions: AssayConditions,
        ) -> list[BatchMoleculeResult]:
            rem = wall_budget_s - (time.perf_counter() - t0)
            if rem <= 0.5:
                raise EngineBusyError("Peptide-Harness wall-clock budget exhausted before batch submission.")
            # Interactive yield gate bounded cleanly by remaining wall-clock budget
            wait_cap = min(self.interactive_yield_wait_s, max(0.5, rem * 0.25))
            ok_to_proceed = self.adapter.wait_for_interactive_idle(
                cancel_event=cancel_evt,
                max_wait_s=wait_cap,
            )
            if not ok_to_proceed or cancel_evt.is_set():
                return []
            if self._custom_eval_batch_fn is not None:
                return self._custom_eval_batch_fn(rows, include_tier2, conditions)
            rem_after_wait = max(5.0, min(45.0, wall_budget_s - (time.perf_counter() - t0)))
            return self.adapter.predict_batch(
                rows,
                include_tier2=include_tier2,
                conditions=conditions,
                http_timeout_s=rem_after_wait,
                is_background=True,
            )

        try:
            res = run_peptide_harness(
                self.adapter,
                query=query,
                sequence=sequence,
                smiles=smiles,
                config=config,
                cancel_event=cancel_evt,
                progress_callback=_on_progress,
                eval_batch_fn=_yielding_eval_batch,
            )
            elapsed = round((time.perf_counter() - t0) * 1000.0, 1)
            with self._lock:
                r = self._jobs.get(job_id)
                if r is not None:
                    r.status = str(res.get("status") or "completed")
                    r.phase = "done" if r.status in ("completed", "partial") else r.status
                    r.progress_pct = 100.0 if r.status in ("completed", "partial") else r.progress_pct
                    r.finished_at = _utc_iso()
                    r.finished_mono = time.monotonic()
                    r.runtime_ms = elapsed
                    pe = res.get("partial_error")
                    r.error = _sanitize_error_msg(pe) if pe else None
                    r.result = res
                    self._prune_locked()
        except (EngineUnavailableError, EngineBusyError, ValueError, Exception) as exc:
            elapsed = round((time.perf_counter() - t0) * 1000.0, 1)
            with self._lock:
                r = self._jobs.get(job_id)
                if r is not None:
                    r.status = "cancelled" if cancel_evt.is_set() else "failed"
                    r.phase = r.status
                    r.finished_at = _utc_iso()
                    r.finished_mono = time.monotonic()
                    r.runtime_ms = elapsed
                    r.error = None if cancel_evt.is_set() else _sanitize_error_msg(f"{type(exc).__name__}: {exc}")
                    self._prune_locked()

    def get_job(self, job_id: str, client_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            self._prune_locked()
            rec = self._jobs.get(job_id)
            if rec is None or rec.client_id != client_id:
                return None
            return rec.to_detail_dict()

    def list_jobs(self, client_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            self._prune_locked()
            mine = [r.to_meta_dict() for r in reversed(self._jobs.values()) if r.client_id == client_id]
            return mine[: max(1, min(64, int(limit)))]

    def cancel_job(self, job_id: str, client_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            self._prune_locked()
            rec = self._jobs.get(job_id)
            if rec is None or rec.client_id != client_id:
                return None
            already_finished = rec.status not in ("queued", "running")
            if not already_finished:
                rec.cancel_requested = True
                rec.cancel_event.set()
                if rec.status == "queued":
                    if rec.future is not None:
                        rec.future.cancel()
                    rec.status = "cancelled"
                    rec.phase = "cancelled"
                    rec.finished_at = _utc_iso()
                    rec.finished_mono = time.monotonic()
                # Wake up worker immediately if it is waiting on the interactive yield condition
                with self.adapter._predict_idle_cond:
                    self.adapter._predict_idle_cond.notify_all()
            out = rec.to_meta_dict()
            out["already_finished"] = already_finished
            return out

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._prune_locked()
            queued = sum(1 for r in self._jobs.values() if r.status == "queued")
            running = sum(1 for r in self._jobs.values() if r.status == "running")
            return {
                "queued": queued,
                "running": running,
                "total_retained": len(self._jobs),
                "max_queue_depth": self.max_queue_depth,
            }
