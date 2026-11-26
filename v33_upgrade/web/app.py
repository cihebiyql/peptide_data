"""FastAPI scientific web platform, async Task Center & REST API for Peptide OmniPanel."""

from __future__ import annotations

import os
import re
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from rdkit import rdBase

from . import __version__
from .adapter import EngineBusyError, EngineUnavailableError, OmniPanelAdapter
from .batch_parser import MAX_BATCH_SIZE, items_from_batch_request, parse_batch_text
from .chem_utils import InputValidationError, compute_physchem, render_molecule_svg
from .drug_library import (
    BATCH_DEMO_SAMPLE,
    BATCH_PRESETS,
    DRUG_LIBRARY,
    EXAMPLE_CATEGORIES,
    QUICK_EXAMPLES,
    lookup_drug,
)
from .job_store import JobManager
from .schemas import (
    AssayConditions,
    AtomAttributionRequest,
    AtomAttributionResponse,
    BatchParseResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    EndpointsResponse,
    HealthResponse,
    JobCreateRequest,
    PredictRequest,
    PredictResponse,
    RenderSvgRequest,
    RenderSvgResponse,
    TopAtomContribution,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
ROOT_PATH = os.environ.get("V33_ROOT_PATH", "").strip()
CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{6,64}$")
MAX_UPLOAD_BYTES = int(os.environ.get("V33_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024)))
ENABLE_DOCS = os.environ.get("V33_ENABLE_DOCS", "1").strip().lower() not in ("0", "false", "no")
REDACT_HEALTH_PATHS = os.environ.get("V33_PUBLIC_HEALTH_REDACT", "0").strip().lower() in ("1", "true", "yes")

adapter = OmniPanelAdapter()
job_manager = JobManager(adapter=adapter)


def _start_background_warmup() -> None:
    if os.environ.get("V33_WARMUP_EXAMPLES", "0").strip().lower() not in ("1", "true", "yes"):
        return
    refresh_sec = max(300.0, float(os.environ.get("V33_WARMUP_REFRESH_SEC", "43200")))

    def _runner() -> None:
        while True:
            try:
                if adapter.remote_url and not adapter.remote_healthy:
                    adapter.refresh_remote_health()
                if adapter.mode == "v33_remote":
                    n = adapter.warmup_curated_examples()
                    print(f"[OmniPanel] Pre-warmed {n}/25 curated examples into 24h TTL cache.", flush=True)
                    if n > 0:
                        time.sleep(refresh_sec)
                        continue
            except Exception as exc:
                print(f"[OmniPanel] Warmup attempt deferred: {exc}", flush=True)
            time.sleep(20.0)

    threading.Thread(target=_runner, name="omnipanel-warmup", daemon=True).start()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _start_background_warmup()
    yield


app = FastAPI(
    title="Peptide OmniPanel API",
    description=(
        "Publication-grade peptide & cyclic-peptide ADMET, pharmacokinetics, safety, and antimicrobial "
        "prediction platform powered by a 4-foundation-model super-ensemble (47 tiered endpoints incl. "
        "macrocycle chameleonicity LogD_decadiene, calibrated probabilities, AD-stratified empirical uncertainty "
        "intervals, exact Tanimoto AD, 59-product NLM DailyMed / US FDA factual regulatory PK reference layer, "
        "and async task tracking). Scope: internal_research_only."
    ),
    version=__version__,
    root_path=ROOT_PATH,
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_client_id(header_id: Optional[str], query_id: Optional[str] = None) -> str:
    cid = (header_id or query_id or "").strip()
    if cid and CLIENT_ID_RE.fullmatch(cid):
        return cid
    # Never share a single anonymous bucket across unrelated callers
    return f"anon-{secrets.token_hex(8)}"


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def get_health() -> HealthResponse:
    """Service liveness, engine readiness, RDKit version, and audited dataset KPIs."""
    if adapter.remote_url and not adapter.remote_healthy:
        adapter.refresh_remote_health()
    catalog = adapter.get_catalog()
    return HealthResponse(
        ok=True,
        version=__version__,
        adapter_mode=adapter.mode,
        bundle_loaded=bool(adapter.manifest),
        bundle_path=None if REDACT_HEALTH_PATHS else (str(adapter.manifest_path) if adapter.manifest_path else None),
        v33_remote_url=None if REDACT_HEALTH_PATHS else adapter.remote_url,
        v33_remote_healthy=adapter.remote_healthy,
        manifest_loaded=bool(adapter.manifest),
        manifest_path=None if REDACT_HEALTH_PATHS else (str(adapter.manifest_path) if adapter.manifest_path else None),
        rdkit_version=rdBase.rdkitVersion,
        endpoints_total=len(catalog),
        tier1_endpoints=sum(1 for e in catalog if e.tier == 1),
        tier2_endpoints=sum(1 for e in catalog if e.tier == 2),
        case17_drugs_loaded=sum(1 for d in DRUG_LIBRARY if d.get("case17")),
        dataset_summary=adapter.get_dataset_summary(),
        uptime_s=round(time.time() - adapter.start_time, 1),
    )


@app.get("/api/v1/endpoints", response_model=EndpointsResponse, tags=["Catalog"])
def get_endpoints() -> EndpointsResponse:
    """List all audited Tier-1 and Tier-2 endpoints and validation transparency metadata."""
    if adapter.remote_url and not adapter.remote_healthy:
        adapter.refresh_remote_health()
    tp = adapter.get_transparency_payload()
    return EndpointsResponse(
        version=__version__,
        adapter_mode=adapter.mode,
        manifest_generated_at=tp["manifest_generated_at"],
        dataset_summary=tp["dataset_summary"],
        overall_summary=tp["overall_summary"],
        representation_ablation_summary=tp["representation_ablation_summary"],
        case17_scorecard=tp["case17_scorecard"],
        selection_protocol=tp["selection_protocol"],
        v335_protocol_audit=tp["v335_protocol_audit"],
        egat_fidelity_audit=tp["egat_fidelity_audit"],
        external_baselines_same_split=tp["external_baselines_same_split"],
        temporal_split_benchmarks=tp["temporal_split_benchmarks"],
        recent_fda_2019_2023_blind_set=tp["recent_fda_2019_2023_blind_set"],
        endpoints=adapter.get_catalog(),
    )


@app.get("/api/v1/examples", tags=["Catalog"])
def get_examples() -> JSONResponse:
    """Curated quick-example chips, categorized clinical library (25 peptides), and batch benchmark presets."""
    drugs_index = [
        {
            "key": d["key"],
            "display_name": d["display_name"],
            "zh_name": d["zh_name"],
            "category": d.get("category", "linear_clinical"),
            "drug_class": d["drug_class"],
            "case17": d.get("case17", False),
            "fda_half_life": d.get("fda_half_life"),
            "fda_ppb": d.get("fda_ppb"),
            "query": d["sequence"] if (not d.get("smiles") and d.get("sequence")) else d["display_name"],
            "has_smiles": bool(d.get("smiles")),
            "has_sequence": bool(d.get("sequence")),
        }
        for d in DRUG_LIBRARY
    ]
    return JSONResponse(
        {
            "quick_examples": QUICK_EXAMPLES,
            "example_categories": EXAMPLE_CATEGORIES,
            "drugs": drugs_index,
            "batch_demo_csv": BATCH_DEMO_SAMPLE,
            "batch_presets": BATCH_PRESETS,
        }
    )


@app.get("/api/v1/clinical-evidence", tags=["Evidence"])
def get_clinical_evidence(
    query: Optional[str] = Query(default=None, description="Drug name, brand/alias, canonical 20-AA sequence, or SMILES."),
    smiles: Optional[str] = Query(default=None, description="Optional explicit covalent SMILES for RDKit InChIKey14 matching."),
    sequence: Optional[str] = Query(default=None, description="Optional explicit canonical 20-AA sequence."),
    limit: int = Query(default=30, ge=1, le=120),
) -> JSONResponse:
    """Query factual human clinical PK records across 59 FDA-approved peptide drugs (verified against NLM DailyMed SPL setids and US FDA Drugs@FDA reviews; zero CC-BY-NC derivative fields)."""
    return JSONResponse(
        adapter.query_clinical_evidence(
            query=query,
            sequence=sequence,
            smiles=smiles,
            limit=limit,
        )
    )


@app.post("/api/v1/render-svg", response_model=RenderSvgResponse, tags=["Chemistry"])
def post_render_svg(req: RenderSvgRequest) -> RenderSvgResponse:
    """Render a server-side 2D molecular structure SVG via RDKit with motif substructure highlights."""
    try:
        resolved = adapter.resolve_query(query=req.query, sequence=req.sequence, smiles=req.smiles)
    except InputValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    physchem = compute_physchem(resolved.mol, sequence=resolved.sequence)
    svg, highlights, modifications = render_molecule_svg(
        resolved.mol,
        sequence=resolved.sequence,
        width=req.width,
        height=req.height,
    )
    atom_cnt = int(resolved.mol.GetNumAtoms()) if resolved.mol is not None else physchem.heavy_atoms
    bond_cnt = int(resolved.mol.GetNumBonds()) if resolved.mol is not None else max(0, atom_cnt - 1)
    return RenderSvgResponse(
        svg=svg,
        resolved_type=resolved.resolved_type,
        formula=physchem.formula,
        atom_count=atom_cnt,
        bond_count=bond_cnt,
        width=req.width,
        height=req.height,
        highlights=highlights,
        modifications=modifications,
    )


@app.post("/api/v1/atom-attribution", response_model=AtomAttributionResponse, tags=["Interpretability"])
def post_atom_attribution(req: AtomAttributionRequest) -> AtomAttributionResponse:
    """Compute live Pure-PyTorch E-GAT + WeightAndSum per-atom attribution heatmap SVG (<100 ms CPU)."""
    try:
        resolved = adapter.resolve_query(query=req.query, sequence=req.sequence, smiles=req.smiles)
    except InputValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    res = adapter.compute_atom_attribution(
        resolved=resolved,
        endpoint=req.endpoint,
        width=req.width,
        height=req.height,
    )
    if res is None:
        raise HTTPException(
            status_code=503,
            detail="Live GNN atom attribution engine is not loaded on this instance.",
        )
    return AtomAttributionResponse(
        svg=res["svg"],
        endpoint=res["endpoint"],
        endpoint_label=res["endpoint_label"],
        n_atoms=res["n_atoms"],
        latency_ms=res["latency_ms"],
        cache_hit=res["cache_hit"],
        gnn_prediction=res["gnn_prediction"],
        top_atoms=[TopAtomContribution(**a) for a in res["top_atoms"]],
        available_endpoints=res["available_endpoints"],
        interpretability_tier=res.get("interpretability_tier"),
        fidelity_vs_4fm_spearman=res.get("fidelity_vs_4fm_spearman"),
        deletion_top_vs_rand_ratio=res.get("deletion_top_vs_rand_ratio"),
        deletion_top_vs_rand_ci95=res.get("deletion_top_vs_rand_ci95"),
    )


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["Inference"])
def post_predict(
    req: PredictRequest,
    x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
) -> PredictResponse:
    """Run full-panel ADMET/PK/Antimicrobial prediction for one peptide (optionally logged to Task Center)."""
    try:
        resp = adapter.predict(
            query=req.query,
            sequence=req.sequence,
            smiles=req.smiles,
            include_tier2=req.include_tier2,
            include_svg=req.include_svg,
            svg_width=req.svg_width,
            svg_height=req.svg_height,
            conditions=req.conditions,
        )
    except InputValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EngineBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "5"}) from exc
    except EngineUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if x_client_id and CLIENT_ID_RE.fullmatch(x_client_id.strip()):
        try:
            job_id = job_manager.record_single_prediction(
                client_id=x_client_id.strip(),
                pred=resp,
                include_tier2=req.include_tier2,
                conditions=req.conditions,
            )
            resp.summary["job_id"] = job_id
        except Exception:
            pass
    return resp


@app.post("/api/v1/batch/parse", response_model=BatchParseResponse, tags=["Batch"])
def post_batch_parse(req: BatchPredictRequest) -> BatchParseResponse:
    """Parse pasted text or CSV/TSV/FASTA content for instant client-side preview before batch inference."""
    if req.items is not None:
        rows = items_from_batch_request(req.items, drug_lookup_fn=lookup_drug)
        valid_c = sum(1 for r in rows if r.valid)
        return BatchParseResponse(
            total=len(rows),
            valid_count=valid_c,
            invalid_count=len(rows) - valid_c,
            truncated=len(req.items) > MAX_BATCH_SIZE,
            max_batch=MAX_BATCH_SIZE,
            rows=rows,
        )
    if not req.raw_text or not req.raw_text.strip():
        raise HTTPException(status_code=400, detail="Provide raw_text or items for batch parsing.")
    return parse_batch_text(req.raw_text, filename=req.filename, drug_lookup_fn=lookup_drug)


def _execute_batch_rows(
    parsed_rows,
    include_tier2: bool,
    conditions: AssayConditions,
) -> BatchPredictResponse:
    t0 = time.perf_counter()
    results = adapter.predict_batch(
        parsed_rows,
        include_tier2=include_tier2,
        conditions=conditions,
    )
    succeeded = sum(1 for r in results if r.status == "ok")
    failed = len(results) - succeeded
    return BatchPredictResponse(
        total=len(parsed_rows),
        succeeded=succeeded,
        failed=failed,
        runtime_ms=round((time.perf_counter() - t0) * 1000.0, 1),
        adapter_mode=adapter.mode,
        results=results,
    )


@app.post("/api/v1/batch", response_model=BatchPredictResponse, tags=["Batch"])
def post_batch_predict(req: BatchPredictRequest) -> BatchPredictResponse:
    """Run synchronous batch ADMET predictions for up to 500 peptides (JSON items or CSV/TSV/TXT/FASTA text)."""
    if req.items is not None:
        if len(req.items) == 0:
            raise HTTPException(status_code=400, detail="Batch items list cannot be empty.")
        parsed_rows = items_from_batch_request(req.items, drug_lookup_fn=lookup_drug)
    elif req.raw_text and req.raw_text.strip():
        parsed = parse_batch_text(req.raw_text, filename=req.filename, drug_lookup_fn=lookup_drug)
        if parsed.total == 0:
            raise HTTPException(status_code=400, detail="No molecules could be parsed from raw_text.")
        parsed_rows = parsed.rows
    else:
        raise HTTPException(status_code=400, detail="Provide either 'items' or 'raw_text' for batch prediction.")

    try:
        return _execute_batch_rows(parsed_rows, include_tier2=req.include_tier2, conditions=req.conditions)
    except EngineBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "5"}) from exc
    except EngineUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/v1/batch/upload", response_model=BatchPredictResponse, tags=["Batch"])
async def post_batch_upload(
    request: Request,
    file: UploadFile = File(...),
    include_tier2: bool = Form(default=False),
    rbc_species: str = Form(default="human"),
    pk_context: str = Form(default="human_plasma"),
) -> BatchPredictResponse:
    """Upload a .csv, .tsv, .txt, or .fasta file (<=2 MB, up to 500 molecules) and run synchronous batch prediction."""
    cl_hdr = request.headers.get("content-length")
    if cl_hdr and cl_hdr.isdigit() and int(cl_hdr) > MAX_UPLOAD_BYTES + 65536:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds maximum allowed size ({MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )
    chunks: list[bytes] = []
    total_bytes = 0
    while True:
        chunk = await file.read(65536)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds maximum allowed size ({MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
            )
        chunks.append(chunk)
    raw_bytes = b"".join(chunks)
    text = ""
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    parsed = parse_batch_text(text, filename=file.filename, drug_lookup_fn=lookup_drug)
    if parsed.total == 0:
        raise HTTPException(status_code=400, detail="No molecules found in uploaded file.")

    cond = AssayConditions(
        rbc_species=rbc_species if rbc_species in ("human", "horse", "mouse") else "human",  # type: ignore[arg-type]
        pk_context=pk_context if pk_context in ("human_plasma", "human_sc", "human_po", "rat_iv", "mouse_sc") else "human_plasma",  # type: ignore[arg-type]
    )
    try:
        return _execute_batch_rows(parsed.rows, include_tier2=include_tier2, conditions=cond)
    except EngineBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "5"}) from exc
    except EngineUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ==================== ASYNC TASK CENTER ENDPOINTS ====================


@app.post("/api/v1/jobs", tags=["Tasks"])
def post_create_job(
    req: JobCreateRequest,
    x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
) -> JSONResponse:
    """Submit a background batch prediction task with live progress tracking in the Task Center."""
    if req.items is not None:
        if len(req.items) == 0:
            raise HTTPException(status_code=400, detail="Batch items list cannot be empty.")
        parsed_rows = items_from_batch_request(req.items, drug_lookup_fn=lookup_drug)
    elif req.raw_text and req.raw_text.strip():
        parsed = parse_batch_text(req.raw_text, filename=req.filename, drug_lookup_fn=lookup_drug)
        if parsed.total == 0:
            raise HTTPException(status_code=400, detail="No molecules could be parsed from raw_text.")
        parsed_rows = parsed.rows
    else:
        raise HTTPException(status_code=400, detail="Provide either 'items' or 'raw_text' to create a task.")

    client_id = _resolve_client_id(x_client_id)
    title = (req.title or "").strip() or (Path(req.filename).stem if req.filename else f"Batch ({len(parsed_rows)})")
    meta = job_manager.submit_batch_job(
        client_id=client_id,
        parsed_rows=parsed_rows,
        include_tier2=req.include_tier2,
        conditions=req.conditions,
        title=title,
    )
    return JSONResponse(meta)


@app.get("/api/v1/jobs", tags=["Tasks"])
def get_list_jobs(
    limit: int = Query(default=50, ge=1, le=100),
    x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
    client_id: Optional[str] = Query(default=None),
) -> JSONResponse:
    """List prediction tasks for the active browser session (newest first)."""
    cid = _resolve_client_id(x_client_id, client_id)
    jobs = job_manager.list_jobs(client_id=cid, limit=limit)
    active_count = sum(1 for j in jobs if j["status"] in ("queued", "running"))
    return JSONResponse({"jobs": jobs, "active_count": active_count, "total": len(jobs)})


@app.get("/api/v1/jobs/{job_id}", tags=["Tasks"])
def get_job_detail(
    job_id: str,
    x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
    client_id: Optional[str] = Query(default=None),
) -> JSONResponse:
    """Fetch full task status, progress, analytics summary, and all completed molecule predictions."""
    cid = _resolve_client_id(x_client_id, client_id)
    detail = job_manager.get_job_detail(job_id=job_id, client_id=cid)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Task '{job_id}' not found.")
    return JSONResponse(detail)


@app.post("/api/v1/jobs/{job_id}/cancel", tags=["Tasks"])
def post_cancel_job(
    job_id: str,
    x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
) -> JSONResponse:
    """Request graceful cancellation of a queued or running background task."""
    cid = _resolve_client_id(x_client_id)
    meta = job_manager.cancel_job(job_id=job_id, client_id=cid)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Task '{job_id}' not found.")
    return JSONResponse(meta)


@app.delete("/api/v1/jobs/{job_id}", tags=["Tasks"])
def delete_job(
    job_id: str,
    x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
) -> JSONResponse:
    """Delete a task from this session's Task Center history."""
    cid = _resolve_client_id(x_client_id)
    deleted = job_manager.delete_job(job_id=job_id, client_id=cid)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Task '{job_id}' not found.")
    return JSONResponse({"deleted": True, "job_id": job_id})


@app.get("/api/v1/jobs/{job_id}/export", tags=["Tasks"])
def export_job(
    job_id: str,
    format: str = Query(default="summary_csv", pattern="^(summary_csv|full_csv|json)$"),
    x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
    client_id: Optional[str] = Query(default=None),
) -> Response:
    """Download completed task results as Summary CSV, Full 46-Endpoint Wide CSV, or JSON."""
    cid = _resolve_client_id(x_client_id, client_id)
    detail = job_manager.get_job_detail(job_id=job_id, client_id=cid)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Task '{job_id}' not found.")
    safe_id = re.sub(r"[^A-Za-z0-9_\-]+", "_", job_id)
    if format == "json":
        return JSONResponse(
            detail,
            headers={"Content-Disposition": f'attachment; filename="omnipanel_{safe_id}.json"'},
        )
    csv_text = job_manager.export_job_csv(detail, full_wide=(format == "full_csv"))
    suffix = f"full_{len(adapter.get_catalog())}ep" if format == "full_csv" else "summary"
    return Response(
        content=csv_text.encode("utf-8-sig"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="omnipanel_{safe_id}_{suffix}.csv"'},
    )


# Mount static SPA assets and serve index.html at root
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend static/index.html not built yet.")
    return FileResponse(str(index_path))
