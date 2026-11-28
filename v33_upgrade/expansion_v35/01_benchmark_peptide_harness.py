#!/usr/bin/env python3
"""Step 3 (v35.2): Equal-Budget, Fair Strong-Baseline & Anti-Circularity Benchmark for Peptide-Harness.

Addresses 100% of independent red-team audit R3 findings (P0-1, P0-2, P1-1, P1-2, P1-3, P2-1..3):
  - P0-1: `Exhaustive-1Step-GreedyCombine` is now a TRUE non-starved strong baseline:
      * Pre-filters proposals with free RDKit Stage A (identical to Harness);
      * Allocates Phase 1 fairly across ALL 5 operator families (4 round-robin chemical leaves +
        11 interleaved point_mutation/ala_scan 1-step edits) + Phase 2 greedy 2-site stacking
        (and reinvests any unused Phase 2 budget into additional Phase 1 candidates so zero budget is wasted!).
  - P0-2: Unambiguously separates recommendation safety and deployment gain:
      * `unsafe_top1_rate` (non-null Top-1 that FAILED Stage A/B: 0% for gated methods vs >0% for NoGate/Unconstrained);
      * `abstention_rate` (safe `top1=None` abstention on toxic parents like Melittin);
      * `safe_recommendation_precision` (feasible / non-null Top-1);
      * `selected_feasible_delta_gmpo` (PRIMARY deployment metric: 0.0 whenever Top-1 is unsafe/None — NO post-hoc rescuing!);
      * `pool_best_feasible_delta_gmpo` (secondary diagnostic).
  - P1-1: Fixes dead `"AD_FLOOR_FAIL"` string -> checks actual `"OOD_DRIFT:"` reject prefix, and separates
      `ad_floor_drift_rate` from `raw_ood_status_rate` (`ad_overall == "ood"`).
  - P1-2: Reports TRUE `stage_a_pass_rate = n_passed / (n_passed + n_rejected)` for both Harness and baselines;
      fixes `key_endpoints` -> `key_metrics` so `top1_hemolysis_cls` and `parent_hemolysis_cls` are 100% populated;
      applies symmetric free Stage A pre-filtering to `Random-EqualBudget` & `StrongBase` (only `Unconstrained-Greedy` skips it).
  - P1-3: Expands panel from 10 to 16 leads (14 linear + 2 cyclic; 8/16 = 50% verified `out_of_sample_novel`
      with `in_corpus_inchikey14=False` AND `in_corpus_sequence=False`), breaking the n=10 Holm ceiling and
      disclosing exact `in_corpus_any` counts without masking Case17 drugs' presence in `compounds_v33.tsv`.
  - P2-1: Stores full `"top1_lineage_path"` and `"top1_lineage_sequence"` in every per-lead method record.

Runs STRICTLY against the live 4-FM V33 production inference engine (zero offline heuristic fallback;
asserts `adapter.mode == 'v33_remote'` and `oracle_is_surrogate_heuristic == False` on every run)
and performs independent external hemolysis cross-checking via official pretrained HemoPI2 v1.3
(canonical-sequence subset only, with explicit coverage % and `unverified_by_hemopi2` tagging).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import csv
import json
import math
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
V33_ROOT = SCRIPT_DIR.parent
REPO_PARENT = V33_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from v33_upgrade.web.adapter import OmniPanelAdapter
from v33_upgrade.web.batch_mpo import compute_peptide_mpo
from v33_upgrade.web.chem_utils import detect_acylation_anchor, parse_smiles_safe, sequence_to_mol
from v33_upgrade.web.drug_library import lookup_drug
from v33_upgrade.web.peptide_harness import (
    ALL_HARNESS_STRATEGIES,
    CHEM_LEAF_STRATEGIES,
    DeterministicGatekeeper,
    HarnessConfig,
    NegativeSampleMemory,
    _CandidateSpec,
    _build_combinatorial_stack_candidates,
    _extract_raw_and_display,
    _sanitize_and_canonicalize,
    compute_biophysical_profile,
    compute_pareto_flags_and_hv,
    default_direction_for_endpoint,
    generate_chemical_leaf_candidates,
    generate_sequence_candidates,
    resolve_core_locks,
    run_peptide_harness,
)
from v33_upgrade.web.schemas import AssayConditions, BatchMoleculeResult, BatchParsedRow, MpoWeights

csv.field_size_limit(10_000_000)

HEMOPI_ROOT = Path("/data/qlyu/projects/peptide_author_reproduction_20261014")
HEMOPI_MODEL_DIR = HEMOPI_ROOT / "peptide_reference_bundle_20261013/external_assets/HemoPI2/extracted/Model"
HEMOPI_SKLEARN_PY_PATH = str(HEMOPI_ROOT / "runtime_sklearn")

EQUAL_BUDGET_EVALS = 20
BENCH_SEED = 20261128
REF_DX = -0.50
REF_DY = -0.50
BENCH_WORKERS = int(os.environ.get("V35_BENCH_WORKERS", "2"))


# ---------------------------------------------------------------------------
# 1. Expanded 16-Lead Panel (14 Linear + 2 Cyclic; 8/16 = 50% Verified OOS Novel!)
# ---------------------------------------------------------------------------

def build_benchmark_leads() -> List[Dict[str, Any]]:
    oct_entry = lookup_drug("Octreotide") or {}
    oxy_entry = lookup_drug("Oxytocin") or {}
    exe_entry = lookup_drug("Exenatide") or {}
    glu_entry = lookup_drug("Glucagon") or {}
    ter_entry = lookup_drug("Teriparatide") or {}

    return [
        # --- Group A: Functional-Guided AMPs (Tier-A MIC_E_coli non-worsening + locked pharmacophore, n=5) ---
        {
            "lead_id": "Magainin_2",
            "group": "amp_functional_guided",
            "sequence": "GIGKFLHSAKKFGKAFVGEIMNS",
            "smiles": None,
            "objective_x": "PAMPA",
            "objective_y": "half_life_reg",
            "functional_endpoint": "MIC_E_coli",
            "locked_positions": [1, 4, 8, 11],
            "note": "Classic frog AMP (23 AA, in-corpus); optimize permeability/T1/2 while preserving MIC_E_coli",
        },
        {
            "lead_id": "Melittin",
            "group": "amp_functional_guided",
            "sequence": "GIGAVLKVLTTGLPALISWIKRKRQQ",
            "smiles": None,
            "objective_x": "PAMPA",
            "objective_y": "half_life_reg",
            "functional_endpoint": "MIC_E_coli",
            "locked_positions": [19, 21, 22],
            "note": "High-hemolysis venom AMP (26 AA, in-corpus); tests safe abstention vs unsafe false positives",
        },
        {
            "lead_id": "Synth_AMP_OOS2",
            "group": "amp_functional_guided",
            "sequence": "GWRKIFKKIEKVGQNIRDG",
            "smiles": None,
            "objective_x": "PAMPA",
            "objective_y": "half_life_reg",
            "functional_endpoint": "MIC_E_coli",
            "locked_positions": [2, 3, 6, 7],
            "note": "Out-of-sample novel synthetic cationic AMP (19 AA, verified absent from compounds_v33.tsv)",
        },
        {
            "lead_id": "Synth_AMP_OOS4",
            "group": "amp_functional_guided",
            "sequence": "RWLKQLKKVGKIVKQLRGN",
            "smiles": None,
            "objective_x": "PAMPA",
            "objective_y": "half_life_reg",
            "functional_endpoint": "MIC_E_coli",
            "locked_positions": [2, 3, 7, 8],
            "note": "Out-of-sample novel cationic amphipathic AMP (19 AA, verified absent from compounds_v33.tsv)",
        },
        {
            "lead_id": "Synth_AMP_OOS5",
            "group": "amp_functional_guided",
            "sequence": "GLWKIFKKIEHVGRNIRNG",
            "smiles": None,
            "objective_x": "PAMPA",
            "objective_y": "half_life_reg",
            "functional_endpoint": "MIC_E_coli",
            "locked_positions": [3, 4, 7, 8],
            "note": "Out-of-sample novel Trp/His-rich AMP (19 AA, verified absent from compounds_v33.tsv)",
        },
        # --- Group B: Clinical & Receptor-Locked Linear Peptides (n=5: 3 Case17 + 2 verified OOS novel analogs) ---
        {
            "lead_id": "Exenatide",
            "group": "clinical_linear_locked",
            "sequence": exe_entry.get("sequence") or "HGEGTFTSDLSKQMEEEAVRLFIEWLKNGGPSSGAPPPS",
            "smiles": None,
            "objective_x": "PAMPA",
            "objective_y": "half_life_reg",
            "functional_endpoint": None,
            "locked_positions": [1, 2, 3, 4, 6, 7],
            "note": "Case17 clinical GLP-1 agonist (39 AA; present in full prod corpus, held-out only in Case17 CV audit)",
        },
        {
            "lead_id": "Glucagon",
            "group": "clinical_linear_locked",
            "sequence": glu_entry.get("sequence") or "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT",
            "smiles": None,
            "objective_x": "PAMPA",
            "objective_y": "half_life_reg",
            "functional_endpoint": None,
            "locked_positions": [1, 2, 3, 4, 6],
            "note": "Case17 clinical linear hormone (29 AA; present in full prod corpus, held-out only in Case17 CV audit)",
        },
        {
            "lead_id": "Teriparatide",
            "group": "clinical_linear_locked",
            "sequence": ter_entry.get("sequence") or "SVSEIQLMHNLGKHLNSMERVEWLRKKLQDVHNF",
            "smiles": None,
            "objective_x": "PAMPA",
            "objective_y": "half_life_reg",
            "functional_endpoint": None,
            "locked_positions": [1, 2, 3, 4, 5],
            "note": "Case17 clinical PTH(1-34) analog (34 AA; present in full prod corpus, held-out only in Case17 CV audit)",
        },
        {
            "lead_id": "Synth_GLP_OOS1",
            "group": "clinical_linear_locked",
            "sequence": "HGEGTFTSDLSKQMEEQAVRLFIEWLKN",
            "smiles": None,
            "objective_x": "PAMPA",
            "objective_y": "half_life_reg",
            "functional_endpoint": None,
            "locked_positions": [1, 2, 3, 4, 6, 7],
            "note": "Out-of-sample novel truncated GLP-1 analog (28 AA, verified absent from compounds_v33.tsv)",
        },
        {
            "lead_id": "Synth_Hormone_OOS1",
            "group": "clinical_linear_locked",
            "sequence": "HSQGTFTSDYSKYLDSKRAQDFVQWLMN",
            "smiles": None,
            "objective_x": "PAMPA",
            "objective_y": "half_life_reg",
            "functional_endpoint": None,
            "locked_positions": [1, 2, 3, 4, 6],
            "note": "Out-of-sample novel glucagon-family analog (28 AA, verified absent from compounds_v33.tsv)",
        },
        # --- Group C: Developability-Only Linear Leads (no locks / no functional EP -> AA<=2, activity_unmodeled=True, n=4) ---
        {
            "lead_id": "Dynorphin_DevOnly",
            "group": "developability_only",
            "sequence": "YGGFLRRIRPKLK",
            "smiles": None,
            "objective_x": "PAMPA",
            "objective_y": "half_life_reg",
            "functional_endpoint": None,
            "locked_positions": [],
            "note": "Developability-only neuropeptide (13 AA, in-corpus; AA edits strictly capped <=2)",
        },
        {
            "lead_id": "Synth_Linear_OOS_DevOnly",
            "group": "developability_only",
            "sequence": "KWFKPLKGIAKWLRSV",
            "smiles": None,
            "objective_x": "PAMPA",
            "objective_y": "half_life_reg",
            "functional_endpoint": None,
            "locked_positions": [],
            "note": "Out-of-sample novel linear 16-mer in developability_only mode (AA<=2)",
        },
        {
            "lead_id": "Synth_DevOnly_OOS3",
            "group": "developability_only",
            "sequence": "YGGFLRRQFKVVTRSQ",
            "smiles": None,
            "objective_x": "PAMPA",
            "objective_y": "half_life_reg",
            "functional_endpoint": None,
            "locked_positions": [],
            "note": "Out-of-sample novel opioid-like 16-mer in developability_only mode (AA<=2)",
        },
        {
            "lead_id": "Synth_DevOnly_OOS5",
            "group": "developability_only",
            "sequence": "KLFKRIVQRIKDFLRNLV",
            "smiles": None,
            "objective_x": "PAMPA",
            "objective_y": "half_life_reg",
            "functional_endpoint": None,
            "locked_positions": [],
            "note": "Out-of-sample novel amphipathic 18-mer in developability_only mode (AA<=2)",
        },
        # --- Group D: Cyclic Disulfide Macrocycles (`chemical_leaf_only` mode, 0 sequence edits, n=2) ---
        {
            "lead_id": "Octreotide",
            "group": "cyclic_chemical_leaf_only",
            "sequence": None,
            "smiles": oct_entry.get("smiles"),
            "objective_x": "PAMPA",
            "objective_y": "half_life_reg",
            "functional_endpoint": None,
            "locked_positions": [],
            "note": "Case17 clinical disulfide macrocycle (`chemical_leaf_only`, S-S locked; in full prod corpus)",
        },
        {
            "lead_id": "Oxytocin",
            "group": "cyclic_chemical_leaf_only",
            "sequence": None,
            "smiles": oxy_entry.get("smiles"),
            "objective_x": "PAMPA",
            "objective_y": "half_life_reg",
            "functional_endpoint": None,
            "locked_positions": [],
            "note": "Case17 clinical disulfide nonapeptide (`chemical_leaf_only`, S-S locked; in full prod corpus)",
        },
    ]


# ---------------------------------------------------------------------------
# 2. Corpus Overlap Audit (`in_sample` vs `out_of_sample` via compounds_v33.tsv)
# ---------------------------------------------------------------------------

def load_corpus_index(compounds_tsv: Path) -> Tuple[set[str], set[str]]:
    ik14_set: set[str] = set()
    seq_set: set[str] = set()
    if not compounds_tsv.exists():
        return ik14_set, seq_set
    with compounds_tsv.open("r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            ik = (row.get("inchikey_14") or "").strip()
            if ik:
                ik14_set.add(ik)
            sq = (row.get("parent_sequence") or "").strip().upper()
            if sq:
                seq_set.add(sq)
    return ik14_set, seq_set


def audit_lead_corpus_status(
    lead: Dict[str, Any],
    ik14_set: set[str],
    seq_set: set[str],
    case17_names: set[str],
) -> Dict[str, Any]:
    mol = parse_smiles_safe(lead["smiles"]) if lead.get("smiles") else sequence_to_mol(lead["sequence"])
    ik14 = MolToInchiKey(mol)[:14] if mol is not None else None
    seq = (lead.get("sequence") or "").strip().upper() or None
    in_ik = bool(ik14 and ik14 in ik14_set)
    in_seq = bool(seq and seq in seq_set)
    in_any = in_ik or in_seq
    is_case17 = lead["lead_id"].lower() in case17_names
    # Explicit transparent status (P1-3): never mask `in_corpus_any=True` behind `case17`!
    if is_case17 and in_any:
        status = "case17_in_prod_corpus"
    elif is_case17:
        status = "case17_out_of_corpus"
    elif in_any:
        status = "in_sample_corpus"
    else:
        status = "out_of_sample_novel"
    return {
        "inchikey_14": ik14,
        "in_corpus_inchikey14": in_ik,
        "in_corpus_sequence": in_seq,
        "in_corpus_any": in_any,
        "is_case17": is_case17,
        "corpus_status": status,
    }


# ---------------------------------------------------------------------------
# 3. Local Ephemeral Live V33 Production Engine Manager (on 64-core node23)
# ---------------------------------------------------------------------------

def start_local_v33_service(port: int = 18955, esmc_port: int = 18956) -> Tuple[List[subprocess.Popen], str]:
    import socket

    url = f"http://127.0.0.1:{port}"
    procs: List[subprocess.Popen] = []
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=2.0) as resp:
            if resp.status == 200:
                print(f"[bench] Reusing existing live V33 service at {url}", flush=True)
                return procs, url
    except Exception:
        pass

    env = os.environ.copy()
    env["V33_ESMC_PYTHON"] = sys.executable
    env["OMP_NUM_THREADS"] = "8"
    env["MKL_NUM_THREADS"] = "8"

    esmc_script = SCRIPT_DIR / "esmc_worker_standalone.py"
    if not esmc_script.exists():
        esmc_script = V33_ROOT / "bundle" / "esmc_worker.py"
    esmc_log = (SCRIPT_DIR / "bench_esmc_worker.log").open("w", encoding="utf-8")
    esmc_proc = subprocess.Popen(
        [sys.executable, str(esmc_script), "--host", "127.0.0.1", "--port", str(esmc_port), "--threads", "20"],
        stdout=esmc_log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    procs.append(esmc_proc)
    print(f"[bench] Booting standalone ESM-C 600M worker on 127.0.0.1:{esmc_port} (pid={esmc_proc.pid})...", flush=True)
    t_e0 = time.time()
    esmc_ok = False
    while time.time() - t_e0 < 180.0:
        if esmc_proc.poll() is not None:
            esmc_log.close()
            tail = (SCRIPT_DIR / "bench_esmc_worker.log").read_text(encoding="utf-8", errors="replace")[-1200:]
            raise RuntimeError(f"esmc_worker_standalone exited early (code={esmc_proc.returncode}):\n{tail}")
        try:
            with socket.create_connection(("127.0.0.1", esmc_port), timeout=1.0):
                esmc_ok = True
                break
        except Exception:
            time.sleep(1.0)
    if not esmc_ok:
        raise TimeoutError("esmc_worker_standalone did not bind within 180s")
    print(f"[bench] Standalone ESM-C 600M worker ready in {time.time()-t_e0:.1f}s", flush=True)

    svc_script = V33_ROOT / "bundle" / "v33_inference_service.py"
    log_path = SCRIPT_DIR / "bench_v33_service.log"
    log_f = log_path.open("w", encoding="utf-8")
    runner_py = SCRIPT_DIR / "_bench_v33_runner.py"
    runner_py.write_text(
        f"""import pathlib, runpy, sys
import joblib
import sklearn.ensemble, sklearn.linear_model, sklearn.isotonic, sklearn.preprocessing, lightgbm, xgboost

_orig_load = joblib.load
def _patched_load(p, *a, **kw):
    ps = str(p)
    if ps.endswith("_cpu.joblib"):
        ep = pathlib.Path(ps).stem[:-4]
        alt = pathlib.Path("/data/qlyu/v33_upgrade/models_cpu") / ep / "models_5fold.joblib"
        if alt.exists():
            return _orig_load(alt)
    return _orig_load(p, *a, **kw)

joblib.load = _patched_load
sys.argv = [{repr(str(svc_script))}, "--host", "127.0.0.1", "--port", {repr(str(port))}, "--esmc-port", {repr(str(esmc_port))}, "--threads", "20", "--max-concurrent", {repr(str(max(2, BENCH_WORKERS)))}]
runpy.run_path({repr(str(svc_script))}, run_name="__main__")
""",
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, str(runner_py)],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
    )
    procs.append(proc)
    print(f"[bench] Booting live 4-FM V33 production service on {url} (pid={proc.pid}, log={log_path})...", flush=True)
    t0 = time.time()
    while time.time() - t0 < 420.0:
        if proc.poll() is not None:
            log_f.close()
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
            raise RuntimeError(f"v33_inference_service exited early (code={proc.returncode}):\n{tail}")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2.0) as resp:
                if resp.status == 200:
                    h = json.loads(resp.read().decode("utf-8"))
                    if h.get("ok"):
                        print(f"[bench] Live V33 service ready in {time.time()-t0:.1f}s ({h.get('endpoints')} endpoints, version={h.get('version')})", flush=True)
                        return procs, url
        except Exception:
            pass
        time.sleep(2.0)
    for p in procs:
        p.terminate()
    raise TimeoutError("v33_inference_service did not become healthy within 420s")


# ---------------------------------------------------------------------------
# 4. Baseline Implementations (Equal Budget, Symmetric Free Stage A Pre-Filter, Same Live Oracle)
# ---------------------------------------------------------------------------

def _evaluate_candidates_under_gatekeeper(
    adapter: OmniPanelAdapter,
    lead: Dict[str, Any],
    cand_specs: List[Tuple[_CandidateSpec, bool, Optional[str]]],
    *,
    budget: int,
    unconstrained_selection: bool = False,
    pre_filter_stage_a: bool = True,
) -> Dict[str, Any]:
    """Evaluate a baseline's proposed candidates under the exact same live Oracle and budget,
    scoring every candidate with BOTH the method's selection criterion AND the authoritative
    post-hoc fail-closed `DeterministicGatekeeper`.

    - When `pre_filter_stage_a=True` (used for gated baselines `StrongBase` and `Random-EqualBudget`),
      RDKit Stage A runs FREE before Oracle calls — 100% symmetric with `run_peptide_harness`,
      while accurately recording `n_stage_a_rejected` so `stage_a_pass_rate` reflects true compliance.
    - When `pre_filter_stage_a=False` (`Unconstrained-Greedy` reward-hacking control), Stage A is
      evaluated strictly post-hoc after spending Oracle budget.
    """
    resolved = adapter.resolve_query(sequence=lead.get("sequence"), smiles=lead.get("smiles"))
    assert resolved.mol is not None
    parent_can = _sanitize_and_canonicalize(resolved.mol)
    assert parent_can is not None
    core_lock = resolve_core_locks(
        parent_mol=resolved.mol,
        parent_sequence=resolved.sequence,
        locked_positions=lead.get("locked_positions") or [],
        lock_disulfides=True,
        functional_endpoint=lead.get("functional_endpoint"),
    )
    parent_bio = compute_biophysical_profile(core_lock.parent_sequence, resolved.mol)
    is_linear = core_lock.parent_sequence is not None
    dir_x = default_direction_for_endpoint(lead["objective_x"])
    dir_y = default_direction_for_endpoint(lead["objective_y"])
    ep_docs = adapter.manifest.get("endpoints") or {}
    req_eps = [
        lead["objective_x"],
        lead["objective_y"],
        "PAMPA",
        "Caco2",
        "half_life_reg",
        "hemolysis_cls",
        "hemolysis_HC50",
        "overall_peptide_toxicity",
        "cytotoxicity_cls",
        "cytotoxicity_CC50",
    ]
    if lead.get("functional_endpoint"):
        req_eps.append(lead["functional_endpoint"])
    include_tier2 = any(int((ep_docs.get(ep) or {}).get("tier", 1)) == 2 for ep in req_eps)

    # First, create a lightweight Stage-A-capable Gatekeeper placeholder (or run Stage A with a minimal stub)
    # Wait: `check_stage_a` only needs `core_lock` and `parent_biophysical`, NOT parent_preds_by_track!
    # So we first determine which tracks we actually need, evaluate parent on ONLY needed tracks (never wasting 1 eval!),
    # then run Stage A pre-filter if `pre_filter_stage_a=True`.
    need_dual = is_linear and any((sp.sequence is not None) for sp, _, _ in cand_specs)
    need_smiles = (not is_linear) or any((sp.sequence is None) for sp, _, _ in cand_specs)
    active_tracks: List[str] = []
    if need_dual or is_linear:
        active_tracks.append("dual_modality")
    if need_smiles:
        active_tracks.append("smiles_only")

    parent_rows: List[BatchParsedRow] = []
    if "dual_modality" in active_tracks:
        parent_rows.append(
            BatchParsedRow(
                index=0,
                name=f"{lead['lead_id']}_dual",
                detected_type="sequence",
                sequence=core_lock.parent_sequence,
                smiles=parent_can,
                valid=True,
            )
        )
    if "smiles_only" in active_tracks:
        parent_rows.append(
            BatchParsedRow(
                index=len(parent_rows),
                name=f"{lead['lead_id']}_smiles",
                detected_type="smiles",
                sequence=None,
                smiles=parent_can,
                valid=True,
            )
        )
    p_bms = adapter.predict_batch(parent_rows, include_tier2=include_tier2, conditions=AssayConditions())
    evals_used = len(parent_rows)
    parent_preds_by_track = {tr: {p.endpoint: p for p in bm.predictions} for tr, bm in zip(active_tracks, p_bms)}

    gk = DeterministicGatekeeper(
        core_lock=core_lock,
        parent_biophysical=parent_bio,
        parent_preds_by_track=parent_preds_by_track,
        objective_x=lead["objective_x"],
        objective_y=lead["objective_y"],
        functional_endpoint=lead.get("functional_endpoint"),
        weights=MpoWeights(),
    )
    parent_summary: Dict[str, Dict[str, Any]] = {}
    for tr in active_tracks:
        sb_p = gk.check_stage_b(list(parent_preds_by_track[tr].values()), tr)
        _, xr, _ = _extract_raw_and_display(parent_preds_by_track[tr], lead["objective_x"])
        _, yr, _ = _extract_raw_and_display(parent_preds_by_track[tr], lead["objective_y"])
        parent_summary[tr] = {"gmpo": sb_p.gmpo_score, "xr": xr, "yr": yr}

    rem_budget = max(0, budget - evals_used)
    seen: set[Tuple[str, Optional[str]]] = {(parent_can, core_lock.parent_sequence)}
    if "smiles_only" in active_tracks:
        seen.add((parent_can, None))

    chosen_with_sa: List[Tuple[_CandidateSpec, bool, Optional[str], Any]] = []
    n_stage_a_rejected = 0
    sa_rejected_keys: set[Tuple[str, Optional[str]]] = set()
    if pre_filter_stage_a:
        for sp, is_leaf, lin_seq in cand_specs:
            if len(chosen_with_sa) >= rem_budget:
                break
            sa = gk.check_stage_a(
                sp.smiles,
                sp.sequence,
                is_chemical_leaf=is_leaf,
                seen_keys=seen,
                lineage_sequence=lin_seq,
                mutation_label=sp.mutation_label,
            )
            if not sa.passed:
                n_stage_a_rejected += 1
                sa_rejected_keys.add((sa.canonical_smiles or sp.smiles, sp.sequence))
                continue
            seen.add((sa.canonical_smiles or sp.smiles, sp.sequence))
            chosen_with_sa.append((sp, is_leaf, lin_seq, sa))
    else:
        for sp, is_leaf, lin_seq in cand_specs[:rem_budget]:
            chosen_with_sa.append((sp, is_leaf, lin_seq, None))

    rows = [
        BatchParsedRow(
            index=i,
            name=sp.mutation_label,
            detected_type="sequence" if sp.sequence else "smiles",
            sequence=sp.sequence,
            smiles=(sa.canonical_smiles if (sa is not None and sa.canonical_smiles) else sp.smiles),
            valid=True,
        )
        for i, (sp, _, _, sa) in enumerate(chosen_with_sa)
    ]
    bms: List[BatchMoleculeResult] = []
    for st in range(0, len(rows), 4):
        bms.extend(adapter.predict_batch(rows[st : st + 4], include_tier2=include_tier2, conditions=AssayConditions()))
    evals_used += len(rows)

    records: List[Dict[str, Any]] = []
    feasible_pool: List[Dict[str, Any]] = [
        {"is_parent": True, "delta_x_raw": 0.0, "delta_y_raw": 0.0, "delta_gmpo": 0.0, "gmpo_score": parent_summary[gk.primary_track]["gmpo"]}
    ]

    for (sp, is_leaf, lin_seq, pre_sa), bm in zip(chosen_with_sa, bms):
        if bm.status != "ok":
            continue
        if pre_sa is not None:
            sa = pre_sa
        else:
            sa = gk.check_stage_a(
                sp.smiles,
                sp.sequence,
                is_chemical_leaf=is_leaf,
                seen_keys=seen,
                lineage_sequence=lin_seq,
                mutation_label=sp.mutation_label,
            )
            seen.add((sa.canonical_smiles or sp.smiles, sp.sequence))
            if not sa.passed:
                n_stage_a_rejected += 1
        tr = "dual_modality" if (sp.sequence is not None and "dual_modality" in parent_summary) else "smiles_only"
        sb = gk.check_stage_b(bm.predictions, tr)
        by_ep = {p.endpoint: p for p in bm.predictions}
        _, xr, _ = _extract_raw_and_display(by_ep, lead["objective_x"])
        _, yr, _ = _extract_raw_and_display(by_ep, lead["objective_y"])
        p_tr = parent_summary[tr]
        dx = (xr - p_tr["xr"]) if (xr is not None and p_tr["xr"] is not None) else None
        dy = (yr - p_tr["yr"]) if (yr is not None and p_tr["yr"] is not None) else None
        d_gmpo = round(sb.gmpo_score - p_tr["gmpo"], 2)

        # Unconstrained greedy selection score: includes mechanism_anchor & ignores Stage A/B gates!
        sel_score = d_gmpo
        if unconstrained_selection:
            cmol = parse_smiles_safe(sp.smiles)
            anch = detect_acylation_anchor(cmol) if cmol is not None else None
            raw_mpo = compute_peptide_mpo(bm.predictions, weights=MpoWeights(), mechanism_anchor=anch)["peptide_mpo_score"]
            sel_score = round(float(raw_mpo) - p_tr["gmpo"], 2)

        feasible = sa.passed and sb.passed
        rec = {
            "is_parent": False,
            "mutation_label": sp.mutation_label,
            "lineage_path": ["Parent", sp.mutation_label],
            "lineage_sequence": lin_seq,
            "strategy": sp.strategy,
            "sequence": sp.sequence,
            "smiles": sa.canonical_smiles or sp.smiles,
            "is_chemical_leaf": is_leaf,
            "modality_track": tr,
            "stage_a_passed": sa.passed,
            "stage_a_rejects": list(sa.reject_codes),
            "stage_b_passed": sb.passed,
            "stage_b_rejects": list(sb.reject_codes),
            "feasible": feasible,
            "gmpo_score": sb.gmpo_score,
            "delta_gmpo": d_gmpo,
            "selection_score": sel_score,
            "delta_x_raw": dx,
            "delta_y_raw": dy,
            "hemolysis_cls": float(by_ep["hemolysis_cls"].value) if "hemolysis_cls" in by_ep and by_ep["hemolysis_cls"].value is not None else None,
            "ad_overall": sb.ad_overall,
        }
        records.append(rec)
        if feasible:
            feasible_pool.append(rec)

    _, hv = compute_pareto_flags_and_hv(feasible_pool, dir_x, dir_y, REF_DX, REF_DY)
    if unconstrained_selection:
        ranked = sorted(records, key=lambda r: float(r["selection_score"]), reverse=True)
        top1 = ranked[0] if ranked else None
    else:
        feas_improved = [r for r in records if r["feasible"] and float(r["delta_gmpo"]) > 0.0]
        feas_improved.sort(key=lambda r: float(r["delta_gmpo"]), reverse=True)
        top1 = feas_improved[0] if feas_improved else None

    return {
        "records": records,
        "feasible_pareto_hv": hv,
        "top1": top1,
        "evals_used": evals_used,
        "parent_summary": parent_summary,
        "parent_hemolysis_cls": float(parent_preds_by_track[gk.primary_track]["hemolysis_cls"].value),
        "gatekeeper": gk,
        "n_stage_a_rejected": n_stage_a_rejected,
        "sa_rejected_keys": sa_rejected_keys,
        "seen_keys": seen,
        "include_tier2": include_tier2,
    }


def run_random_baseline(adapter: OmniPanelAdapter, lead: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Random Equal-Budget baseline with symmetric free Stage A pre-filtering across all 5 strategies."""
    resolved = adapter.resolve_query(sequence=lead.get("sequence"), smiles=lead.get("smiles"))
    assert resolved.mol is not None
    parent_can = _sanitize_and_canonicalize(resolved.mol)
    assert parent_can is not None
    core_lock = resolve_core_locks(
        parent_mol=resolved.mol,
        parent_sequence=resolved.sequence,
        locked_positions=lead.get("locked_positions") or [],
        lock_disulfides=True,
        functional_endpoint=lead.get("functional_endpoint"),
    )
    mem = NegativeSampleMemory()
    pool: List[Tuple[_CandidateSpec, bool, Optional[str]]] = []
    if core_lock.parent_sequence is not None:
        seq_cands = generate_sequence_candidates(
            core_lock.parent_sequence, parent_can, core_lock, mem, strategies=("ala_scan", "point_mutation"), limit=40
        )
        pool.extend([(c, False, c.sequence) for c in seq_cands])
    leaf_cands = generate_chemical_leaf_candidates(
        resolved.mol, core_lock.parent_sequence, parent_can, core_lock, mem, strategies=tuple(CHEM_LEAF_STRATEGIES), limit=12
    )
    pool.extend([(c, True, core_lock.parent_sequence) for c in leaf_cands])
    rng.shuffle(pool)
    return _evaluate_candidates_under_gatekeeper(
        adapter, lead, pool, budget=EQUAL_BUDGET_EVALS, unconstrained_selection=False, pre_filter_stage_a=True
    )


def run_unconstrained_greedy_baseline(adapter: OmniPanelAdapter, lead: Dict[str, Any]) -> Dict[str, Any]:
    """Reward-hacking control: maximizes raw MPO WITH mechanism_anchor=True and NO Stage A/B gates!"""
    resolved = adapter.resolve_query(sequence=lead.get("sequence"), smiles=lead.get("smiles"))
    assert resolved.mol is not None
    parent_can = _sanitize_and_canonicalize(resolved.mol)
    assert parent_can is not None
    core_lock = resolve_core_locks(
        parent_mol=resolved.mol,
        parent_sequence=resolved.sequence,
        locked_positions=lead.get("locked_positions") or [],
        lock_disulfides=True,
        functional_endpoint=lead.get("functional_endpoint"),
    )
    mem = NegativeSampleMemory()
    pool: List[Tuple[_CandidateSpec, bool, Optional[str]]] = []
    leaf_cands = generate_chemical_leaf_candidates(
        resolved.mol, core_lock.parent_sequence, parent_can, core_lock, mem,
        strategies=("lipidation_c16_c18", "n_methylation", "cyclization"), limit=12
    )
    pool.extend([(c, True, core_lock.parent_sequence) for c in leaf_cands])
    if core_lock.parent_sequence is not None:
        seq_cands = generate_sequence_candidates(
            core_lock.parent_sequence, parent_can, core_lock, mem, strategies=("point_mutation", "ala_scan"), limit=30
        )
        pool.extend([(c, False, c.sequence) for c in seq_cands])
    return _evaluate_candidates_under_gatekeeper(
        adapter, lead, pool, budget=EQUAL_BUDGET_EVALS, unconstrained_selection=True, pre_filter_stage_a=False
    )


def run_exhaustive_1step_greedy_combine(adapter: OmniPanelAdapter, lead: Dict[str, Any]) -> Dict[str, Any]:
    """TRUE STRONG SIMPLE BASELINE (Red Line 7 + R3 P0-1 Fix):
    - Uses the EXACT same free Stage A pre-filter, Stage B fail-closed gate, and Pure-ML Feas-gMPO as Harness!
    - Phase 1 (~15 evals on linear leads) allocates a BALANCED cross-strategy 1-step budget across ALL 5 operator families:
      * 4 Stage-A-passing chemical leaves (round-robin across `n_methylation`, `cyclization`, `lipidation_c16_c18`), PLUS
      * 11 Stage-A-passing 1-step sequence edits (50/50 interleaved `point_mutation` + `ala_scan`).
    - Phase 2 (remaining 3 evals) greedily stacks top-2 individually beneficial 1-step sequence winners (`greedy_combine`),
      and if < 3 stacks can be formed, automatically reinvests leftover slots into additional Stage-A-passing Phase 1 candidates!
    """
    resolved = adapter.resolve_query(sequence=lead.get("sequence"), smiles=lead.get("smiles"))
    assert resolved.mol is not None
    parent_can = _sanitize_and_canonicalize(resolved.mol)
    assert parent_can is not None
    core_lock = resolve_core_locks(
        parent_mol=resolved.mol,
        parent_sequence=resolved.sequence,
        locked_positions=lead.get("locked_positions") or [],
        lock_disulfides=True,
        functional_endpoint=lead.get("functional_endpoint"),
    )
    mem = NegativeSampleMemory()

    # Cyclic macrocycle (`chemical_leaf_only`): spend full budget on round-robin Stage-A-passing leaves
    if core_lock.parent_sequence is None:
        leaf_1step = generate_chemical_leaf_candidates(
            resolved.mol, None, parent_can, core_lock, mem,
            strategies=tuple(CHEM_LEAF_STRATEGIES), limit=EQUAL_BUDGET_EVALS * 2
        )
        phase1_specs = [(c, True, None) for c in leaf_1step]
        return _evaluate_candidates_under_gatekeeper(
            adapter, lead, phase1_specs, budget=EQUAL_BUDGET_EVALS, unconstrained_selection=False, pre_filter_stage_a=True
        )

    # Linear peptide: build balanced cross-strategy Phase 1 pool spanning ALL 5 operators!
    # `generate_sequence_candidates` now interleaves point_mutation & ala_scan 50/50!
    seq_1step = generate_sequence_candidates(
        core_lock.parent_sequence, parent_can, core_lock, mem, strategies=("point_mutation", "ala_scan"), limit=24
    )
    leaf_1step = generate_chemical_leaf_candidates(
        resolved.mol, core_lock.parent_sequence, parent_can, core_lock, mem,
        strategies=tuple(CHEM_LEAF_STRATEGIES), limit=12
    )
    # Interleave 4 leaves across the front 15 slots so EVEN IF Stage A rejects some leaves, StrongBase STILL
    # evaluates up to 4 valid leaves across n_methylation / cyclization / lipidation_c16_c18 AND ~11 seq edits!
    # Specifically: put up to 4 Stage-A-valid leaves FIRST (via a quick pre-screen with a dummy GK? No: interleave
    # 1 leaf every 3 sequence edits, plus keep overflow tails for Phase 2 fallback!).
    seq_wrapped = [(c, False, c.sequence) for c in seq_1step]
    leaf_wrapped = [(c, True, core_lock.parent_sequence) for c in leaf_1step]

    # Put 1 leaf per 2 sequence specs -> in the first 15 Stage-A-passing slots, we get ~5 chemical leaves
    # (covering all 3 chem families!) and ~10 balanced point_mutation + ala_scan 1-step edits!
    # Wait: let's guarantee EXACT balanced mix: interleave [leaf, seq, seq] until 4 leaves or leaf pool exhausted,
    # then append remaining seq_wrapped, then remaining leaf_wrapped!
    phase1_ordered: List[Tuple[_CandidateSpec, bool, Optional[str]]] = []
    li, si = 0, 0
    target_leaves_front = min(6, len(leaf_wrapped))  # generous overfetch for Stage A filtering (e.g. Exenatide charge filter)
    while li < target_leaves_front or si < len(seq_wrapped):
        if li < target_leaves_front:
            phase1_ordered.append(leaf_wrapped[li])
            li += 1
        for _ in range(2):
            if si < len(seq_wrapped):
                phase1_ordered.append(seq_wrapped[si])
                si += 1
    phase1_ordered.extend(leaf_wrapped[li:])

    # Phase 1 budget = 17 (2 parent tracks + 15 candidates), reserving 3 slots for Phase 2 greedy_combine!
    p1_budget = EQUAL_BUDGET_EVALS - 3
    p1 = _evaluate_candidates_under_gatekeeper(
        adapter, lead, phase1_ordered, budget=p1_budget, unconstrained_selection=False, pre_filter_stage_a=True
    )
    rem = max(0, EQUAL_BUDGET_EVALS - p1["evals_used"])
    if rem <= 0:
        return p1

    # Phase 2: Greedily combine top individually beneficial 1-step feasible edits!
    mut_re = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
    beneficial: List[Tuple[int, str, float]] = []
    seen_local: set[Tuple[str, Optional[str]]] = set(p1["seen_keys"])
    mem2 = NegativeSampleMemory()
    mem2.seen_keys = set(seen_local)
    for r in p1["records"]:
        if r["feasible"] and r["sequence"] is not None and float(r["delta_gmpo"]) > 0.0:
            m = mut_re.match(r["mutation_label"].strip())
            if m:
                beneficial.append((int(m.group(2)), m.group(3), float(r["delta_gmpo"])))

    gk: DeterministicGatekeeper = p1["gatekeeper"]
    p_sum = p1["parent_summary"]
    include_tier2 = bool(p1["include_tier2"])

    comb_raw = _build_combinatorial_stack_candidates(core_lock.parent_sequence, beneficial, core_lock, mem2, limit=rem * 2)
    p2_chosen: List[Tuple[_CandidateSpec, bool, Optional[str], Any, str]] = []
    n_sa_rej_p2 = 0
    for sp in comb_raw:
        if len(p2_chosen) >= rem:
            break
        sa = gk.check_stage_a(
            sp.smiles, sp.sequence, is_chemical_leaf=False, seen_keys=seen_local,
            lineage_sequence=sp.sequence, mutation_label=sp.mutation_label
        )
        if not sa.passed:
            n_sa_rej_p2 += 1
            continue
        seen_local.add((sa.canonical_smiles or sp.smiles, sp.sequence))
        p2_chosen.append((sp, False, sp.sequence, sa, "greedy_combine"))

    # If Phase 2 produced fewer than `rem` valid combinations, reinvest leftover budget into unused Phase 1 specs!
    p1_sa_rej_keys: set[Tuple[str, Optional[str]]] = set(p1.get("sa_rejected_keys") or set())
    if len(p2_chosen) < rem:
        for sp, is_leaf, lin_seq in phase1_ordered:
            if len(p2_chosen) >= rem:
                break
            if (sp.smiles, sp.sequence) in p1_sa_rej_keys:
                continue
            sa = gk.check_stage_a(
                sp.smiles, sp.sequence, is_chemical_leaf=is_leaf, seen_keys=seen_local,
                lineage_sequence=lin_seq, mutation_label=sp.mutation_label
            )
            if not sa.passed:
                k_rej = (sa.canonical_smiles or sp.smiles, sp.sequence)
                if k_rej not in p1_sa_rej_keys and not any(c in ("DUPLICATE_CANDIDATE", "IDENTICAL_TO_PARENT") for c in sa.reject_codes):
                    n_sa_rej_p2 += 1
                    p1_sa_rej_keys.add(k_rej)
                continue
            seen_local.add((sa.canonical_smiles or sp.smiles, sp.sequence))
            p2_chosen.append((sp, is_leaf, lin_seq, sa, sp.strategy))

    if not p2_chosen:
        p1["n_stage_a_rejected"] = int(p1.get("n_stage_a_rejected", 0)) + n_sa_rej_p2
        return p1

    rows = [
        BatchParsedRow(
            index=i,
            name=sp.mutation_label,
            detected_type="sequence" if sp.sequence else "smiles",
            sequence=sp.sequence,
            smiles=sa.canonical_smiles or sp.smiles,
            valid=True,
        )
        for i, (sp, _, _, sa, _) in enumerate(p2_chosen)
    ]
    bms = adapter.predict_batch(rows, include_tier2=include_tier2, conditions=AssayConditions())
    all_recs = list(p1["records"])
    for (sp, is_leaf, lin_seq, sa, strat_tag), bm in zip(p2_chosen, bms):
        if bm.status != "ok":
            continue
        tr = "dual_modality" if (sp.sequence is not None and "dual_modality" in p_sum) else "smiles_only"
        sb = gk.check_stage_b(bm.predictions, tr)
        by_ep = {p.endpoint: p for p in bm.predictions}
        _, xr, _ = _extract_raw_and_display(by_ep, lead["objective_x"])
        _, yr, _ = _extract_raw_and_display(by_ep, lead["objective_y"])
        dx = (xr - p_sum[tr]["xr"]) if (xr is not None and p_sum[tr]["xr"] is not None) else None
        dy = (yr - p_sum[tr]["yr"]) if (yr is not None and p_sum[tr]["yr"] is not None) else None
        all_recs.append(
            {
                "is_parent": False,
                "mutation_label": sp.mutation_label,
                "lineage_path": ["Parent", sp.mutation_label],
                "lineage_sequence": lin_seq,
                "strategy": strat_tag,
                "sequence": sp.sequence,
                "smiles": sa.canonical_smiles or sp.smiles,
                "is_chemical_leaf": is_leaf,
                "modality_track": tr,
                "stage_a_passed": sa.passed,
                "stage_a_rejects": list(sa.reject_codes),
                "stage_b_passed": sb.passed,
                "stage_b_rejects": list(sb.reject_codes),
                "feasible": sa.passed and sb.passed,
                "gmpo_score": sb.gmpo_score,
                "delta_gmpo": round(sb.gmpo_score - p_sum[tr]["gmpo"], 2),
                "selection_score": round(sb.gmpo_score - p_sum[tr]["gmpo"], 2),
                "delta_x_raw": dx,
                "delta_y_raw": dy,
                "hemolysis_cls": float(by_ep["hemolysis_cls"].value) if "hemolysis_cls" in by_ep and by_ep["hemolysis_cls"].value is not None else None,
                "ad_overall": sb.ad_overall,
            }
        )

    feas_pool = [{"is_parent": True, "delta_x_raw": 0.0, "delta_y_raw": 0.0, "delta_gmpo": 0.0}] + [r for r in all_recs if r["feasible"]]
    dir_x = default_direction_for_endpoint(lead["objective_x"])
    dir_y = default_direction_for_endpoint(lead["objective_y"])
    _, hv = compute_pareto_flags_and_hv(feas_pool, dir_x, dir_y, REF_DX, REF_DY)
    feas_imp = sorted([r for r in all_recs if r["feasible"] and float(r["delta_gmpo"]) > 0.0], key=lambda r: float(r["delta_gmpo"]), reverse=True)
    return {
        "records": all_recs,
        "feasible_pareto_hv": hv,
        "top1": feas_imp[0] if feas_imp else None,
        "evals_used": p1["evals_used"] + len(rows),
        "parent_summary": p_sum,
        "parent_hemolysis_cls": p1["parent_hemolysis_cls"],
        "gatekeeper": gk,
        "n_stage_a_rejected": int(p1.get("n_stage_a_rejected", 0)) + n_sa_rej_p2,
    }


def run_harness_variant(
    adapter: OmniPanelAdapter,
    lead: Dict[str, Any],
    *,
    use_negative_memory: bool = True,
    use_ad_safety_gate: bool = True,
    strategies: Tuple[str, ...] = ALL_HARNESS_STRATEGIES,
    compute_cross_track: bool = True,
) -> Dict[str, Any]:
    cfg = HarnessConfig(
        objective_x=lead["objective_x"],
        objective_y=lead["objective_y"],
        functional_endpoint=lead.get("functional_endpoint"),
        strategies=strategies,
        locked_positions=tuple(lead.get("locked_positions") or []),
        lock_disulfides=True,
        rounds=3,
        population_size=6,
        top_k=5,
        critic_max_perturbations=5,
        max_total_evals=EQUAL_BUDGET_EVALS,
        wall_clock_budget_s=1800.0,
        chunk_size=4,
        require_remote_oracle=True,
        compute_elite_cross_track_audit=compute_cross_track,
        use_negative_memory=use_negative_memory,
        use_ad_safety_gate=use_ad_safety_gate,
        use_critic=True,
    )
    res = run_peptide_harness(
        adapter,
        sequence=lead.get("sequence"),
        smiles=lead.get("smiles"),
        config=cfg,
    )
    assert res["oracle_is_surrogate_heuristic"] is False, "Benchmark MUST run on real v33_remote oracle!"
    assert res["oracle_mode"] == "v33_remote"

    # Re-audit every evaluated candidate's TRUE Stage B pass status (`stage_b_passed` & `reject_codes`),
    # read `key_metrics` (P1-2 fix!) and preserve full `lineage_path` & `lineage_sequence` (P2-1 fix!).
    cands = res["candidates"]
    recs: List[Dict[str, Any]] = []
    for c in cands:
        sb_codes = list(c.get("reject_codes") or [])
        true_sb_pass = bool(c.get("stage_b_passed", len(sb_codes) == 0))
        km = c.get("key_metrics") or {}
        hem_val = (km.get("hemolysis_cls") or {}).get("value")
        recs.append(
            {
                "is_parent": False,
                "mutation_label": c["mutation_label"],
                "lineage_path": list(c.get("lineage_path") or ["Parent", c["mutation_label"]]),
                "lineage_sequence": c.get("lineage_sequence"),
                "strategy": c["strategy"],
                "sequence": c["sequence"],
                "smiles": c["smiles"],
                "is_chemical_leaf": c["is_chemical_leaf"],
                "modality_track": c["modality_track"],
                "stage_a_passed": True,
                "stage_a_rejects": [],
                "stage_b_passed": true_sb_pass,
                "stage_b_rejects": sb_codes,
                "feasible": true_sb_pass,
                "gmpo_score": c["gmpo_score"],
                "delta_gmpo": c["delta_gmpo"],
                "delta_x_raw": c["delta_x_raw"],
                "delta_y_raw": c["delta_y_raw"],
                "hemolysis_cls": float(hem_val) if hem_val is not None else None,
                "ad_overall": c["ad_overall"],
            }
        )

    dir_x = default_direction_for_endpoint(lead["objective_x"])
    dir_y = default_direction_for_endpoint(lead["objective_y"])
    feas_pool = [{"is_parent": True, "delta_x_raw": 0.0, "delta_y_raw": 0.0, "delta_gmpo": 0.0}] + [r for r in recs if r["feasible"]]
    _, hv = compute_pareto_flags_and_hv(feas_pool, dir_x, dir_y, REF_DX, REF_DY)

    if use_ad_safety_gate:
        top1_raw = res["final_recommendations"][0] if res["final_recommendations"] else None
        top1 = next(
            (r for r in recs if top1_raw and r["smiles"] == top1_raw["smiles"] and r["sequence"] == top1_raw["sequence"]),
            None,
        )
    else:
        # Un-gated variant selects highest delta_gmpo regardless of Stage B, then we audit whether its Top-1 is actually feasible!
        best_any = sorted(recs, key=lambda r: float(r["delta_gmpo"]), reverse=True)
        top1 = best_any[0] if best_any else None

    p_km = res["parent"].get("key_metrics") or {}
    parent_hem = (p_km.get("hemolysis_cls") or {}).get("value")
    return {
        "records": recs,
        "feasible_pareto_hv": hv,
        "top1": top1,
        "evals_used": res["metrics"]["evals_used"],
        "cross_track_audit": res.get("cross_track_epistemic_audit") or [],
        "harness_mode": res["harness_mode"],
        "activity_unmodeled": res["activity_unmodeled"],
        "parent_hemolysis_cls": float(parent_hem) if parent_hem is not None else None,
        "n_stage_a_rejected": sum(int(rd.get("n_stage_a_rejected", 0)) for rd in res.get("rounds") or []),
    }


# ---------------------------------------------------------------------------
# 5. External HemoPI2 v1.3 Cross-Check (Canonical Sequences ONLY, sklearn 1.3.1)
# ---------------------------------------------------------------------------

def score_sequences_hemopi2(seqs: List[str]) -> Dict[str, Optional[float]]:
    """Score unique canonical 20-AA sequences with official pretrained HemoPI2 v1.3 (`seq[:40]`)."""
    aa20 = set("ACDEFGHIKLMNPQRSTVWY")
    clean_unique = sorted({s.strip().upper() for s in seqs if s and set(s.strip().upper()).issubset(aa20)})
    if not clean_unique or not (HEMOPI_MODEL_DIR / "hemopi2_ml_clf.sav").exists():
        return {}
    trunc_seqs = [s[:40] for s in clean_unique]
    with tempfile.TemporaryDirectory(prefix="v35_hemopi2_") as tmpd:
        tmp = Path(tmpd)
        (tmp / "seqs.json").write_text(json.dumps({"orig": clean_unique, "trunc": trunc_seqs}), encoding="utf-8")
        helper = tmp / "score_hemopi2.py"
        helper.write_text(
            f"""
import sys
sys.path.insert(0, {repr(HEMOPI_SKLEARN_PY_PATH)})
import json, pickle, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_DIR = Path({repr(str(HEMOPI_MODEL_DIR))})
tmp = Path({repr(str(tmp))})
data = json.loads((tmp / "seqs.json").read_text(encoding="utf-8"))
orig = data["orig"]
trunc = data["trunc"]

seq_f = tmp / "input.seq"
seq_f.write_text("\\n".join(trunc) + "\\n", encoding="utf-8")
wdir = tmp / "work"
wdir.mkdir(exist_ok=True)
out_csv = tmp / "feats.csv"
proc = subprocess.run(
    [sys.executable, str(MODEL_DIR / "composition_calculate.py"), str(seq_f), str(wdir), str(out_csv)],
    cwd=str(MODEL_DIR.parent),
    capture_output=True,
    text=True,
)
if proc.returncode != 0:
    raise RuntimeError(f"composition_calculate.py failed: {{proc.stderr[-500:]}}")
clf = pickle.load(open(MODEL_DIR / "hemopi2_ml_clf.sav", "rb"))
feat_cols = list(clf.feature_names_in_)
df = pd.read_csv(out_csv)
for c in feat_cols:
    if c not in df.columns:
        df[c] = 0.0
X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
probs = clf.predict_proba(X)[:, 1].tolist()
(tmp / "probs.json").write_text(json.dumps(dict(zip(orig, [round(float(p), 4) for p in probs]))), encoding="utf-8")
""",
            encoding="utf-8",
        )
        hemopi_py = "/home/qlyu/anaconda3/envs/cuda12.1/bin/python"
        if not Path(hemopi_py).exists():
            hemopi_py = sys.executable
        proc = subprocess.run([hemopi_py, str(helper)], capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[bench] WARNING HemoPI2 v1.3 failed: {proc.stderr[-400:]}", flush=True)
            return {}
        return json.loads((tmp / "probs.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 6. Statistical Helpers (Paired Wilcoxon + Bootstrap 95% CI + Holm-Bonferroni)
# ---------------------------------------------------------------------------

def paired_stats(a: List[float], b: List[float], seed: int = BENCH_SEED) -> Dict[str, Any]:
    xa = np.array(a, dtype=np.float64)
    xb = np.array(b, dtype=np.float64)
    diff = xa - xb
    n = len(diff)
    rng = np.random.RandomState(seed)
    boots = [float(np.mean(diff[rng.randint(0, n, size=n)])) for _ in range(2000)]
    ci95 = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
    if np.all(np.abs(diff) < 1e-12):
        p_wilcox = 1.0
    else:
        try:
            _, p_wilcox = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
            p_wilcox = float(p_wilcox)
        except Exception:
            p_wilcox = 1.0
    return {
        "n_leads": n,
        "mean_delta": round(float(np.mean(diff)), 4),
        "median_delta": round(float(np.median(diff)), 4),
        "wins_ties_losses": [
            int(np.sum(diff > 1e-9)),
            int(np.sum(np.abs(diff) <= 1e-9)),
            int(np.sum(diff < -1e-9)),
        ],
        "bootstrap_ci95": ci95,
        "wilcoxon_p_raw": round(p_wilcox, 4),
    }


def holm_bonferroni(p_map: Dict[str, float]) -> Dict[str, float]:
    items = sorted(p_map.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, float] = {}
    rmax = 0.0
    for i, (k, p) in enumerate(items):
        rmax = max(rmax, min(1.0, p * (m - i)))
        out[k] = round(rmax, 4)
    return out


# ---------------------------------------------------------------------------
# 7. Per-Run Summary (Fixes P0-2, P1-1, P1-2, P2-1!)
# ---------------------------------------------------------------------------

def summarize_method_run(out: Dict[str, Any]) -> Dict[str, Any]:
    recs = out["records"]
    n_eval = len(recs)
    n_sa_passed = sum(1 for r in recs if r["stage_a_passed"])
    # P1-2 FIX: TRUE Stage A pass rate incorporating pre-filtered Stage A rejections!
    n_sa_rej = int(out.get("n_stage_a_rejected", 0))
    n_sa_total = n_sa_passed + n_sa_rej if n_sa_passed == n_eval else n_eval
    n_feas = sum(1 for r in recs if r["feasible"])
    # P1-1 FIX: Actual "OOD_DRIFT:" prefix vs raw `ad_overall == "ood"` (parent-inherent OOD)
    n_ad_floor_drift = sum(
        1 for r in recs if any(str(c).startswith("OOD_DRIFT:") for c in (r.get("stage_b_rejects") or []))
    )
    n_raw_ood = sum(1 for r in recs if r.get("ad_overall") == "ood")
    n_lock_viol = sum(
        1
        for r in recs
        if any("LOCKED_POS" in c or "DISULFIDE_LOCK" in c or "SUBSTRUCTURE" in c for c in (r.get("stage_a_rejects") or []))
    )

    # Operator coverage diagnostic (verifies zero strategy starvation in StrongBase & Harness!)
    eval_strat_counts: Dict[str, int] = {}
    for r in recs:
        st = str(r.get("strategy") or "unknown")
        eval_strat_counts[st] = eval_strat_counts.get(st, 0) + 1

    # P0-2 FIX: Unambiguously separate safe recommendation, safe abstention, and unsafe false-positive!
    top1 = out["top1"]
    top1_abstained = top1 is None
    top1_feasible = bool(top1 is not None and top1["feasible"])
    top1_unsafe = bool(top1 is not None and not top1["feasible"])
    selected_feasible_delta_gmpo = round(max(0.0, float(top1["delta_gmpo"])), 3) if top1_feasible else 0.0
    pool_best_feasible_delta_gmpo = round(
        max((float(r["delta_gmpo"]) for r in recs if r["feasible"] and float(r["delta_gmpo"]) > 0.0), default=0.0),
        3,
    )
    lin_path = " -> ".join(top1["lineage_path"]) if (top1 and top1.get("lineage_path")) else (top1["mutation_label"] if top1 else None)

    return {
        "evals_used": int(out["evals_used"]),
        "n_candidates_scored": n_eval,
        "n_stage_a_rejected": n_sa_rej,
        "stage_a_pass_rate": round(n_sa_passed / max(n_sa_total, 1), 4),
        "stage_b_feasible_rate": round(n_feas / max(n_eval, 1), 4),
        "ad_floor_drift_rate": round(n_ad_floor_drift / max(n_eval, 1), 4),
        "raw_ood_status_rate": round(n_raw_ood / max(n_eval, 1), 4),
        "core_lock_violations_in_evaluated": n_lock_viol,
        "evaluated_strategy_counts": eval_strat_counts,
        "top1_abstained": top1_abstained,
        "top1_selected_is_feasible": top1_feasible,
        "top1_selected_is_unsafe": top1_unsafe,
        "top1_label": top1["mutation_label"] if top1 else None,
        "top1_lineage_path": lin_path,
        "top1_lineage_sequence": top1.get("lineage_sequence") if top1 else None,
        "top1_strategy": top1["strategy"] if top1 else None,
        "top1_sequence": top1["sequence"] if top1 else None,
        "top1_smiles": top1["smiles"] if top1 else None,
        "top1_is_chemical_leaf": bool(top1["is_chemical_leaf"]) if top1 else None,
        "top1_delta_gmpo": float(top1["delta_gmpo"]) if top1 else 0.0,
        "top1_hemolysis_cls": top1.get("hemolysis_cls") if top1 else None,
        "parent_hemolysis_cls": out.get("parent_hemolysis_cls"),
        "selected_feasible_delta_gmpo": selected_feasible_delta_gmpo,
        "pool_best_feasible_delta_gmpo": pool_best_feasible_delta_gmpo,
        "feasible_pareto_hv": round(float(out["feasible_pareto_hv"]), 6),
        "cross_track_audit": out.get("cross_track_audit") or [],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only-update-methods",
        type=str,
        default="",
        help="Comma-separated subset of methods to re-run and patch into existing v35_peptide_harness_benchmark_receipt.json",
    )
    args = parser.parse_args()
    only_update = {m.strip() for m in args.only_update_methods.split(",") if m.strip()}

    t_all = time.time()
    svc_procs, svc_url = start_local_v33_service(port=18955, esmc_port=18956)
    try:
        os.environ["V33_REMOTE_URL"] = svc_url
        os.environ["V33_STRICT_REMOTE"] = "1"
        os.environ["V33_CACHE_MAXSIZE"] = "8192"

        leads = build_benchmark_leads()
        ik14_set, seq_set = load_corpus_index(V33_ROOT / "data" / "compounds_v33.tsv")
        # Quick health check + manifest case17 names via one adapter
        probe_adapter = OmniPanelAdapter()
        assert probe_adapter.mode == "v33_remote" and probe_adapter.remote_healthy, f"Expected live v33_remote, got {probe_adapter.mode}"
        case17_names = {
            str(r.get("drug_name", "")).strip().lower()
            for r in ((probe_adapter.manifest.get("case17_scorecard") or {}).get("drugs") or [])
        }

        methods = [
            "Peptide-Harness-Full",
            "Exhaustive-1Step-GreedyCombine",
            "Harness-NoLipidation",
            "Harness-NoMemory",
            "Harness-NoGate",
            "Random-EqualBudget",
            "Unconstrained-Greedy",
        ]

        existing_by_lead: Dict[str, Dict[str, Any]] = {}
        if only_update:
            prev_path = SCRIPT_DIR / "v35_peptide_harness_benchmark_receipt.json"
            assert prev_path.exists(), f"--only-update-methods requires existing {prev_path}"
            prev_data = json.loads(prev_path.read_text(encoding="utf-8"))
            existing_by_lead = {r["lead_id"]: r["methods"] for r in prev_data.get("per_lead_results") or []}
            print(f"[bench] Incremental re-run of {sorted(only_update)} on live 4-FM V33 engine (preserving unaffected deterministic methods from v35.2 receipt)", flush=True)

        ckpt_lock = threading.Lock()
        completed_map: Dict[int, Dict[str, Any]] = {}

        def _run_one_lead(idx: int, lead: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
            # Each worker thread gets its own OmniPanelAdapter (zero OrderedDict race!) hitting the shared live V33 HTTP server
            adapter = OmniPanelAdapter()
            assert adapter.mode == "v33_remote" and adapter.remote_healthy
            t_lead = time.time()
            rng_lead = random.Random(BENCH_SEED + idx * 97)
            corpus_audit = audit_lead_corpus_status(lead, ik14_set, seq_set, case17_names)
            print(
                f"[{idx:02d}/{len(leads)}] START Lead={lead['lead_id']} ({lead['group']}, corpus={corpus_audit['corpus_status']}, in_corpus_any={corpus_audit['in_corpus_any']})",
                flush=True,
            )

            m_outs: Dict[str, Dict[str, Any]] = dict(existing_by_lead.get(lead["lead_id"]) or {}) if only_update else {}
            is_cyclic = lead["group"] == "cyclic_chemical_leaf_only"
            should_run = lambda m: (not only_update) or (m in only_update)

            # 1. Peptide-Harness-Full
            if should_run("Peptide-Harness-Full"):
                m_outs["Peptide-Harness-Full"] = summarize_method_run(
                    run_harness_variant(adapter, lead, use_negative_memory=True, use_ad_safety_gate=True, strategies=ALL_HARNESS_STRATEGIES, compute_cross_track=False)
                )
            # 2. Fair Strong Simple Baseline: Exhaustive-1-Step (all 5 families!) + Greedy-Combine
            if should_run("Exhaustive-1Step-GreedyCombine"):
                m_outs["Exhaustive-1Step-GreedyCombine"] = summarize_method_run(
                    run_exhaustive_1step_greedy_combine(adapter, lead)
                )
            # 3. Ablation: No-Lipidation
            if should_run("Harness-NoLipidation"):
                no_lip_strats = ("n_methylation", "cyclization") if is_cyclic else ("ala_scan", "point_mutation", "n_methylation", "cyclization")
                m_outs["Harness-NoLipidation"] = summarize_method_run(
                    run_harness_variant(adapter, lead, use_negative_memory=True, use_ad_safety_gate=True, strategies=no_lip_strats, compute_cross_track=False)
                )
            # 4. Ablation: w/o Negative Memory
            if should_run("Harness-NoMemory"):
                m_outs["Harness-NoMemory"] = summarize_method_run(
                    run_harness_variant(adapter, lead, use_negative_memory=False, use_ad_safety_gate=True, strategies=ALL_HARNESS_STRATEGIES, compute_cross_track=False)
                )
            # 5. Ablation: w/o Stage-B Gate
            if should_run("Harness-NoGate"):
                m_outs["Harness-NoGate"] = summarize_method_run(
                    run_harness_variant(adapter, lead, use_negative_memory=True, use_ad_safety_gate=False, strategies=ALL_HARNESS_STRATEGIES, compute_cross_track=False)
                )
            # 6. Random Equal-Budget (with symmetric free Stage A pre-filter)
            if should_run("Random-EqualBudget"):
                m_outs["Random-EqualBudget"] = summarize_method_run(run_random_baseline(adapter, lead, rng_lead))
            # 7. Unconstrained Greedy (Reward-Hacking Control)
            if should_run("Unconstrained-Greedy"):
                m_outs["Unconstrained-Greedy"] = summarize_method_run(run_unconstrained_greedy_baseline(adapter, lead))

            # Uniform post-hoc Dual-to-SMILES cross-track audit for EVERY updated method's Top-1 dual_modality recommendation!
            if not is_cyclic and lead.get("sequence"):
                methods_to_audit = [m for m in methods if should_run(m)]
                need_xt = any(
                    m_outs[m].get("top1_sequence") and m_outs[m].get("top1_smiles") and (not m_outs[m].get("top1_is_chemical_leaf"))
                    for m in methods_to_audit
                )
                if need_xt:
                    resolved_l = adapter.resolve_query(sequence=lead["sequence"])
                    p_can_l = _sanitize_and_canonicalize(resolved_l.mol) if resolved_l.mol is not None else None
                    if resolved_l.mol is not None and p_can_l is not None:
                        cl_l = resolve_core_locks(
                            parent_mol=resolved_l.mol,
                            parent_sequence=resolved_l.sequence,
                            locked_positions=lead.get("locked_positions") or [],
                            lock_disulfides=True,
                            functional_endpoint=lead.get("functional_endpoint"),
                        )
                        pbio_l = compute_biophysical_profile(cl_l.parent_sequence, resolved_l.mol)
                        inc_t2 = bool(lead.get("functional_endpoint"))
                        p_dual_bm, p_smi_bm = adapter.predict_batch(
                            [
                                BatchParsedRow(index=0, name="p_dual", detected_type="sequence", sequence=cl_l.parent_sequence, smiles=p_can_l, valid=True),
                                BatchParsedRow(index=1, name="p_smi", detected_type="smiles", sequence=None, smiles=p_can_l, valid=True),
                            ],
                            include_tier2=inc_t2,
                            conditions=AssayConditions(),
                        )
                        gk_audit = DeterministicGatekeeper(
                            core_lock=cl_l,
                            parent_biophysical=pbio_l,
                            parent_preds_by_track={
                                "dual_modality": {p.endpoint: p for p in p_dual_bm.predictions},
                                "smiles_only": {p.endpoint: p for p in p_smi_bm.predictions},
                            },
                            objective_x=lead["objective_x"],
                            objective_y=lead["objective_y"],
                            functional_endpoint=lead.get("functional_endpoint"),
                            weights=MpoWeights(),
                        )
                        p_smi_gmpo = gk_audit.check_stage_b(p_smi_bm.predictions, "smiles_only").gmpo_score
                        for mname in methods_to_audit:
                            ms = m_outs[mname]
                            if ms.get("top1_sequence") and ms.get("top1_smiles") and (not ms.get("top1_is_chemical_leaf")):
                                bm_s = adapter.predict_batch(
                                    [BatchParsedRow(index=0, name=f"xt_{mname}", detected_type="smiles", sequence=None, smiles=ms["top1_smiles"], valid=True)],
                                    include_tier2=inc_t2,
                                    conditions=AssayConditions(),
                                )[0]
                                if bm_s.status == "ok":
                                    sb_s = gk_audit.check_stage_b(bm_s.predictions, "smiles_only")
                                    d_smi = round(sb_s.gmpo_score - p_smi_gmpo, 2)
                                    d_dual = float(ms.get("top1_delta_gmpo") or 0.0)
                                    ret = round(100.0 * d_smi / max(d_dual, 1e-6), 1) if d_dual > 0.5 else None
                                    ms["cross_track_audit"] = [
                                        {
                                            "mutation_label": ms["top1_label"],
                                            "dual_delta_gmpo": d_dual,
                                            "smiles_only_delta_gmpo": d_smi,
                                            "smiles_only_stage_b_passed": sb_s.passed,
                                            "smiles_only_reject_codes": list(sb_s.reject_codes),
                                            "winners_curse_retention_pct": ret,
                                        }
                                    ]

            row_out = {
                "lead_id": lead["lead_id"],
                "group": lead["group"],
                "sequence": lead.get("sequence"),
                "objective_x": lead["objective_x"],
                "objective_y": lead["objective_y"],
                "functional_endpoint": lead.get("functional_endpoint"),
                "locked_positions": lead.get("locked_positions") or [],
                "corpus_audit": corpus_audit,
                "methods": m_outs,
            }
            print(
                f"[{idx:02d}/{len(leads)}] DONE {lead['lead_id']} in {time.time()-t_lead:.1f}s | "
                f"Full selΔgMPO={m_outs['Peptide-Harness-Full']['selected_feasible_delta_gmpo']:+.2f} "
                f"(HV={m_outs['Peptide-Harness-Full']['feasible_pareto_hv']:.3f}, path={m_outs['Peptide-Harness-Full']['top1_lineage_path']}) vs "
                f"StrongBase selΔgMPO={m_outs['Exhaustive-1Step-GreedyCombine']['selected_feasible_delta_gmpo']:+.2f} "
                f"(HV={m_outs['Exhaustive-1Step-GreedyCombine']['feasible_pareto_hv']:.3f}, path={m_outs['Exhaustive-1Step-GreedyCombine']['top1_lineage_path']}, "
                f"strats={m_outs['Exhaustive-1Step-GreedyCombine']['evaluated_strategy_counts']}) | "
                f"NoGate unsafe={m_outs['Harness-NoGate']['top1_selected_is_unsafe']} | Unconstr unsafe={m_outs['Unconstrained-Greedy']['top1_selected_is_unsafe']}",
                flush=True,
            )
            with ckpt_lock:
                completed_map[idx] = row_out
                ordered_done = [completed_map[k] for k in sorted(completed_map.keys())]
                (SCRIPT_DIR / "v35_benchmark_checkpoint.json").write_text(
                    json.dumps(
                        {"completed_leads": len(ordered_done), "total_leads": len(leads), "per_lead_results": ordered_done},
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            return idx, row_out

        print(f"[bench] Running {len(leads)} leads x {len(methods)} methods with BENCH_WORKERS={BENCH_WORKERS}...", flush=True)
        if BENCH_WORKERS <= 1:
            for idx, lead in enumerate(leads, start=1):
                _run_one_lead(idx, lead)
        else:
            with ThreadPoolExecutor(max_workers=BENCH_WORKERS) as pool:
                futs = [pool.submit(_run_one_lead, idx, lead) for idx, lead in enumerate(leads, start=1)]
                for fut in as_completed(futs):
                    fut.result()

        per_lead_results = [completed_map[k] for k in sorted(completed_map.keys())]

        # Collect all canonical sequences (parents + canonical Top-1s) for external HemoPI2 v1.3 cross-check
        all_canonical_seqs_for_hemopi2: List[str] = []
        for r in per_lead_results:
            if r.get("sequence"):
                all_canonical_seqs_for_hemopi2.append(r["sequence"])
            for ms in r["methods"].values():
                if ms.get("top1_sequence"):
                    all_canonical_seqs_for_hemopi2.append(ms["top1_sequence"])

        print("\n[bench] Running independent external HemoPI2 v1.3 cross-check on canonical sequences...", flush=True)
        hemopi2_scores = score_sequences_hemopi2(all_canonical_seqs_for_hemopi2)

        def _aggregate_subset(rows_sub: List[Dict[str, Any]]) -> Dict[str, Any]:
            agg: Dict[str, Any] = {}
            for mname in methods:
                ms_list = [r["methods"][mname] for r in rows_sub]
                n = len(ms_list)
                strat_counts: Dict[str, int] = {}
                hem_deltas: List[float] = []
                hem_non_worse = 0
                hem_verifiable = 0
                hem_unverified_noncanon = 0
                xtrack_retentions: List[float] = []
                xtrack_pass = 0
                xtrack_n = 0

                n_abstain = sum(1 for m in ms_list if m["top1_abstained"])
                n_safe_rec = sum(1 for m in ms_list if m["top1_selected_is_feasible"])
                n_unsafe_rec = sum(1 for m in ms_list if m["top1_selected_is_unsafe"])
                n_non_null = n_safe_rec + n_unsafe_rec

                for r, ms in zip(rows_sub, ms_list):
                    st = ms.get("top1_strategy")
                    if st:
                        strat_counts[st] = strat_counts.get(st, 0) + 1
                    p_seq = r.get("sequence")
                    c_seq = ms.get("top1_sequence")
                    if ms.get("top1_label") is not None:
                        if p_seq and c_seq and p_seq in hemopi2_scores and c_seq in hemopi2_scores:
                            dh = float(hemopi2_scores[c_seq]) - float(hemopi2_scores[p_seq])
                            hem_deltas.append(dh)
                            hem_verifiable += 1
                            if dh <= 0.05:
                                hem_non_worse += 1
                        else:
                            hem_unverified_noncanon += 1
                    for xt in ms.get("cross_track_audit") or []:
                        xtrack_n += 1
                        if xt.get("smiles_only_stage_b_passed"):
                            xtrack_pass += 1
                        if xt.get("winners_curse_retention_pct") is not None:
                            xtrack_retentions.append(float(xt["winners_curse_retention_pct"]))

                n_top1_any = hem_verifiable + hem_unverified_noncanon
                agg[mname] = {
                    "n_leads": n,
                    "constraint_compliance": {
                        "mean_stage_a_pass_rate": round(float(np.mean([m["stage_a_pass_rate"] for m in ms_list])), 4),
                        "mean_stage_b_feasible_rate": round(float(np.mean([m["stage_b_feasible_rate"] for m in ms_list])), 4),
                        "mean_ad_floor_drift_rate": round(float(np.mean([m["ad_floor_drift_rate"] for m in ms_list])), 4),
                        "mean_raw_ood_status_rate": round(float(np.mean([m["raw_ood_status_rate"] for m in ms_list])), 4),
                        "unsafe_top1_rate": round(n_unsafe_rec / max(n, 1), 4),
                        "unsafe_top1_count": n_unsafe_rec,
                        "abstention_rate": round(n_abstain / max(n, 1), 4),
                        "abstention_count": n_abstain,
                        "safe_recommendation_precision": round(n_safe_rec / max(n_non_null, 1), 4) if n_non_null > 0 else 1.0,
                        "safe_top1_over_all_leads_rate": round(n_safe_rec / max(n, 1), 4),
                        "total_core_lock_violations": int(sum(m["core_lock_violations_in_evaluated"] for m in ms_list)),
                    },
                    "oracle_surrogate_search": {
                        "mean_evals_used": round(float(np.mean([m["evals_used"] for m in ms_list])), 2),
                        "mean_selected_feasible_delta_gmpo": round(float(np.mean([m["selected_feasible_delta_gmpo"] for m in ms_list])), 3),
                        "median_selected_feasible_delta_gmpo": round(float(np.median([m["selected_feasible_delta_gmpo"] for m in ms_list])), 3),
                        "mean_pool_best_feasible_delta_gmpo": round(float(np.mean([m["pool_best_feasible_delta_gmpo"] for m in ms_list])), 3),
                        "median_pool_best_feasible_delta_gmpo": round(float(np.median([m["pool_best_feasible_delta_gmpo"] for m in ms_list])), 3),
                        "mean_feasible_pareto_hv": round(float(np.mean([m["feasible_pareto_hv"] for m in ms_list])), 5),
                        "median_feasible_pareto_hv": round(float(np.median([m["feasible_pareto_hv"] for m in ms_list])), 5),
                        "top1_strategy_attribution": strat_counts,
                    },
                    "independent_anti_circularity": {
                        "hemopi2_canonical_verified_count": hem_verifiable,
                        "hemopi2_unverified_noncanonical_count": hem_unverified_noncanon,
                        "hemopi2_coverage_pct": round(100.0 * hem_verifiable / max(n_top1_any, 1), 1),
                        "hemopi2_mean_delta_prob": round(float(np.mean(hem_deltas)), 4) if hem_deltas else None,
                        "hemopi2_non_worsening_rate": round(hem_non_worse / max(hem_verifiable, 1), 4) if hem_verifiable else None,
                        "cross_track_elites_audited": xtrack_n,
                        "cross_track_smiles_only_stage_b_pass_rate": round(xtrack_pass / max(xtrack_n, 1), 4) if xtrack_n else None,
                        "cross_track_median_gain_retention_pct": round(float(np.median(xtrack_retentions)), 1) if xtrack_retentions else None,
                        "caveat": (
                            f"External HemoPI2 v1.3 covers ONLY canonical 20-AA sequences ({hem_verifiable}/{n_top1_any}); "
                            f"dual->smiles cross-track audit covers ONLY dual-modality Top-1s (n={xtrack_n}). "
                            f"Chemical-leaf Top-1s have no external hemolysis or cross-track verification."
                        ),
                    },
                }

            # Paired comparisons: Peptide-Harness-Full vs each competitor (Holm-Bonferroni adjusted)
            full_sel = [r["methods"]["Peptide-Harness-Full"]["selected_feasible_delta_gmpo"] for r in rows_sub]
            full_pool = [r["methods"]["Peptide-Harness-Full"]["pool_best_feasible_delta_gmpo"] for r in rows_sub]
            full_hv = [r["methods"]["Peptide-Harness-Full"]["feasible_pareto_hv"] for r in rows_sub]
            vs_sel, vs_pool, vs_hv = {}, {}, {}
            p_sel_raw, p_pool_raw, p_hv_raw = {}, {}, {}
            for comp in methods[1:]:
                c_sel = [r["methods"][comp]["selected_feasible_delta_gmpo"] for r in rows_sub]
                c_pool = [r["methods"][comp]["pool_best_feasible_delta_gmpo"] for r in rows_sub]
                c_hv = [r["methods"][comp]["feasible_pareto_hv"] for r in rows_sub]
                ss = paired_stats(full_sel, c_sel)
                sp = paired_stats(full_pool, c_pool)
                sh = paired_stats(full_hv, c_hv)
                vs_sel[comp] = ss
                vs_pool[comp] = sp
                vs_hv[comp] = sh
                p_sel_raw[comp] = ss["wilcoxon_p_raw"]
                p_pool_raw[comp] = sp["wilcoxon_p_raw"]
                p_hv_raw[comp] = sh["wilcoxon_p_raw"]
            adj_s = holm_bonferroni(p_sel_raw)
            adj_p = holm_bonferroni(p_pool_raw)
            adj_h = holm_bonferroni(p_hv_raw)
            for comp in methods[1:]:
                vs_sel[comp]["wilcoxon_p_holm"] = adj_s[comp]
                vs_pool[comp]["wilcoxon_p_holm"] = adj_p[comp]
                vs_hv[comp]["wilcoxon_p_holm"] = adj_h[comp]

            return {
                "methods": agg,
                "paired_full_vs_baselines_selected_feasible_delta_gmpo": vs_sel,
                "paired_full_vs_baselines_pool_best_delta_gmpo": vs_pool,
                "paired_full_vs_baselines_feas_chv": vs_hv,
            }

        linear_rows = [r for r in per_lead_results if r["group"] != "cyclic_chemical_leaf_only"]
        cyclic_rows = [r for r in per_lead_results if r["group"] == "cyclic_chemical_leaf_only"]
        oos_rows = [r for r in per_lead_results if not r["corpus_audit"]["in_corpus_any"]]

        receipt = {
            "benchmark_version": "v35.2_peptide_harness_fair_strongbase_anti_circularity_benchmark",
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "oracle_mode": "v33_remote",
            "oracle_is_surrogate_heuristic": False,
            "equal_budget_max_total_evals": EQUAL_BUDGET_EVALS,
            "pareto_hv_reference_delta": {"ref_dx": REF_DX, "ref_dy": REF_DY},
            "runtime_total_s": round(time.time() - t_all, 1),
            "runtime_provenance": {
                "is_incremental_update": bool(only_update),
                "incremental_updated_methods": sorted(only_update) if only_update else None,
                "note": (
                    "Incremental re-run of affected methods on live 4-FM V33 engine; unaffected deterministic methods preserved from the full 7-method live 4-FM v35.2 run."
                    if only_update
                    else "Full 7-method live 4-FM V33 run."
                ),
            },
            "n_leads_total": len(per_lead_results),
            "n_leads_linear": len(linear_rows),
            "n_leads_cyclic_chemical_leaf_only": len(cyclic_rows),
            "corpus_overlap_summary": {
                "in_corpus_inchikey14_count": sum(1 for r in per_lead_results if r["corpus_audit"]["in_corpus_inchikey14"]),
                "in_corpus_sequence_count": sum(1 for r in per_lead_results if r["corpus_audit"]["in_corpus_sequence"]),
                "in_corpus_any_count": sum(1 for r in per_lead_results if r["corpus_audit"]["in_corpus_any"]),
                "out_of_sample_novel_count": len(oos_rows),
                "case17_clinical_drug_count": sum(1 for r in per_lead_results if r["corpus_audit"]["is_case17"]),
                "corpus_status_counts": {
                    st: sum(1 for r in per_lead_results if r["corpus_audit"]["corpus_status"] == st)
                    for st in sorted({r["corpus_audit"]["corpus_status"] for r in per_lead_results})
                },
            },
            "hemopi2_model_used": "official_pretrained_hemopi2_ml_clf_sav_v1.3_seq40_sklearn131",
            "hemopi2_scored_unique_canonical_sequences": len(hemopi2_scores),
            "aggregate_all_leads": _aggregate_subset(per_lead_results),
            "aggregate_linear_leads": _aggregate_subset(linear_rows),
            "aggregate_oos_novel_leads": _aggregate_subset(oos_rows),
            "aggregate_cyclic_leads": _aggregate_subset(cyclic_rows),
            "per_lead_results": per_lead_results,
        }

        out_path = SCRIPT_DIR / "v35_peptide_harness_benchmark_receipt.json"
        out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[bench] COMPLETE in {receipt['runtime_total_s']}s -> saved {out_path}", flush=True)
    finally:
        for p in reversed(svc_procs):
            print(f"[bench] Shutting down ephemeral worker pid={p.pid}...", flush=True)
            p.terminate()
            try:
                p.wait(timeout=15)
            except Exception:
                p.kill()


if __name__ == "__main__":
    main()
