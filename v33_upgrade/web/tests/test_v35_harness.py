"""Step 1a Adversarial & Invariant Unit Tests for Peptide-Harness (v35).

Covers every P0 & P1 defect identified in Step 0 and Step 1a Red-Team audits:
1. Full 47x2 Tier A/B/C snapshot + Tier B rejected as optimization objective + invalid track error
2. Stage B FAIL-CLOSED on missing functional endpoint, missing safety pillars, missing max_tanimoto, cross-track missing parent
3. Stage A anti-forgery (unlabeled SMILES, forged sequence != SMILES, parent-identical, Di-N-Me=2 & c[C-seq-C]=3 chem edit count)
4. Parent sequence-vs-SMILES mismatch (C-term amidation) -> refuses locked_positions & forces chemical_leaf_only
5. peptide_opt lock bypass fixes: c[C-seq-C] len<=30 termini lock AND lock_disulfides=False explicit Cys lock
6. NegativeSampleMemory tabu genuinely filters ala_scan & point_mutation
7. Consistent SMARTS charge estimator on LL-37 chemical leaf (no false rejection!)
8. Disulfide toxicity uniform formula + strict hemolysis/cyto bad-parent exemption (>=0.15 drop + <=0.70 cap)
9. Stage A chemistry reject codes (radical, disallowed element, MW, multi-lipidation, free thiols, charge window)
10. Adapter tier2->tier1 cache reuse filters out Tier-2 endpoints cleanly without mutating cached dict
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import pytest
from rdkit import Chem

from v33_upgrade.web.adapter import OmniPanelAdapter
from v33_upgrade.web.chem_utils import ResolvedInput, parse_smiles_safe, sequence_to_mol
from v33_upgrade.web.peptide_harness import (
    DeterministicGatekeeper,
    NegativeSampleMemory,
    _count_distinct_lipid_chains,
    classify_endpoint_tier,
    compute_biophysical_profile,
    default_direction_for_endpoint,
    generate_chemical_leaf_candidates,
    generate_sequence_candidates,
    resolve_core_locks,
    validate_optimization_objectives,
)
from v33_upgrade.web.peptide_opt import _gen_cyclization, _sanitize_and_canonicalize
from v33_upgrade.web.schemas import EndpointPrediction

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "bundle" / "v33_manifest.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _fake_pred(
    endpoint: str,
    value: float,
    raw_value: Optional[float] = None,
    ad_status: str = "in_domain",
    max_tanimoto: Optional[float] = 0.75,
    evidence_grade: str = "quantitative",
    kind: str = "regression",
) -> EndpointPrediction:
    return EndpointPrediction(
        endpoint=endpoint,
        display_en=endpoint,
        display_zh=endpoint,
        domain="pk",
        tier=1,
        kind=kind,
        value=value,
        raw_value=value if raw_value is None else raw_value,
        value_display=f"{value:.3g}",
        unit="u",
        risk_level="favorable",
        risk_label="OK",
        ad_status=ad_status,
        max_tanimoto=max_tanimoto,
        model_family="surrogate_mock",
        evidence_grade=evidence_grade,
    )


def _baseline_preds(
    *,
    pampa_log10: float = -6.0,
    hl_h: float = 5.0,
    hem_prob: float = 0.12,
    hc50_um: float = 120.0,
    tox_prob: float = 0.15,
    cyto_prob: float = 0.10,
    cc50_um: float = 150.0,
    mic_log10: float = 0.5,
    ad_status: str = "in_domain",
    tc: Optional[float] = 0.72,
) -> list[EndpointPrediction]:
    return [
        _fake_pred("PAMPA", 10.0 ** pampa_log10, raw_value=pampa_log10, ad_status=ad_status, max_tanimoto=tc),
        _fake_pred("Caco2", 10.0 ** (pampa_log10 - 0.3), raw_value=pampa_log10 - 0.3, ad_status=ad_status, max_tanimoto=tc),
        _fake_pred("half_life_reg", hl_h, raw_value=math.log10(max(hl_h, 1e-3)), ad_status=ad_status, max_tanimoto=tc),
        _fake_pred("hemolysis_cls", hem_prob, raw_value=hem_prob, ad_status=ad_status, max_tanimoto=tc, kind="classification"),
        _fake_pred("hemolysis_HC50", hc50_um, raw_value=math.log10(max(hc50_um, 1e-3)), ad_status=ad_status, max_tanimoto=tc),
        _fake_pred("overall_peptide_toxicity", tox_prob, raw_value=tox_prob, ad_status=ad_status, max_tanimoto=tc, kind="classification"),
        _fake_pred("cytotoxicity_cls", cyto_prob, raw_value=cyto_prob, ad_status=ad_status, max_tanimoto=tc, kind="classification"),
        _fake_pred("cytotoxicity_CC50", cc50_um, raw_value=math.log10(max(cc50_um, 1e-3)), ad_status=ad_status, max_tanimoto=tc),
        _fake_pred("MIC_E_coli", 10.0 ** mic_log10, raw_value=mic_log10, ad_status=ad_status, max_tanimoto=tc),
    ]


# 1. Tier Snapshot + Tier B Rejected as Objective + Invalid Track Error
def test_tier_snapshot_strict_validation(manifest):
    dual_counts = {"tier_a_primary": 0, "tier_b_soft_only": 0, "tier_c_forbidden": 0}
    smi_counts = {"tier_a_primary": 0, "tier_b_soft_only": 0, "tier_c_forbidden": 0}
    for ep in manifest["endpoints"]:
        dual_counts[classify_endpoint_tier(manifest, ep, track="dual_modality").tier] += 1
        smi_counts[classify_endpoint_tier(manifest, ep, track="smiles_only").tier] += 1
    assert dual_counts == {"tier_a_primary": 20, "tier_b_soft_only": 8, "tier_c_forbidden": 19}
    assert smi_counts == {"tier_a_primary": 20, "tier_b_soft_only": 9, "tier_c_forbidden": 18}

    with pytest.raises(ValueError):
        classify_endpoint_tier(manifest, "PAMPA", track="unknown_track")
    # Tier B endpoints MUST be rejected as optimization objectives
    for tier_b_ep in ("CL_per_kg", "RRCK", "efflux_ratio", "solubility_reg"):
        assert classify_endpoint_tier(manifest, tier_b_ep, track="dual_modality").tier == "tier_b_soft_only"
        with pytest.raises(ValueError):
            validate_optimization_objectives(manifest, tier_b_ep, "PAMPA", tracks=("dual_modality", "smiles_only"))
    # Ambiguous direction raises ValueError
    with pytest.raises(ValueError):
        default_direction_for_endpoint("LogD7.4")


# 2. P0-1: Stage B Fail-Closed on Missing Endpoints / Missing AD / Missing Parent Track
def test_stage_b_fail_closed_and_cross_track_guard():
    seq = "KWFKPLKGI"
    mol = sequence_to_mol(seq)
    lock = resolve_core_locks(mol, seq, functional_endpoint="MIC_E_coli")
    bio = compute_biophysical_profile(seq, mol)
    p_preds = {p.endpoint: p for p in _baseline_preds()}

    # Parent missing required endpoint raises ValueError upfront
    with pytest.raises(ValueError):
        DeterministicGatekeeper(lock, bio, {"dual_modality": {k: v for k, v in p_preds.items() if k != "MIC_E_coli"}}, "PAMPA", "half_life_reg", functional_endpoint="MIC_E_coli")

    gk = DeterministicGatekeeper(lock, bio, {"dual_modality": p_preds}, "PAMPA", "half_life_reg", functional_endpoint="MIC_E_coli")

    # Missing functional endpoint fails CLOSED
    no_mic = [p for p in _baseline_preds() if p.endpoint != "MIC_E_coli"]
    res_no_mic = gk.check_stage_b(no_mic, "dual_modality")
    assert res_no_mic.passed is False and "MISSING_REQUIRED_ENDPOINT:MIC_E_coli" in res_no_mic.reject_codes

    # Missing safety endpoints fail CLOSED
    no_safety = [p for p in _baseline_preds() if p.endpoint not in ("hemolysis_cls", "cytotoxicity_cls")]
    res_no_saf = gk.check_stage_b(no_safety, "dual_modality")
    assert res_no_saf.passed is False and any("MISSING_REQUIRED_ENDPOINT:hemolysis_cls" in c for c in res_no_saf.reject_codes)

    # Missing max_tanimoto fails CLOSED
    no_tc = _baseline_preds(tc=None)
    res_no_tc = gk.check_stage_b(no_tc, "dual_modality")
    assert res_no_tc.passed is False and any("missing_max_tanimoto" in c for c in res_no_tc.reject_codes)

    # Missing parent track raises ValueError (no silent cross-track fallback!)
    with pytest.raises(ValueError):
        gk.check_stage_b(_baseline_preds(), "smiles_only")


# 3. P0-2: Stage A Anti-Forgery, Parent-Identical Rejection & Multi-Site Chem Edit Count
def test_stage_a_anti_forgery_and_chem_edit_counting():
    seq = "KWFKPLKGI"
    mol = sequence_to_mol(seq)
    can = _sanitize_and_canonicalize(mol)
    lock = resolve_core_locks(mol, seq, locked_positions=[1, 2])
    bio = compute_biophysical_profile(seq, mol)
    p_preds = {p.endpoint: p for p in _baseline_preds()}
    gk = DeterministicGatekeeper(lock, bio, {"dual_modality": p_preds}, "PAMPA", "half_life_reg")

    # Parent-identical rejected even when seen_keys=None
    res_same = gk.check_stage_a(can, seq)
    assert res_same.passed is False and "IDENTICAL_TO_PARENT" in res_same.reject_codes

    # Forged sequence label paired with wild SMILES rejected!
    wild_smi = _sanitize_and_canonicalize(sequence_to_mol("KAAKAAKAI"))
    res_forged = gk.check_stage_a(wild_smi, seq)
    assert res_forged.passed is False and "SEQUENCE_SMILES_MISMATCH" in res_forged.reject_codes

    # Unlabeled sequence=None with is_chemical_leaf=False rejected!
    res_unlabeled = gk.check_stage_a(wild_smi, None, is_chemical_leaf=False)
    assert res_unlabeled.passed is False and "UNLABELED_NON_SEQUENCE_CANDIDATE" in res_unlabeled.reject_codes

    # Di-N-Me (2 amides) is pre-filtered by generator under max_total_chem_edits=1 AND rejected if fed directly to Stage A!
    from v33_upgrade.web.peptide_opt import _gen_n_methylation
    mem = NegativeSampleMemory()
    leaves = generate_chemical_leaf_candidates(mol, seq, can, lock, mem, strategies=("n_methylation",), limit=8)
    assert not any("Di-N-Me" in c.mutation_label for c in leaves), "Generator must pre-filter over-budget Di-N-Me"
    raw_nme, _ = _gen_n_methylation(mol, can, limit=8, locked_atom_indices=lock.locked_atom_indices)
    di_nme = [c for c in raw_nme if "Di-N-Me" in c.mutation_label]
    assert di_nme, "Expected raw Di-N-Me candidate to test Stage A chem edit counter"
    res_di = gk.check_stage_a(di_nme[0].smiles, None, is_chemical_leaf=True, lineage_sequence=seq, mutation_label=di_nme[0].mutation_label)
    assert res_di.passed is False and res_di.chem_edits_from_seed >= 2 and any("CHEM_EDIT_BUDGET_EXCEEDED" in c for c in res_di.reject_codes)


# 4. P0-3: Parent Sequence-vs-SMILES Mismatch (C-term Amidation) Never Silently Drops Locks/Amide
def test_parent_sequence_smiles_mismatch_safety():
    seq = "KWFKPLKGI"
    mol = sequence_to_mol(seq)
    # Create C-term amidated SMILES by replacing C-term -C(=O)O with -C(=O)N
    can = _sanitize_and_canonicalize(mol)
    amidated_smi = can.replace("C(=O)O", "C(=O)N", 1)
    amidated_mol = parse_smiles_safe(amidated_smi)
    assert amidated_mol is not None and _sanitize_and_canonicalize(amidated_mol) != can

    # Passing locked_positions on modified parent MUST raise ValueError (not silently zero out!)
    with pytest.raises(ValueError):
        resolve_core_locks(amidated_mol, seq, locked_positions=[2])

    # Without locked_positions, cleanly forces chemical_leaf_only with parent_sequence=None
    lock_mod = resolve_core_locks(amidated_mol, seq)
    assert lock_mod.harness_mode == "chemical_leaf_only"
    assert lock_mod.parent_sequence is None
    assert lock_mod.max_total_aa_edits == 0
    assert lock_mod.sequence_smiles_mismatch_reason is not None


# 5. P0-4: peptide_opt Lock Bypass Fixes (len<=30 c[C-seq-C] AND lock_disulfides=False explicit Cys lock)
def test_peptide_opt_lock_bypass_fixes():
    # (a) 16-AA (len<=30) with pos 1 & 16 locked -> c[C-seq-C] MUST NOT be generated
    seq16 = "HAEGTFTSDVSSYLEG"
    mol16 = sequence_to_mol(seq16)
    can16 = _sanitize_and_canonicalize(mol16)
    lock16 = resolve_core_locks(mol16, seq16, locked_positions=[1, 16])
    cyc16, _ = _gen_cyclization(mol16, seq16, can16, locked_positions=lock16.locked_positions, locked_atom_indices=lock16.locked_atom_indices)
    assert not any("c[C-seq-C]" in s.mutation_label for s in cyc16)
    assert not any("Head-to-Tail" in s.mutation_label for s in cyc16)

    # (b) Explicit Cys lock [2, 10] with lock_disulfides=False -> MUST NOT cyclize locked Cys 2 & 10!
    seq_cys = "ACDEFGHIKCLM"
    mol_cys = sequence_to_mol(seq_cys)
    can_cys = _sanitize_and_canonicalize(mol_cys)
    lock_cys = resolve_core_locks(mol_cys, seq_cys, locked_positions=[2, 10], lock_disulfides=False)
    cyc_cys, _ = _gen_cyclization(
        mol_cys, seq_cys, can_cys, locked_positions=lock_cys.locked_positions, locked_atom_indices=lock_cys.locked_atom_indices, lock_disulfides=False
    )
    assert not any("Cys-Cys Disulfide" in s.mutation_label for s in cyc_cys)


# 6. P1-1: NegativeSampleMemory Tabu Enforced Across ala_scan & point_mutation
def test_negative_memory_tabu_enforced_in_all_generators():
    seq = "KWFKPLKGI"
    mol = sequence_to_mol(seq)
    can = _sanitize_and_canonicalize(mol)
    lock = resolve_core_locks(mol, seq, locked_positions=[1])
    mem = NegativeSampleMemory()
    # Record K4A, W2A, F3A and W2S as tabu
    for lbl in ("W2A", "F3A", "K4A", "W2S", "F3S", "K4E"):
        mem.record_failure(1, lbl, "ala_scan", ["OOD_DRIFT"])
    cands = generate_sequence_candidates(seq, can, lock, mem, strategies=("ala_scan", "point_mutation"), limit=12)
    labels = {c.mutation_label for c in cands}
    for forbidden in ("W2A", "F3A", "K4A", "W2S", "F3S", "K4E"):
        assert forbidden not in labels


# 7. P1-7 & Stage A Chemistry Reject Codes + Dual Lipid Counter + Consistent LL-37 Charge
def test_stage_a_chemistry_rejects_and_ll37_charge_consistency():
    # LL-37 chemical leaf charge must NOT jump due to sequence vs SMARTS mismatch!
    ll37 = "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"
    mol_ll37 = sequence_to_mol(ll37)
    can_ll37 = _sanitize_and_canonicalize(mol_ll37)
    lock_ll37 = resolve_core_locks(mol_ll37, ll37, locked_positions=[1, 37])
    bio_ll37 = compute_biophysical_profile(ll37, mol_ll37)
    gk_ll37 = DeterministicGatekeeper(
        lock_ll37, bio_ll37, {"dual_modality": {p.endpoint: p for p in _baseline_preds()}}, "PAMPA", "half_life_reg"
    )
    # Single N-Me chemical leaf on LL-37 MUST pass Stage A charge window!
    nme_leaves = generate_chemical_leaf_candidates(
        mol_ll37, ll37, can_ll37, lock_ll37, NegativeSampleMemory(), strategies=("n_methylation",), limit=2
    )
    single_nme = [c for c in nme_leaves if "Di-N-Me" not in c.mutation_label][0]
    res_ll37_leaf = gk_ll37.check_stage_a(
        single_nme.smiles, None, is_chemical_leaf=True, lineage_sequence=ll37, mutation_label=single_nme.mutation_label
    )
    assert res_ll37_leaf.passed is True, res_ll37_leaf.reject_codes

    # Chemistry reject codes: radical, disallowed element (Pt), dual lipid chain
    res_rad = gk_ll37.check_stage_a("[CH2]CCCC", None, is_chemical_leaf=True)
    assert "RADICAL_ELECTRON" in res_rad.reject_codes
    res_elem = gk_ll37.check_stage_a("N[Pt](N)(Cl)Cl", None, is_chemical_leaf=True)
    assert "DISALLOWED_ELEMENT" in res_elem.reject_codes

    # Dual lipid chain detection
    dual_lipid_mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC(=O)NCC(=O)NC(CCCCNC(=O)CCCCCCCCCCCCCCC)C(=O)O")
    assert _count_distinct_lipid_chains(dual_lipid_mol) == 2
    res_multi_lip = gk_ll37.check_stage_a(Chem.MolToSmiles(dual_lipid_mol), None, is_chemical_leaf=True)
    assert "MULTI_LIPIDATION_EXCEEDED" in res_multi_lip.reject_codes


# 8. P1-5 & P1-6: Non-Noise Hemolysis Exemption & Uniform Disulfide gMPO
def test_non_noise_safety_exemption_and_ad_factor_rule():
    seq = "KWFKPLKGI"
    mol = sequence_to_mol(seq)
    lock = resolve_core_locks(mol, seq, locked_positions=[1])
    bio = compute_biophysical_profile(seq, mol)

    # Bad parent (hem=0.95, HC50=10): tiny drop to 0.89 (still >0.70, drop 0.06<0.15) MUST FAIL veto!
    bad_p = {p.endpoint: p for p in _baseline_preds(hem_prob=0.95, hc50_um=10.0)}
    gk_bad = DeterministicGatekeeper(lock, bio, {"dual_modality": bad_p}, "PAMPA", "half_life_reg")
    res_noise = gk_bad.check_stage_b(_baseline_preds(hem_prob=0.89, hc50_um=2.0), "dual_modality")
    assert res_noise.passed is False and any("HEMOLYSIS_PILLAR_VETO" in c for c in res_noise.reject_codes)

    # Good parent (in-domain): candidate stays >=0.40 -> ad_factor == 1.0 (not 0.72)
    good_p = {p.endpoint: p for p in _baseline_preds(ad_status="in_domain", tc=0.75)}
    gk_good = DeterministicGatekeeper(lock, bio, {"dual_modality": good_p}, "PAMPA", "half_life_reg")
    res_id = gk_good.check_stage_b(_baseline_preds(ad_status="borderline", tc=0.45), "dual_modality")
    assert res_id.passed is True and res_id.ad_factor == 1.0


# 9. P1-9: Adapter Tier-2 Cache Reuse Filters Out Tier-2 Endpoints & Protects Cache Dict
def test_adapter_batch_exact_tuple_dedup_and_tier1_filtered_cache_reuse(monkeypatch):
    adapter = OmniPanelAdapter()
    adapter.remote_url = "http://127.0.0.1:8902"
    adapter.remote_healthy = True
    adapter.mode = "v33_remote"

    seq = "KWFKPLKGI"
    can = _sanitize_and_canonicalize(sequence_to_mol(seq))
    # Pre-seed tier2=True cache with BOTH Tier-1 (PAMPA) and Tier-2 (MIC_E_coli) endpoints
    ckey_t2 = adapter._cache_key(can, seq, True, {"pH": "7.4"})
    orig_cached = {"status": "ok", "endpoints": {"PAMPA": {"value": -6.0}, "MIC_E_coli": {"value": 0.5}}}
    adapter.cache.set(ckey_t2, orig_cached)

    captured_payloads: list[dict] = []

    class _DummyResp:
        def __init__(self, n: int):
            self._body = json.dumps({"results": [{"status": "ok", "endpoints": {"PAMPA": {"value": -5.8}}} for _ in range(n)]}).encode("utf-8")
        def read(self):
            return self._body
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    def _fake_urlopen(req, timeout=240.0):
        body = json.loads(req.data.decode("utf-8"))
        captured_payloads.append(body)
        return _DummyResp(len(body["items"]))

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    items = [
        ResolvedInput(query_name="dual", resolved_type="sequence", sequence=seq, smiles=can, canonical_smiles=can, mol=None),
        ResolvedInput(query_name="smi1", resolved_type="smiles", sequence=None, smiles=can, canonical_smiles=can, mol=None),
        ResolvedInput(query_name="smi2", resolved_type="smiles", sequence=None, smiles=can, canonical_smiles=can, mol=None),
    ]
    res = adapter._call_v33_remote_batch(items, include_tier2=False, v33_cond={"pH": "7.4"})
    # Item 0 reused tier2 cache BUT filtered out Tier-2 endpoint MIC_E_coli!
    assert res[0][1] is True
    assert "PAMPA" in res[0][0]["endpoints"]
    assert "MIC_E_coli" not in res[0][0]["endpoints"]
    # Original cached tier2 dict MUST NOT have been mutated!
    assert "MIC_E_coli" in adapter.cache.get(ckey_t2)["endpoints"]
    # Exact tuple dedup: only 1 uncached smiles_only item sent to remote /batch
    assert len(captured_payloads) == 1 and len(captured_payloads[0]["items"]) == 1
    assert "sequence" not in captured_payloads[0]["items"][0]
    assert res[1][1] is False and res[2][1] is True



# ---------------------------------------------------------------------------
# Step 1b End-to-End Multi-Round Evolution, Critic Recycling & Cancellation Tests
# ---------------------------------------------------------------------------

def test_step1b_multi_round_evolution_recycling_cross_track_and_cancel():
    from v33_upgrade.web.peptide_harness import HarnessConfig, run_peptide_harness
    from v33_upgrade.web.schemas import BatchMoleculeResult
    import threading

    adapter = OmniPanelAdapter()
    # Deterministic mock evaluator: rewards Ala/Ser/Gln substitutions on PAMPA & HL, penalizes pos 2->E as OOD
    call_log: list[int] = []

    def _mock_eval(rows, include_tier2, cond):
        call_log.append(len(rows))
        out = []
        for r in rows:
            seq = r.sequence or ""
            # Track shift simulation: dual vs smiles_only differ slightly
            is_dual = r.sequence is not None
            n_ala = seq.count("A") if is_dual else 1
            pampa_log = -6.2 + 0.25 * min(n_ala, 3) + (0.05 if is_dual else 0.0)
            hl_h = 4.0 + 3.0 * min(n_ala, 3)
            # Trigger OOD drift on any sequence containing 'E' at pos 1 (0-idx)
            is_ood = len(seq) >= 2 and seq[1] == "E"
            preds = _baseline_preds(
                pampa_log10=pampa_log,
                hl_h=hl_h,
                hem_prob=0.14,
                tox_prob=0.16,
                cyto_prob=0.10,
                mic_log10=0.45,
                ad_status="ood" if is_ood else "in_domain",
                tc=0.25 if is_ood else 0.72,
            )
            out.append(
                BatchMoleculeResult(
                    index=r.index,
                    name=r.name,
                    status="ok",
                    resolved_type=r.detected_type,
                    sequence=r.sequence,
                    smiles=r.smiles,
                    canonical_smiles=r.smiles,
                    predictions=preds,
                )
            )
        return out

    cfg = HarnessConfig(
        objective_x="PAMPA",
        objective_y="half_life_reg",
        functional_endpoint="MIC_E_coli",
        locked_positions=(1,),
        rounds=2,
        population_size=4,
        critic_max_perturbations=4,
        max_total_evals=24,
        chunk_size=4,
        compute_elite_cross_track_audit=True,
    )
    res = run_peptide_harness(adapter, sequence="KWFKPLKGI", config=cfg, eval_batch_fn=_mock_eval)
    assert res["status"] == "completed"
    assert res["harness_mode"] == "functional_guided"
    assert res["activity_unmodeled"] is False
    assert res["oracle_is_surrogate_heuristic"] is True  # offline adapter honestly flagged!
    assert len(res["rounds"]) == 2
    assert res["rounds"][0]["recycled_critic_evals"] >= 1  # zero-cost Ala-scan recycling verified!
    assert res["metrics"]["evals_used"] <= cfg.max_total_evals
    assert res["metrics"]["best_delta_gmpo"] > 0.0
    assert len(res["final_recommendations"]) >= 1
    assert len(res["cross_track_epistemic_audit"]) >= 1
    # Verify locked position 1 ('K') was NEVER mutated in any candidate
    for c in res["candidates"]:
        if c["sequence"] is not None:
            assert c["sequence"][0] == "K"
            assert c["aa_edits_from_seed"] <= 3

    # Test strict remote oracle guard raises EngineUnavailableError when adapter is offline
    from v33_upgrade.web.adapter import EngineUnavailableError
    cfg_strict = HarnessConfig(objective_x="PAMPA", objective_y="half_life_reg", require_remote_oracle=True)
    with pytest.raises(EngineUnavailableError):
        run_peptide_harness(adapter, sequence="KWFKPLKGI", config=cfg_strict, eval_batch_fn=_mock_eval)

    # Test cooperative cancellation mid-run
    ev = threading.Event()
    ev.set()
    res_cancel = run_peptide_harness(adapter, sequence="KWFKPLKGI", config=cfg, cancel_event=ev, eval_batch_fn=_mock_eval)
    assert res_cancel["status"] == "cancelled"
    assert res_cancel["metrics"]["cancelled"] is True



# ---------------------------------------------------------------------------
# Step 1b Red-Team R1b Adversarial Regression Tests (P0-1..P0-3 & P1-1..P1-8)
# ---------------------------------------------------------------------------

def test_r1b_p0_1_delta_space_pareto_blocks_track_intercept_cheat():
    """P0-1: A smiles_only leaf worse than its own parent (-3.87 delta_gmpo) MUST NOT outrank positive dual candidates!"""
    from v33_upgrade.web.peptide_harness import HarnessConfig, run_peptide_harness
    from v33_upgrade.web.schemas import BatchMoleculeResult

    adapter = OmniPanelAdapter()

    def _offset_mock(rows, include_tier2, cond):
        out = []
        for r in rows:
            is_dual = r.sequence is not None
            if is_dual:
                n_ala = (r.sequence or "").count("A")
                # Dual parent (-5.65, 5h), dual Ala mutants improve (+0.20, +2h)
                preds = _baseline_preds(pampa_log10=-5.65 + 0.20 * n_ala, hl_h=5.0 + 2.0 * n_ala)
            else:
                # smiles_only track has huge +0.60 intercept bias, BUT leaves are WORSE than smiles_only parent!
                is_parent_smi = r.name.endswith("_smiles") or r.name.startswith("xtrack_")
                preds = _baseline_preds(
                    pampa_log10=-5.05 if is_parent_smi else -5.15,
                    hl_h=12.0 if is_parent_smi else 6.0,
                )
            out.append(BatchMoleculeResult(index=r.index, name=r.name, status="ok", resolved_type=r.detected_type, sequence=r.sequence, smiles=r.smiles, canonical_smiles=r.smiles, predictions=preds))
        return out

    cfg = HarnessConfig(
        objective_x="PAMPA",
        objective_y="half_life_reg",
        locked_positions=(1,),
        rounds=2,
        population_size=4,
        max_total_evals=24,
        compute_elite_cross_track_audit=True,
    )
    res = run_peptide_harness(adapter, sequence="KWFKPLKGI", config=cfg, eval_batch_fn=_offset_mock)
    # Every recommended candidate MUST have delta_gmpo > 0 (worse smiles_only leaves excluded!)
    assert len(res["final_recommendations"]) >= 1
    for rec in res["final_recommendations"]:
        assert rec["delta_gmpo"] > 0.0
    # Cross-track epistemic audit MUST run via reserved budget AND report offset-free |d_dual - d_smiles|!
    assert len(res["cross_track_epistemic_audit"]) >= 1
    assert "smiles_only_stage_b_passed" in res["cross_track_epistemic_audit"][0]
    assert "epistemic_disagreement_dx" in res["cross_track_epistemic_audit"][0]
    # Round-robin leaf check: lipidation_c16_c18 MUST actually be proposed alongside n_methylation!
    strategies_seen = {c["strategy"] for c in res["candidates"]}
    assert "lipidation_c16_c18" in strategies_seen


def test_r1b_p0_2_critic_cancel_no_crash_and_p1_8_partial_on_busy():
    """P0-2 & P1-8: Cancelling during Critic callback returns clean cancelled (never IndexError); EngineBusy returns partial."""
    from v33_upgrade.web.adapter import EngineBusyError
    from v33_upgrade.web.peptide_harness import HarnessConfig, run_peptide_harness
    from v33_upgrade.web.schemas import BatchMoleculeResult
    import threading

    adapter = OmniPanelAdapter()
    ev = threading.Event()

    def _ok_eval(rows, include_tier2, cond):
        return [
            BatchMoleculeResult(index=r.index, name=r.name, status="ok", resolved_type=r.detected_type, sequence=r.sequence, smiles=r.smiles, canonical_smiles=r.smiles, predictions=_baseline_preds())
            for r in rows
        ]

    def _cancel_on_critic(snap):
        if snap.get("phase") == "critic_attribution":
            ev.set()

    cfg = HarnessConfig(objective_x="PAMPA", objective_y="half_life_reg", rounds=2, population_size=4, max_total_evals=20)
    res = run_peptide_harness(adapter, sequence="KWFKPLKGI", config=cfg, cancel_event=ev, progress_callback=_cancel_on_critic, eval_batch_fn=_ok_eval)
    assert res["status"] == "cancelled"

    # EngineBusyError on Round 1 scoring -> retries once then returns clean status="partial" (not crash!)
    calls = {"n": 0}
    def _busy_after_parent(rows, include_tier2, cond):
        calls["n"] += 1
        if calls["n"] == 1:
            return _ok_eval(rows, include_tier2, cond)
        raise EngineBusyError("simulated busy")

    res_partial = run_peptide_harness(adapter, sequence="KWFKPLKGI", config=cfg, eval_batch_fn=_busy_after_parent)
    assert res_partial["status"] == "partial" and res_partial["partial_error"] is not None


def test_r1b_p0_3_forged_chemical_leaf_blocked_and_p1_7_hemolysis_spike_veto():
    """P0-3: Mutating locked K1->R1/W2->Y2/F3->Y3 and disguising as chemical leaf MUST FAIL substructure check!
    P1-7: Spiking hemolysis_cls 0.12->0.90 while HC50=120 MUST FAIL HEMOLYSIS_CLS_SPIKE!"""
    seq = "KWFKPLKGIAKWLRSV"
    mol = sequence_to_mol(seq)
    lock = resolve_core_locks(mol, seq, locked_positions=[1, 2, 3])
    bio = compute_biophysical_profile(seq, mol)
    p_preds = {p.endpoint: p for p in _baseline_preds(hem_prob=0.12, hc50_um=120.0)}
    gk = DeterministicGatekeeper(lock, bio, {"dual_modality": p_preds}, "PAMPA", "half_life_reg")

    # Forged chemical leaf with K1->R1, W2->Y2, F3->Y3 (MW increases -> passes naive MW check, caught by substructure!)
    forged_smi = _sanitize_and_canonicalize(sequence_to_mol("RYYKPLKGIAKWLRSV"))
    res_forged = gk.check_stage_a(forged_smi, None, is_chemical_leaf=True, lineage_sequence=seq)
    assert res_forged.passed is False and "CHEMICAL_LEAF_SUBSTRUCTURE_MISMATCH" in res_forged.reject_codes

    # Genuine single N-Me leaf MUST STILL PASS substructure check!
    leaves = generate_chemical_leaf_candidates(
        mol, seq, _sanitize_and_canonicalize(mol), lock, NegativeSampleMemory(), strategies=("n_methylation", "lipidation_c16_c18"), limit=4
    )
    assert any(c.strategy == "lipidation_c16_c18" for c in leaves)
    assert any(c.strategy == "n_methylation" for c in leaves)
    for lf in leaves:
        res_ok = gk.check_stage_a(lf.smiles, None, is_chemical_leaf=True, lineage_sequence=seq, mutation_label=lf.mutation_label)
        assert res_ok.passed is True, (lf.mutation_label, res_ok.reject_codes)

    # P1-7: hemolysis_cls spikes 0.12 -> 0.90 while HC50 stays 120 -> MUST FAIL HEMOLYSIS_CLS_SPIKE!
    res_spike = gk.check_stage_b(_baseline_preds(hem_prob=0.90, hc50_um=120.0), "dual_modality")
    assert res_spike.passed is False and any("HEMOLYSIS_CLS_SPIKE" in c for c in res_spike.reject_codes)


def test_r1b_p1_3_no_duplicate_memory_pollution_and_p1_6_canonical_atom_alignment():
    """P1-3: DUPLICATE_CANDIDATE never enters negative_memory; P1-6: Octreotide disulfide locked atoms align on canonical SMILES!"""
    mem = NegativeSampleMemory()
    mem.record_failure(1, "K2A", "ala_scan", ["DUPLICATE_CANDIDATE"])
    mem.record_failure(1, "K2A", "ala_scan", ["IDENTICAL_TO_PARENT"])
    assert len(mem.entries) == 0 and not mem.is_tabu_seq_edit(2, "A")

    # P1-6: Disulfide cyclic peptide locked_atom_indices MUST point to actual sulfur atoms (atomic num 16) on canonical Mol!
    cyc_specs, _ = _gen_cyclization(sequence_to_mol("CAKLVFFKC"), "CAKLVFFKC", _sanitize_and_canonicalize(sequence_to_mol("CAKLVFFKC")))
    ss_smi = next(s.smiles for s in cyc_specs if "Disulfide" in s.mutation_label)
    ss_mol = parse_smiles_safe(ss_smi)
    lock = resolve_core_locks(ss_mol, None, lock_disulfides=True)
    canon_mol = Chem.MolFromSmiles(lock.parent_canonical_smiles)
    assert len(lock.locked_atom_indices) == 2
    for aidx in lock.locked_atom_indices:
        assert canon_mol.GetAtomWithIdx(aidx).GetAtomicNum() == 16


# ---------------------------------------------------------------------------
# Step 2: Async HarnessJobManager, FastAPI Routes, Security & Interactive Yield Tests
# ---------------------------------------------------------------------------

def _fast_mock_eval_for_jobs(rows, include_tier2=False, conditions=None):
    from v33_upgrade.web.schemas import BatchMoleculeResult

    out = []
    for r in rows:
        is_dual = r.sequence is not None
        n_ala = (r.sequence or "").count("A") if is_dual else 1
        preds = _baseline_preds(
            pampa_log10=-6.1 + 0.20 * min(n_ala, 3),
            hl_h=4.0 + 2.5 * min(n_ala, 3),
            hem_prob=0.12,
            tox_prob=0.14,
            cyto_prob=0.10,
            mic_log10=0.45,
        )
        out.append(
            BatchMoleculeResult(
                index=r.index,
                name=r.name,
                status="ok",
                resolved_type=r.detected_type,
                sequence=r.sequence,
                smiles=r.smiles,
                canonical_smiles=r.smiles,
                predictions=preds,
            )
        )
    return out


def test_step2_async_job_lifecycle_ownership_ssrf_and_sync_cap(monkeypatch):
    import time
    from fastapi.testclient import TestClient
    from v33_upgrade.web import app as web_app
    from v33_upgrade.web.harness_jobs import HarnessJobManager

    mgr = HarnessJobManager(adapter=web_app.adapter, eval_batch_fn=_fast_mock_eval_for_jobs)
    monkeypatch.setattr(web_app, "harness_job_manager", mgr)
    monkeypatch.setattr(web_app.adapter, "predict_batch", _fast_mock_eval_for_jobs)

    client = TestClient(web_app.app)

    # 1. /health reports harness_llm_enabled=False by default
    r_health = client.get("/api/v1/health")
    assert r_health.status_code == 200
    assert r_health.json()["harness_llm_enabled"] is False

    # 2. /tiers dynamically classifies dual_modality & smiles_only
    r_tiers = client.get("/api/v1/peptide-harness/tiers")
    assert r_tiers.status_code == 200
    tdata = r_tiers.json()
    assert tdata["tracks"]["dual_modality"]["PAMPA"]["short_tier"] == "A"
    assert tdata["tracks"]["dual_modality"]["Vd_absolute"]["short_tier"] == "C"

    # 3. Missing or invalid X-Client-Id strictly rejected with 400 (no anonymous fallback!)
    r_no_cid = client.post("/api/v1/peptide-harness/jobs", json={"sequence": "KWFKPLKGI"})
    assert r_no_cid.status_code == 400
    r_bad_cid = client.post("/api/v1/peptide-harness/jobs", json={"sequence": "KWFKPLKGI"}, headers={"X-Client-Id": "bad"})
    assert r_bad_cid.status_code == 400

    # 4. Zero SSRF: extra unknown fields like llm_url / webhook_url rejected with 422!
    r_ssrf = client.post(
        "/api/v1/peptide-harness/jobs",
        json={"sequence": "KWFKPLKGI", "llm_url": "http://169.254.169.254/latest/meta-data/"},
        headers={"X-Client-Id": "client-alpha-01"},
    )
    assert r_ssrf.status_code == 422

    # 5. Preflight sync validation rejects Tier-C objective & identical X==Y BEFORE returning 202!
    r_tier_c = client.post(
        "/api/v1/peptide-harness/jobs",
        json={
            "sequence": "KWFKPLKGI",
            "objective_x": {"endpoint": "Vd_absolute", "direction": "max"},
            "objective_y": {"endpoint": "half_life_reg", "direction": "max"},
        },
        headers={"X-Client-Id": "client-alpha-01"},
    )
    assert r_tier_c.status_code == 400
    r_same_xy = client.post(
        "/api/v1/peptide-harness/jobs",
        json={
            "sequence": "KWFKPLKGI",
            "objective_x": {"endpoint": "PAMPA", "direction": "max"},
            "objective_y": {"endpoint": "PAMPA", "direction": "max"},
        },
        headers={"X-Client-Id": "client-alpha-01"},
    )
    assert r_same_xy.status_code == 400

    # 6. Happy path: 202 Accepted -> poll GET -> completed + cross-client IDOR blocked (404)!
    r_sub = client.post(
        "/api/v1/peptide-harness/jobs",
        json={
            "sequence": "KWFKPLKGI",
            "locked_positions": [1],
            "rounds": 2,
            "population_size": 4,
            "max_total_evals": 16,
        },
        headers={"X-Client-Id": "client-alpha-01"},
    )
    assert r_sub.status_code == 202
    job_id = r_sub.json()["job_id"]
    assert job_id.startswith("H-")

    # Cross-client GET and DELETE must return 404!
    assert client.get(f"/api/v1/peptide-harness/jobs/{job_id}", headers={"X-Client-Id": "client-other-99"}).status_code == 404
    assert client.delete(f"/api/v1/peptide-harness/jobs/{job_id}", headers={"X-Client-Id": "client-other-99"}).status_code == 404

    # Poll until completed
    final_detail = None
    for _ in range(80):
        r_get = client.get(f"/api/v1/peptide-harness/jobs/{job_id}", headers={"X-Client-Id": "client-alpha-01"})
        assert r_get.status_code == 200
        final_detail = r_get.json()
        if final_detail["status"] in ("completed", "partial", "failed", "cancelled"):
            break
        time.sleep(0.05)
    assert final_detail is not None and final_detail["status"] == "completed"
    assert final_detail["result"]["metrics"]["evals_used"] <= 16
    assert len(final_detail["result"]["final_recommendations"]) >= 1

    # List endpoint shows only caller's jobs
    r_list_a = client.get("/api/v1/peptide-harness/jobs", headers={"X-Client-Id": "client-alpha-01"})
    r_list_b = client.get("/api/v1/peptide-harness/jobs", headers={"X-Client-Id": "client-other-99"})
    assert r_list_a.json()["total"] == 1 and r_list_b.json()["total"] == 0

    # 7. Strictly capped sync route: rejects rounds=2 or max_total_evals=20 (422), accepts rounds=1 max_total_evals=6 (200)
    r_sync_bad = client.post(
        "/api/v1/peptide-harness",
        json={"sequence": "KWFKPLKGI", "rounds": 2, "max_total_evals": 6},
        headers={"X-Client-Id": "client-alpha-01"},
    )
    assert r_sync_bad.status_code == 422
    r_sync_ok = client.post(
        "/api/v1/peptide-harness",
        json={"sequence": "KWFKPLKGI", "rounds": 1, "population_size": 2, "critic_max_perturbations": 2, "max_total_evals": 6},
        headers={"X-Client-Id": "client-alpha-01"},
    )
    assert r_sync_ok.status_code == 200
    assert r_sync_ok.json()["status"] == "completed"
    assert r_sync_ok.json()["metrics"]["evals_used"] <= 6


def test_step2_per_ip_abuse_guard_cancel_and_interactive_yield_gate():
    import threading
    import time
    from v33_upgrade.web.harness_jobs import (
        HarnessClientBusyError,
        HarnessJobManager,
        HarnessQueueFullError,
    )
    from v33_upgrade.web.peptide_harness import HarnessConfig

    adapter = OmniPanelAdapter()
    gate = threading.Event()

    def _slow_eval(rows, include_tier2, cond):
        gate.wait(timeout=3.0)
        return _fast_mock_eval_for_jobs(rows, include_tier2, cond)

    mgr = HarnessJobManager(adapter=adapter, max_queue_depth=2, max_retained=4, ttl_seconds=60.0, eval_batch_fn=_slow_eval)
    cfg = HarnessConfig(rounds=2, population_size=4, max_total_evals=14)

    # Submit job 1 from client-1 / IP 10.0.0.1
    j1 = mgr.submit_job(client_id="client-one-01", client_ip="10.0.0.1", query=None, sequence="KWFKPLKGI", smiles=None, config=cfg)
    assert j1["status"] in ("queued", "running")

    # Forging a NEW X-Client-Id from the SAME IP MUST be blocked (per-IP cap = 1)!
    with pytest.raises(HarnessClientBusyError):
        mgr.submit_job(client_id="client-forged-02", client_ip="10.0.0.1", query=None, sequence="KWFKPLKGI", smiles=None, config=cfg)

    # Same client_id from a DIFFERENT IP MUST ALSO be blocked (per-client cap = 1)!
    with pytest.raises(HarnessClientBusyError):
        mgr.submit_job(client_id="client-one-01", client_ip="10.0.0.2", query=None, sequence="KWFKPLKGI", smiles=None, config=cfg)

    # Different client + different IP queues cleanly (reaches max_queue_depth=2)
    j2 = mgr.submit_job(client_id="client-two-02", client_ip="10.0.0.2", query=None, sequence="KWFKPLKGI", smiles=None, config=cfg)
    assert j2["status"] == "queued"

    # 3rd concurrent caller hits global queue cap!
    with pytest.raises(HarnessQueueFullError):
        mgr.submit_job(client_id="client-three-03", client_ip="10.0.0.3", query=None, sequence="KWFKPLKGI", smiles=None, config=cfg)

    # Cancel queued j2 -> immediately transitions to cancelled!
    c2 = mgr.cancel_job(j2["job_id"], "client-two-02")
    assert c2 is not None and c2["status"] == "cancelled" and c2["cancel_requested"] is True

    # Cancel running j1, then release gate -> finishes cleanly as cancelled!
    c1 = mgr.cancel_job(j1["job_id"], "client-one-01")
    assert c1 is not None and c1["cancel_requested"] is True
    gate.set()
    for _ in range(60):
        d1 = mgr.get_job(j1["job_id"], "client-one-01")
        if d1 and d1["status"] == "cancelled":
            break
        time.sleep(0.05)
    assert d1 is not None and d1["status"] == "cancelled"

    # Interactive yield gate test: when an interactive /predict is active, wait_for_interactive_idle blocks until released!
    with adapter._predict_idle_cond:
        adapter._predict_active_count = 1
    res_box = {"returned": False, "t_done": 0.0}

    def _waiter():
        ok = adapter.wait_for_interactive_idle(max_wait_s=2.0)
        res_box["returned"] = ok
        res_box["t_done"] = time.monotonic()

    t_start = time.monotonic()
    th = threading.Thread(target=_waiter, daemon=True)
    th.start()
    time.sleep(0.15)
    assert res_box["returned"] is False  # Still yielding while interactive /predict is active!
    with adapter._predict_idle_cond:
        adapter._predict_active_count = 0
        adapter._predict_idle_cond.notify_all()
    th.join(timeout=2.0)
    assert res_box["returned"] is True
    assert (res_box["t_done"] - t_start) >= 0.12

    # Interactive yield gate MUST wake immediately when cancel_event is set!
    with adapter._predict_idle_cond:
        adapter._predict_active_count = 1
    ev = threading.Event()
    ev.set()
    assert adapter.wait_for_interactive_idle(cancel_event=ev, max_wait_s=5.0) is False
    with adapter._predict_idle_cond:
        adapter._predict_active_count = 0
        adapter._predict_idle_cond.notify_all()


# ---------------------------------------------------------------------------
# Step 2 Red-Team R2 Adversarial Regression Tests (P0-C, R1b P0-3/P1-3, P1-1..P1-6)
# ---------------------------------------------------------------------------

def test_r2_regression_suite_all_p0_p1_fixes():
    import math
    from v33_upgrade.web.chem_utils import parse_smiles_safe
    from v33_upgrade.web.harness_jobs import HarnessJobManager
    from v33_upgrade.web.peptide_harness import HarnessConfig, run_peptide_harness

    # 1. P0-C: Giant chain SMILES ("C"*3000) rejected cleanly before RDKit MolToSmiles stack overflow!
    assert parse_smiles_safe("C" * 3000) is None
    assert parse_smiles_safe("C" * 20) is not None

    # 2. R1b P0-3 FULL: Supergraph AA mutations (G->A, G->W, A->L, A->F, S->T) disguised as chemical leaves MUST FAIL!
    seq = "KWFKPLKGIAKSWRSV"
    mol = sequence_to_mol(seq)
    lock = resolve_core_locks(mol, seq, locked_positions=[8, 10, 12])
    bio = compute_biophysical_profile(seq, mol)
    p_preds = {p.endpoint: p for p in _baseline_preds()}
    gk = DeterministicGatekeeper(lock, bio, {"dual_modality": p_preds}, "PAMPA", "half_life_reg")

    for bad_seq in (
        "KWFKPLKAIAKSWRSV",  # G8->A
        "KWFKPLKWIAKSWRSV",  # G8->W
        "KWFKPLKGILKSWRSV",  # A10->L
        "KWFKPLKGIFKSWRSV",  # A10->F
        "KWFKPLKGIAKTWRSV",  # S12->T
    ):
        forged_smi = _sanitize_and_canonicalize(sequence_to_mol(bad_seq))
        res_f = gk.check_stage_a(forged_smi, None, is_chemical_leaf=True, lineage_sequence=seq)
        assert res_f.passed is False, f"Supergraph mutation {bad_seq} must NOT pass Stage A!"
        assert any(c in ("CHEMICAL_LEAF_SIDECHAIN_MUTATION", "CHEMICAL_LEAF_SUBSTRUCTURE_MISMATCH") for c in res_f.reject_codes)

    # Genuine leaves on the same Trp/Ser/Gly/Ala peptide MUST STILL PASS!
    leaves = generate_chemical_leaf_candidates(
        mol, seq, _sanitize_and_canonicalize(mol), lock, NegativeSampleMemory(),
        strategies=("n_methylation", "cyclization", "lipidation_c16_c18"), limit=6,
    )
    assert len(leaves) >= 2
    for lf in leaves:
        res_ok = gk.check_stage_a(lf.smiles, None, is_chemical_leaf=True, lineage_sequence=seq, mutation_label=lf.mutation_label)
        assert res_ok.passed is True, (lf.mutation_label, res_ok.reject_codes)

    # 3. R1b P1-3 FULL: rounds=4, max_total_evals=36 with all 5 strategies MUST STILL evaluate chemical leaves!
    adapter = OmniPanelAdapter()
    cfg_r4 = HarnessConfig(
        objective_x="PAMPA",
        objective_y="half_life_reg",
        locked_positions=(1,),
        rounds=4,
        population_size=6,
        max_total_evals=36,
    )
    res_r4 = run_peptide_harness(adapter, sequence="KWFKPLKGI", config=cfg_r4, eval_batch_fn=_fast_mock_eval_for_jobs)
    assert any(c["is_chemical_leaf"] for c in res_r4["candidates"]), "Chemical leaves must never be starved in rounds=4, mte=36!"
    assert res_r4["metrics"]["evals_used"] <= 36

    # 4. Step 2 P1-6: NaN/Inf oracle prediction MUST FAIL CLOSED in Stage B (never feasible=True)!
    nan_preds = _baseline_preds()
    for p in nan_preds:
        if p.endpoint == "hemolysis_cls":
            p.value = float("nan")
    res_nan = gk.check_stage_b(nan_preds, "dual_modality")
    assert res_nan.passed is False and any("NONFINITE_ENDPOINT:hemolysis_cls" in c for c in res_nan.reject_codes)

    # 5. Step 2 P1-2: Attacker spamming 20 submit->cancel cycles MUST NOT evict victim's recent finished job!
    mgr = HarnessJobManager(
        adapter=adapter,
        max_retained=10,
        max_retained_per_client=4,
        min_protected_ttl_s=300.0,
        eval_batch_fn=_fast_mock_eval_for_jobs,
    )
    cfg_small = HarnessConfig(rounds=1, population_size=2, critic_max_perturbations=2, max_total_evals=6)
    v_meta = mgr.submit_job(client_id="victim-client-01", client_ip="10.1.1.1", query=None, sequence="KWFKPLKGI", smiles=None, config=cfg_small)
    import time
    for _ in range(60):
        vd = mgr.get_job(v_meta["job_id"], "victim-client-01")
        if vd and vd["status"] == "completed":
            break
        time.sleep(0.05)
    assert vd and vd["status"] == "completed"
    for i in range(15):
        a_meta = mgr.submit_job(client_id=f"attacker-cid-{i:02d}", client_ip=f"10.2.0.{i+1}", query=None, sequence="KWFKPLKGI", smiles=None, config=cfg_small)
        mgr.cancel_job(a_meta["job_id"], f"attacker-cid-{i:02d}")
    # Victim's finished job MUST STILL EXIST!
    assert mgr.get_job(v_meta["job_id"], "victim-client-01") is not None

    # 6. Step 2 P2-6: Invalid direction rejected cleanly!
    with pytest.raises(ValueError):
        run_peptide_harness(adapter, sequence="KWFKPLKGI", config=HarnessConfig(direction_x="bogus"), eval_batch_fn=_fast_mock_eval_for_jobs)


def test_v352_round_robin_seq_parent_key_metrics_and_leaf_reserve_priority():
    """Verify v35.2 refinements from Step 3/5 red-team audit:
    - generate_sequence_candidates interleaves point_mutation & ala_scan (never starves point_mutation)
    - parent_record exposes key_metrics (hemolysis_cls, PAMPA, half_life_reg) matching candidates
    - leaf-round with evals_used >= seq_eval_cap prioritizes cross-strategy leaves before sequence specs
    """
    from v33_upgrade.web.peptide_harness import (
        HarnessConfig,
        NegativeSampleMemory,
        generate_sequence_candidates,
        resolve_core_locks,
        run_peptide_harness,
    )

    long_seq = "GIGKFLHSAKKFGKAFVGEIMNS"
    mol = sequence_to_mol(long_seq)
    can = _sanitize_and_canonicalize(mol)
    lock = resolve_core_locks(mol, long_seq, locked_positions=[1, 4, 8, 11])
    specs = generate_sequence_candidates(
        long_seq, can, lock, NegativeSampleMemory(), strategies=("ala_scan", "point_mutation"), limit=12
    )
    strats = {s.strategy for s in specs}
    assert "point_mutation" in strats and "ala_scan" in strats, f"Expected both seq strategies, got {strats}"
    assert sum(1 for s in specs if s.strategy == "point_mutation") >= 5

    adapter = OmniPanelAdapter()
    cfg = HarnessConfig(
        objective_x="PAMPA",
        objective_y="half_life_reg",
        locked_positions=(1, 4),
        rounds=3,
        population_size=6,
        critic_max_perturbations=5,
        max_total_evals=20,
        compute_elite_cross_track_audit=False,
    )
    res = run_peptide_harness(adapter, sequence=long_seq, config=cfg, eval_batch_fn=_fast_mock_eval_for_jobs)
    assert res["metrics"]["evals_used"] <= 20
    # parent.key_metrics populated
    p_km = res["parent"].get("key_metrics") or {}
    assert "hemolysis_cls" in p_km and p_km["hemolysis_cls"]["value"] is not None
    assert "PAMPA" in p_km and "half_life_reg" in p_km
    # Tight 20-eval budget still evaluates >= 3 distinct chemical leaves across strategies
    leaf_cands = [c for c in res["candidates"] if c["is_chemical_leaf"]]
    assert len(leaf_cands) >= 3, f"Expected >=3 chemical leaves under mte=20, got {len(leaf_cands)}"
    assert len({c["strategy"] for c in leaf_cands}) >= 2

    # Regression: Round>=2 combinatorial_stack MUST NOT be self-rejected as DUPLICATE_CANDIDATE when use_negative_memory=True!
    from v33_upgrade.web.schemas import BatchMoleculeResult

    def _all_positive_eval(rows, include_tier2, cond):
        out = []
        for r in rows:
            # Make every mutated sequence score higher than parent so >=2 beneficial single edits exist in Round 1
            is_mut = bool(r.sequence and r.sequence != long_seq)
            preds = _baseline_preds()
            for p in preds:
                if p.endpoint == "PAMPA":
                    p.raw_value = -5.5 + (0.6 if is_mut else 0.0)
                    p.value = 10.0 ** p.raw_value
                elif p.endpoint == "half_life_reg":
                    p.raw_value = 0.7 + (0.5 if is_mut else 0.0)
                    p.value = 10.0 ** p.raw_value
            out.append(BatchMoleculeResult(index=r.index, name=r.name, detected_type=r.detected_type, status="ok", predictions=preds))
        return out

    cfg_comb = HarnessConfig(
        objective_x="PAMPA",
        objective_y="half_life_reg",
        locked_positions=(1, 4),
        rounds=3,
        population_size=6,
        critic_max_perturbations=4,
        max_total_evals=24,
        use_negative_memory=True,
        compute_elite_cross_track_audit=False,
    )
    res_comb = run_peptide_harness(adapter, sequence=long_seq, config=cfg_comb, eval_batch_fn=_all_positive_eval)
    comb_cands = [c for c in res_comb["candidates"] if "+" in str(c.get("mutation_label") or "")]
    assert len(comb_cands) >= 2, f"Expected combinatorial stack ('+') to be evaluated with use_negative_memory=True, got {len(comb_cands)}"
    # Verify no combinatorial stack was falsely rejected as DUPLICATE_CANDIDATE in Stage A
    for rd in res_comb["rounds"]:
        for rej in rd.get("stage_a_rejects") or []:
            assert not ("+" in str(rej.get("mutation_label") or "") and "DUPLICATE_CANDIDATE" in (rej.get("codes") or [])), rej




