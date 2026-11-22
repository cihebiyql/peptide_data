#!/usr/bin/env python3
"""46-Endpoint In-VRAM Multimodal Data Bank (50,995 compounds, 82,327 rows) with
Strict Per-Target Zero-Cross-Endpoint-Cluster-Leakage Masking for Plan B-2 Upgrade 1.

Strictly READ-ONLY on:
  - D:\work\中检院\v33_upgrade\models_gpu_deep\_cache\data\ (24 core endpoints + Case17)
  - D:\work\中检院\v33_upgrade\v2_upgrade\gpu_sota\_cache\expansion_v331\ (22 V33.1 endpoints + 50,995 features & 4-FM embeddings)
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

V33_ROOT = Path(r"D:\work\中检院\v33_upgrade")
if str(V33_ROOT / "models_cpu") not in sys.path:
    sys.path.insert(0, str(V33_ROOT / "models_cpu"))
if str(V33_ROOT / "expansion_v331") not in sys.path:
    sys.path.insert(0, str(V33_ROOT / "expansion_v331"))

from v33_config import (  # noqa: E402
    CASE17_CLINICAL_ROUTE,
    ENDPOINT_SPECS as CORE24_SPECS,
)
from v33_features import TanimotoADIndex, fast_tanimoto_max_and_top5  # noqa: E402
from v33_metrics import (  # noqa: E402
    ConformalIntervalCalibrator,
    ProbabilityCalibrator,
    classification_metrics,
    compute_ad_stratification,
    regression_metrics,
)
from v331_config import V331_ENDPOINT_SPECS  # noqa: E402

csv.field_size_limit(10_000_000)

GPU_SOTA_ROOT = V33_ROOT / "v2_upgrade" / "gpu_sota"
CACHE_V331_DIR = GPU_SOTA_ROOT / "_cache" / "expansion_v331"
CORE24_DATA_DIR = V33_ROOT / "models_gpu_deep" / "_cache" / "data"
OUT_MT46_DIR = GPU_SOTA_ROOT / "multitask_v331_22ep"

# Unified 46-endpoint contract (24 core + 22 V33.1 expansion)
ENDPOINT_SPECS_46: Dict[str, Dict[str, Any]] = {}
for _k, _v in CORE24_SPECS.items():
    ENDPOINT_SPECS_46[_k] = dict(_v)
for _k, _v in V331_ENDPOINT_SPECS.items():
    assert _k not in ENDPOINT_SPECS_46, f"Duplicate endpoint {_k}!"
    ENDPOINT_SPECS_46[_k] = dict(_v)

CORE_24_ENDPOINTS: List[str] = sorted(CORE24_SPECS.keys())
NEW_22_ENDPOINTS: List[str] = sorted(V331_ENDPOINT_SPECS.keys())
ALL_46_ENDPOINTS: List[str] = sorted(ENDPOINT_SPECS_46.keys())
assert len(CORE_24_ENDPOINTS) == 24 and len(NEW_22_ENDPOINTS) == 22 and len(ALL_46_ENDPOINTS) == 46
EP46_TO_ID: Dict[str, int] = {ep: i for i, ep in enumerate(ALL_46_ENDPOINTS)}

CENSOR_TO_INT = {"exact": 0, "right": 1, "left": 2, "interval": 3}

# Biophysical families across all 46 endpoints for multi-task transfer sampling
ENDPOINT_FAMILY_46: Dict[str, str] = {
    # Core 24
    "CL_absolute": "pk",
    "CL_per_kg": "pk",
    "Cmax": "pk",
    "F": "pk",
    "PPB": "pk",
    "Vd_per_kg": "pk",
    "half_life_reg": "pk",
    "half_life_cls": "pk",
    "PAMPA": "perm",
    "Caco2": "perm",
    "MDCK": "perm",
    "RRCK": "perm",
    "LogD7.4": "perm",
    "solubility_reg": "perm",
    "solubility_cls": "perm",
    "BBB": "perm",
    "hemolysis_cls": "tox",
    "hemolysis_HC50": "tox",
    "cytotoxicity_cls": "tox",
    "overall_peptide_toxicity": "tox",
    "MIC_E_coli": "mic",
    "MIC_S_aureus": "mic",
    "MIC_P_aeruginosa": "mic",
    "MIC_C_albicans": "mic",
    # New 22 V33.1
    "cytotoxicity_CC50": "tox",
    "hemolysis_MHC": "tox",
    "efflux_ratio": "perm",
    "Vd_absolute": "pk",
    "MIC_B_subtilis": "mic",
    "MIC_A_baumannii": "mic",
    "MIC_K_pneumoniae": "mic",
    "MIC_S_epidermidis": "mic",
    "MIC_Salmonella": "mic",
    "MIC_E_faecalis": "mic",
    "MIC_E_faecium": "mic",
    "MIC_L_monocytogenes": "mic",
    "MIC_C_neoformans": "mic",
    "MBC_S_aureus": "mic",
    "MBC_E_coli": "mic",
    "MBC_P_aeruginosa": "mic",
    "MBC_A_baumannii": "mic",
    "MBC_E_faecalis": "mic",
    "MFC_C_albicans": "mic",
    "MBIC_biofilm": "mic",
    "MBEC_biofilm": "mic",
    "LC_antimicrobial": "mic",
}

# Close mechanistic sibling tasks (extra sampling boost while keeping ALL 45 aux active!)
MECHANISTIC_SIBLINGS: Dict[str, Set[str]] = {
    "cytotoxicity_CC50": {"cytotoxicity_cls", "overall_peptide_toxicity", "hemolysis_HC50", "hemolysis_MHC", "hemolysis_cls"},
    "hemolysis_MHC": {"hemolysis_HC50", "hemolysis_cls", "cytotoxicity_CC50", "cytotoxicity_cls", "overall_peptide_toxicity"},
    "efflux_ratio": {"Caco2", "MDCK", "PAMPA", "RRCK", "LogD7.4", "BBB", "F"},
    "Vd_absolute": {"Vd_per_kg", "CL_absolute", "CL_per_kg", "PPB", "half_life_reg", "F", "Cmax"},
    "MBC_S_aureus": {"MIC_S_aureus", "MIC_S_epidermidis", "MBC_E_coli", "MBC_E_faecalis", "LC_antimicrobial"},
    "MBC_E_coli": {"MIC_E_coli", "MIC_K_pneumoniae", "MIC_Salmonella", "MBC_P_aeruginosa", "MBC_S_aureus"},
    "MBC_P_aeruginosa": {"MIC_P_aeruginosa", "MIC_A_baumannii", "MBC_E_coli", "MBC_A_baumannii", "MBIC_biofilm"},
    "MBC_A_baumannii": {"MIC_A_baumannii", "MIC_P_aeruginosa", "MIC_E_coli", "MBC_P_aeruginosa", "MBC_E_coli"},
    "MBC_E_faecalis": {"MIC_E_faecalis", "MIC_E_faecium", "MIC_S_aureus", "MBC_S_aureus"},
    "MFC_C_albicans": {"MIC_C_albicans", "MIC_C_neoformans", "hemolysis_HC50", "cytotoxicity_CC50"},
    "MIC_C_neoformans": {"MIC_C_albicans", "MFC_C_albicans", "MIC_S_aureus", "MIC_E_coli"},
    "MBIC_biofilm": {"MBEC_biofilm", "MIC_P_aeruginosa", "MIC_S_aureus", "MBC_P_aeruginosa", "MBC_S_aureus"},
    "MBEC_biofilm": {"MBIC_biofilm", "MIC_S_aureus", "MIC_P_aeruginosa", "MBC_S_aureus", "MBC_P_aeruginosa"},
    "LC_antimicrobial": {"MBC_S_aureus", "MBC_E_coli", "MIC_S_aureus", "MIC_E_coli", "MIC_P_aeruginosa", "hemolysis_HC50"},
}


class DeepConditionEncoder:
    """Fold-internal categorical encoder for `cond_*` columns."""

    def __init__(self, cond_cols: List[str], min_count: int = 3):
        self.cond_cols = list(cond_cols)
        self.min_count = min_count
        self.active_cols_: List[str] = []
        self.categories_: Dict[str, List[str]] = {}
        self.feature_names_: List[str] = []

    @staticmethod
    def _clean(val: Any) -> str:
        if val is None:
            return "unspecified"
        s = str(val).strip()
        if s == "" or s.lower() in ("nan", "none"):
            return "unspecified"
        return s

    def fit(self, cond_rows: List[Dict[str, str]]) -> "DeepConditionEncoder":
        self.active_cols_ = []
        self.categories_ = {}
        self.feature_names_ = []
        if not self.cond_cols or len(cond_rows) == 0:
            return self
        for col in self.cond_cols:
            vals = [self._clean(r.get(col, "unspecified")) for r in cond_rows]
            counts: Dict[str, int] = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            collapsed = [v if counts[v] >= self.min_count else "other" for v in vals]
            uniq = sorted(set(collapsed))
            if len(uniq) <= 1:
                continue
            self.active_cols_.append(col)
            self.categories_[col] = uniq
            for cat in uniq:
                self.feature_names_.append(f"{col}={cat}")
        return self

    def transform(self, cond_rows: List[Dict[str, str]]) -> np.ndarray:
        n = len(cond_rows)
        d = len(self.feature_names_)
        if d == 0:
            return np.zeros((n, 0), dtype=np.float32)
        out = np.zeros((n, d), dtype=np.float32)
        offset = 0
        for col in self.active_cols_:
            cats = self.categories_[col]
            cat_to_j = {c: j for j, c in enumerate(cats)}
            other_j = cat_to_j.get("other", -1)
            for i, r in enumerate(cond_rows):
                v = self._clean(r.get(col, "unspecified"))
                j = cat_to_j.get(v, other_j)
                if j >= 0:
                    out[i, offset + j] = 1.0
            offset += len(cats)
        return out

    def transform_constant(self, n: int, default_cond: Dict[str, str]) -> np.ndarray:
        if len(self.feature_names_) == 0:
            return np.zeros((n, 0), dtype=np.float32)
        row = self.transform([default_cond])[0]
        return np.repeat(row[None, :], n, axis=0)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "cond_cols": self.cond_cols,
            "min_count": self.min_count,
            "active_cols_": self.active_cols_,
            "categories_": self.categories_,
            "feature_names_": self.feature_names_,
        }

    @classmethod
    def from_state_dict(cls, sd: Dict[str, Any]) -> "DeepConditionEncoder":
        obj = cls(sd["cond_cols"], sd["min_count"])
        obj.active_cols_ = sd["active_cols_"]
        obj.categories_ = sd["categories_"]
        obj.feature_names_ = sd["feature_names_"]
        return obj


class MultimodalGPUBank46:
    """Read-only 50,995-compound x 46-endpoint multimodal feature & 4-FM embedding bank in GPU VRAM (~1.05 GB)."""

    def __init__(self, device: torch.device):
        self.device = device
        feat_dir = CACHE_V331_DIR / "features"
        emb_dir = CACHE_V331_DIR / "embeddings"
        v331_ep_dir = CACHE_V331_DIR / "data" / "endpoints"
        core_ep_dir = CORE24_DATA_DIR / "endpoints"

        self.manifest = json.loads((feat_dir / "feature_manifest.json").read_text(encoding="utf-8"))
        self.case17_indices: Dict[str, int] = self.manifest["case17_indices"]
        self.physchem_names: List[str] = self.manifest["physchem_core"]["feature_names"]
        self.n_total: int = int(self.manifest["n_compounds_total"])

        # Compound index (50,995 rows)
        self.key_to_idx: Dict[str, int] = {}
        self.idx_to_key: List[str] = []
        with (feat_dir / "compound_index.tsv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                idx = int(r["compound_idx"])
                ck = r["compound_key"]
                self.key_to_idx[ck] = idx
                self.idx_to_key.append(ck)
        assert len(self.idx_to_key) == self.n_total, f"Expected {self.n_total}, got {len(self.idx_to_key)}"
        keys_arr = np.array(self.idx_to_key)

        # Physchem core (83D) + Morgan packed bits for AD
        pc = np.load(feat_dir / "physchem_core.npz")
        self.physchem_cpu: np.ndarray = np.ascontiguousarray(pc["X"], dtype=np.float32)
        self.smiles_valid: np.ndarray = np.ascontiguousarray(pc["smiles_valid_mask"], dtype=bool)
        self.seq_valid: np.ndarray = np.ascontiguousarray(pc["seq_valid_mask"], dtype=bool)
        pc.close()

        mb = np.load(feat_dir / "morgan_bits_1024.npz")
        self.morgan_packed: np.ndarray = np.ascontiguousarray(mb["morgan_bits_packed"], dtype=np.uint8)
        mb.close()

        # Load all 4 verified merged V33.1 embeddings & assert exact compound_keys alignment
        ncaa = np.load(emb_dir / "ncaa_repdistill_v2.npz")
        assert np.array_equal(ncaa["compound_keys"], keys_arr), "ncaa compound_keys mismatch"
        ncaa_smi = np.ascontiguousarray(ncaa["smiles_repr"], dtype=np.float16)
        ncaa_fus = np.ascontiguousarray(ncaa["fusion_hf"], dtype=np.float16)
        ncaa_prot = np.ascontiguousarray(ncaa["protein_proj"], dtype=np.float16)
        ncaa.close()

        pclm = np.load(emb_dir / "peptideclm2.npz")
        assert np.array_equal(pclm["compound_keys"], keys_arr), "peptideclm2 compound_keys mismatch"
        pclm_emb = np.ascontiguousarray(pclm["embeddings"], dtype=np.float16)
        pclm.close()

        cb = np.load(emb_dir / "chemberta_mtr.npz")
        assert np.array_equal(cb["compound_keys"], keys_arr), "chemberta compound_keys mismatch"
        cb_emb = np.ascontiguousarray(cb["embeddings"], dtype=np.float16)
        cb.close()

        esmc = np.load(emb_dir / "esmc_600m_multiregion.npz")
        assert np.array_equal(esmc["compound_keys"], keys_arr), "esmc compound_keys mismatch"
        esmc_4 = np.stack(
            [esmc["mean"], esmc["max"], esmc["nterm5"], esmc["cterm5"]], axis=1
        ).astype(np.float16, copy=False)
        esmc.close()

        # Explicit zero-masking for invalid modalities
        ncaa_smi[~self.smiles_valid] = 0
        ncaa_fus[~self.smiles_valid] = 0
        ncaa_prot[~self.smiles_valid] = 0
        pclm_emb[~self.smiles_valid] = 0
        cb_emb[~self.smiles_valid] = 0
        esmc_4[~self.seq_valid] = 0

        # Move entire read-only bank to GPU VRAM in native bf16 (~1.05 GB total)
        self.phys_chem_raw = torch.from_numpy(self.physchem_cpu[:, :48].copy()).to(device=device, dtype=torch.float32)
        self.phys_seq_raw = torch.from_numpy(self.physchem_cpu[:, 48:83].copy()).to(device=device, dtype=torch.float32)
        self.ncaa_smi = torch.from_numpy(ncaa_smi).to(device=device, dtype=torch.bfloat16)
        self.ncaa_fus = torch.from_numpy(ncaa_fus).to(device=device, dtype=torch.bfloat16)
        self.ncaa_prot = torch.from_numpy(ncaa_prot).to(device=device, dtype=torch.bfloat16)
        self.pclm2 = torch.from_numpy(pclm_emb).to(device=device, dtype=torch.bfloat16)
        self.chemberta = torch.from_numpy(cb_emb).to(device=device, dtype=torch.bfloat16)
        self.esmc_4reg = torch.from_numpy(np.ascontiguousarray(esmc_4)).to(device=device, dtype=torch.bfloat16)
        self.smi_valid_t = torch.from_numpy(self.smiles_valid).to(device=device)
        self.seq_valid_t = torch.from_numpy(self.seq_valid).to(device=device)
        del ncaa_smi, ncaa_fus, ncaa_prot, pclm_emb, cb_emb, esmc_4
        import gc
        gc.collect()

        # Case17 queries
        self.case17_queries: List[Dict[str, Any]] = []
        with (CORE24_DATA_DIR / "case17_queries.tsv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                r["compound_idx"] = self.case17_indices[r["drug_name"]]
                self.case17_queries.append(r)
        assert len(self.case17_queries) == 17

        # Load ALL 46 endpoints (24 core + 22 V33.1)
        self.endpoints: Dict[str, Dict[str, Any]] = {}
        self.all_case17_clusters: Set[str] = set()
        self.cluster_str_to_int: Dict[str, int] = {}
        self.total_rows: int = 0

        for ep in ALL_46_ENDPOINTS:
            ep_path = (core_ep_dir / f"{ep}.tsv") if ep in CORE24_SPECS else (v331_ep_dir / f"{ep}.tsv")
            assert ep_path.exists(), f"Missing endpoint TSV: {ep_path}"
            d = self._load_endpoint(ep, ep_path)
            self.endpoints[ep] = d
            self.total_rows += d["n"]
            c17_m = d["case17_excluded"] == 1
            self.all_case17_clusters.update(d["cluster_ids"][c17_m].tolist())
            for cl in d["cluster_ids"]:
                if cl not in self.cluster_str_to_int:
                    self.cluster_str_to_int[cl] = len(self.cluster_str_to_int)

        self.all_case17_clusters_int = np.array(
            [self.cluster_str_to_int[c] for c in self.all_case17_clusters], dtype=np.int64
        )

        # Pre-encode auxiliary condition tensors strictly on CV rows (and non-Case17 rows)
        for ep, d in self.endpoints.items():
            d["cluster_int"] = np.array([self.cluster_str_to_int[c] for c in d["cluster_ids"]], dtype=np.int64)
            cv_enc = DeepConditionEncoder(d["cond_cols"], min_count=3).fit([d["cond_rows"][i] for i in d["cv_idx"]])
            d["aux_cond_enc_cv"] = cv_enc
            d["aux_cond_mat_all_cvfit"] = torch.from_numpy(cv_enc.transform(d["cond_rows"])).to(self.device)

            nc17_idx = np.where(
                (d["case17_excluded"] == 0) & (~np.isin(d["cluster_int"], self.all_case17_clusters_int))
            )[0]
            c17_enc = DeepConditionEncoder(d["cond_cols"], min_count=3).fit([d["cond_rows"][i] for i in nc17_idx])
            d["aux_cond_enc_c17"] = c17_enc
            d["aux_cond_mat_all_c17fit"] = torch.from_numpy(c17_enc.transform(d["cond_rows"])).to(self.device)
            d["cidx_t"] = torch.from_numpy(d["compound_idx"]).to(self.device)
            d["censor_t"] = torch.from_numpy(d["censor_int"]).to(self.device)

    def _load_endpoint(self, endpoint: str, path: Path) -> Dict[str, Any]:
        with path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            cols = list(reader.fieldnames or [])
            rows = list(reader)

        cond_cols = [c for c in cols if c.startswith("cond_")]
        n = len(rows)
        record_ids = [r["record_id"] for r in rows]
        compound_keys = [r["compound_key"] for r in rows]
        compound_idx = np.array([self.key_to_idx[k] for k in compound_keys], dtype=np.int64)
        cluster_ids = np.array([r["cluster_id"] for r in rows], dtype=object)
        is_holdout = np.array([int(r["is_holdout"]) for r in rows], dtype=np.int32)
        fold_id = np.array([int(r["fold_id"]) for r in rows], dtype=np.int32)
        case17_excluded = np.array([int(r["case17_excluded"]) for r in rows], dtype=np.int32)
        censor_type = np.array([r["censor_type"] for r in rows], dtype=object)
        censor_int = np.array([CENSOR_TO_INT[c] for c in censor_type], dtype=np.int64)
        y_value = np.array([float(r["y_value"]) for r in rows], dtype=np.float64)
        y_lower = np.array([float(r["y_lower"]) for r in rows], dtype=np.float64)
        y_upper = np.array([float(r["y_upper"]) for r in rows], dtype=np.float64)
        cond_rows = [{c: r.get(c, "unspecified") for c in cond_cols} for r in rows]

        cv_mask = is_holdout == 0
        ho_mask = is_holdout == 1
        assert set(cluster_ids[cv_mask]).isdisjoint(set(cluster_ids[ho_mask])), f"{endpoint}: HO/CV cluster leak"

        return {
            "endpoint": endpoint,
            "task_id": EP46_TO_ID[endpoint],
            "spec": ENDPOINT_SPECS_46[endpoint],
            "task_type": ENDPOINT_SPECS_46[endpoint]["task_type"],
            "n": n,
            "cond_cols": cond_cols,
            "record_ids": record_ids,
            "compound_keys": compound_keys,
            "compound_idx": compound_idx,
            "cluster_ids": cluster_ids,
            "is_holdout": is_holdout,
            "fold_id": fold_id,
            "case17_excluded": case17_excluded,
            "censor_type": censor_type,
            "censor_int": censor_int,
            "y_value": y_value,
            "y_lower": y_lower,
            "y_upper": y_upper,
            "cond_rows": cond_rows,
            "cv_idx": np.where(cv_mask)[0],
            "ho_idx": np.where(ho_mask)[0],
        }

    def compute_phys_stats(self, train_cidx: np.ndarray) -> Dict[str, torch.Tensor]:
        uniq = np.unique(train_cidx)
        smi_u = uniq[self.smiles_valid[uniq]]
        seq_u = uniq[self.seq_valid[uniq]]

        if len(smi_u) >= 2:
            pc_m = self.physchem_cpu[smi_u, :48].mean(axis=0)
            pc_s = self.physchem_cpu[smi_u, :48].std(axis=0)
            pc_s[pc_s < 1e-5] = 1.0
        else:
            pc_m = np.zeros(48, dtype=np.float32)
            pc_s = np.ones(48, dtype=np.float32)

        if len(seq_u) >= 2:
            ps_m = self.physchem_cpu[seq_u, 48:83].mean(axis=0)
            ps_s = self.physchem_cpu[seq_u, 48:83].std(axis=0)
            ps_s[ps_s < 1e-5] = 1.0
        else:
            ps_m = np.zeros(35, dtype=np.float32)
            ps_s = np.ones(35, dtype=np.float32)

        return {
            "pc_mean": torch.from_numpy(pc_m.astype(np.float32)).to(self.device),
            "pc_std": torch.from_numpy(pc_s.astype(np.float32)).to(self.device),
            "ps_mean": torch.from_numpy(ps_m.astype(np.float32)).to(self.device),
            "ps_std": torch.from_numpy(ps_s.astype(np.float32)).to(self.device),
        }

    def gather_inputs(
        self,
        cidx: torch.Tensor,
        phys_stats: Dict[str, torch.Tensor],
        eval_mode: str = "dual_modality",
    ) -> Dict[str, torch.Tensor]:
        smi_m = self.smi_valid_t[cidx]
        seq_m = self.seq_valid_t[cidx]

        p_chem = ((self.phys_chem_raw[cidx] - phys_stats["pc_mean"]) / phys_stats["pc_std"]).clamp_(-8.0, 8.0)
        p_seq = ((self.phys_seq_raw[cidx] - phys_stats["ps_mean"]) / phys_stats["ps_std"]).clamp_(-8.0, 8.0)

        if eval_mode == "dual_modality":
            p_chem = p_chem * smi_m.unsqueeze(-1).to(p_chem.dtype)
            p_seq = p_seq * seq_m.unsqueeze(-1).to(p_seq.dtype)
            return {
                "phys_chem": p_chem,
                "phys_seq": p_seq,
                "ncaa_smi": self.ncaa_smi[cidx],
                "ncaa_fus": self.ncaa_fus[cidx],
                "ncaa_prot": self.ncaa_prot[cidx],
                "pclm2": self.pclm2[cidx],
                "chemberta": self.chemberta[cidx],
                "esmc_4reg": self.esmc_4reg[cidx],
                "smi_mask": smi_m,
                "seq_mask": seq_m,
            }

        if eval_mode == "smiles_only":
            seq_zero = torch.zeros_like(seq_m)
            return {
                "phys_chem": p_chem * smi_m.unsqueeze(-1).to(p_chem.dtype),
                "phys_seq": torch.zeros_like(p_seq),
                "ncaa_smi": self.ncaa_smi[cidx],
                "ncaa_fus": self.ncaa_fus[cidx],
                "ncaa_prot": self.ncaa_prot[cidx],
                "pclm2": self.pclm2[cidx],
                "chemberta": self.chemberta[cidx],
                "esmc_4reg": torch.zeros((len(cidx), 4, 1152), device=self.device, dtype=torch.bfloat16),
                "smi_mask": smi_m,
                "seq_mask": seq_zero,
            }

        smi_zero = torch.zeros_like(smi_m)
        return {
            "phys_chem": torch.zeros_like(p_chem),
            "phys_seq": p_seq * seq_m.unsqueeze(-1).to(p_seq.dtype),
            "ncaa_smi": torch.zeros((len(cidx), 768), device=self.device, dtype=torch.bfloat16),
            "ncaa_fus": torch.zeros((len(cidx), 2048), device=self.device, dtype=torch.bfloat16),
            "ncaa_prot": torch.zeros((len(cidx), 1280), device=self.device, dtype=torch.bfloat16),
            "pclm2": torch.zeros((len(cidx), 512), device=self.device, dtype=torch.bfloat16),
            "chemberta": torch.zeros((len(cidx), 384), device=self.device, dtype=torch.bfloat16),
            "esmc_4reg": self.esmc_4reg[cidx],
            "smi_mask": smi_zero,
            "seq_mask": seq_m,
        }

    def build_leak_free_multitask_split_46(
        self,
        target_ep: str,
        fold: int,
        aux_endpoints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build 100% ZERO-LEAKAGE 46-endpoint multi-task split for (target_ep, fold).

        HARD PROGRAMMATIC ZERO-LEAKAGE ASSERTIONS:
          - forbidden_clusters = HoldoutClusters(target_ep) | ValFoldClusters(target_ep, fold)
          - forbidden_cidx     = HoldoutCidx(target_ep) | ValFoldCidx(target_ep, fold)
          - For target_ep: 100% of CV rows with fold_id != fold
          - For ALL 45 auxiliary endpoints: ONLY CV rows (is_holdout == 0) whose
            cluster_id NOT IN forbidden_clusters AND compound_idx NOT IN forbidden_cidx!
          - `assert len(train_clusters_all_46 & forbidden_clusters) == 0`
          - `assert len(train_cidx_all_46 & forbidden_cidx) == 0`
        """
        if aux_endpoints is None:
            aux_endpoints = [ep for ep in ALL_46_ENDPOINTS if ep != target_ep]
        else:
            aux_endpoints = [ep for ep in aux_endpoints if ep != target_ep]

        t_data = self.endpoints[target_ep]
        active_eps = [target_ep] + aux_endpoints

        ho_rows = t_data["ho_idx"]
        cv_rows = t_data["cv_idx"]
        va_rows = cv_rows[t_data["fold_id"][cv_rows] == fold]
        tr_rows = cv_rows[t_data["fold_id"][cv_rows] != fold]

        ho_clusters = set(t_data["cluster_ids"][ho_rows].tolist())
        va_clusters = set(t_data["cluster_ids"][va_rows].tolist())
        forbidden_clusters = ho_clusters | va_clusters

        ho_cidx = set(t_data["compound_idx"][ho_rows].tolist())
        va_cidx = set(t_data["compound_idx"][va_rows].tolist())
        forbidden_cidx = ho_cidx | va_cidx

        forbidden_int = np.array([self.cluster_str_to_int[c] for c in forbidden_clusters], dtype=np.int64)
        forbidden_cidx_arr = np.array(sorted(forbidden_cidx), dtype=np.int64)

        task_train_rows: Dict[str, np.ndarray] = {target_ep: tr_rows}
        train_clusters_all_46: Set[str] = set(t_data["cluster_ids"][tr_rows].tolist())
        train_cidx_all_46: Set[int] = set(t_data["compound_idx"][tr_rows].tolist())
        n_aux_rows_dropped_for_leak = 0
        n_aux_rows_kept = 0

        for a_ep in aux_endpoints:
            a_data = self.endpoints[a_ep]
            cand = a_data["cv_idx"]
            keep_mask = (~np.isin(a_data["cluster_int"][cand], forbidden_int)) & (
                ~np.isin(a_data["compound_idx"][cand], forbidden_cidx_arr)
            )
            kept = cand[keep_mask]
            n_aux_rows_dropped_for_leak += int((~keep_mask).sum())
            n_aux_rows_kept += int(len(kept))
            if len(kept) >= 4:
                task_train_rows[a_ep] = kept
                train_clusters_all_46.update(a_data["cluster_ids"][kept].tolist())
                train_cidx_all_46.update(a_data["compound_idx"][kept].tolist())

        # =====================================================================
        # HARD ZERO-LEAKAGE RUNTIME ASSERTIONS (CLUSTER + COMPOUND!)
        # =====================================================================
        assert len(train_clusters_all_46 & forbidden_clusters) == 0, (
            f"CRITICAL CLUSTER LEAKAGE for {target_ep} fold={fold}: "
            f"{len(train_clusters_all_46 & forbidden_clusters)} clusters!"
        )
        assert len(train_cidx_all_46 & forbidden_cidx) == 0, (
            f"CRITICAL COMPOUND LEAKAGE for {target_ep} fold={fold}: "
            f"{len(train_cidx_all_46 & forbidden_cidx)} compounds!"
        )

        # Fold-internal physchem standardizer on clean training compounds
        all_tr_cidx = np.concatenate([self.endpoints[ep_k]["compound_idx"][r_idx] for ep_k, r_idx in task_train_rows.items()])
        phys_stats = self.compute_phys_stats(all_tr_cidx)

        cond_encoders: Dict[str, DeepConditionEncoder] = {}
        y_scalers: Dict[str, Tuple[float, float]] = {}
        task_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
        cond_dims: Dict[str, int] = {}

        for ep_k in active_eps:
            d = self.endpoints[ep_k]
            r_idx = task_train_rows.get(ep_k, np.array([], dtype=np.int64))
            if len(r_idx) == 0:
                continue
            if ep_k == target_ep:
                enc = DeepConditionEncoder(d["cond_cols"], min_count=3).fit([d["cond_rows"][i] for i in r_idx])
                cond_encoders[ep_k] = enc
                cond_dims[ep_k] = len(enc.feature_names_)
                c_ten = torch.from_numpy(enc.transform([d["cond_rows"][i] for i in r_idx])).to(self.device)
            else:
                enc = d["aux_cond_enc_cv"]
                cond_encoders[ep_k] = enc
                cond_dims[ep_k] = len(enc.feature_names_)
                c_ten = d["aux_cond_mat_all_cvfit"][torch.from_numpy(r_idx).to(self.device)]

            y_v = d["y_value"][r_idx].copy()
            y_l = d["y_lower"][r_idx].copy()
            y_u = d["y_upper"][r_idx].copy()
            ct = d["censor_int"][r_idx].copy()

            if d["task_type"] == "regression":
                cap = d["spec"].get("outlier_winsor_cap")
                if cap is not None:
                    ex_m = ct == 0
                    y_v[ex_m] = np.minimum(y_v[ex_m], float(cap))
                ex_m = ct == 0
                ref_y = y_v[ex_m] if ex_m.sum() >= 3 else y_v
                ym = float(np.mean(ref_y))
                ys = float(np.std(ref_y))
                if not np.isfinite(ys) or ys < 1e-4:
                    ys = 1.0
                y_scalers[ep_k] = (ym, ys)
                y_v = (y_v - ym) / ys
                y_l = (y_l - ym) / ys
                y_u = (y_u - ym) / ys
            else:
                y_scalers[ep_k] = (0.0, 1.0)

            r_idx_t = torch.from_numpy(r_idx).to(self.device)
            task_tensors[ep_k] = {
                "cidx": d["cidx_t"][r_idx_t],
                "cond": c_ten,
                "y": torch.from_numpy(y_v.astype(np.float32)).to(self.device),
                "y_lo": torch.from_numpy(y_l.astype(np.float32)).to(self.device),
                "y_hi": torch.from_numpy(y_u.astype(np.float32)).to(self.device),
                "censor": d["censor_t"][r_idx_t],
                "has_censor": bool((ct != 0).any()),
            }

        t_enc = cond_encoders[target_ep]
        va_cond = torch.from_numpy(t_enc.transform([t_data["cond_rows"][i] for i in va_rows])).to(self.device)
        ho_cond = torch.from_numpy(t_enc.transform([t_data["cond_rows"][i] for i in ho_rows])).to(self.device)

        va_pack = {
            "orig_rows": va_rows,
            "cidx": torch.from_numpy(t_data["compound_idx"][va_rows]).to(self.device),
            "cond": va_cond,
            "y_raw": t_data["y_value"][va_rows],
            "y_lo_raw": t_data["y_lower"][va_rows],
            "y_hi_raw": t_data["y_upper"][va_rows],
            "censor_str": t_data["censor_type"][va_rows],
            "censor_int": t_data["censor_int"][va_rows],
        }
        ho_pack = {
            "orig_rows": ho_rows,
            "cidx": torch.from_numpy(t_data["compound_idx"][ho_rows]).to(self.device),
            "cond": ho_cond,
            "y_raw": t_data["y_value"][ho_rows],
            "y_lo_raw": t_data["y_lower"][ho_rows],
            "y_hi_raw": t_data["y_upper"][ho_rows],
            "censor_str": t_data["censor_type"][ho_rows],
            "censor_int": t_data["censor_int"][ho_rows],
        }

        return {
            "target_ep": target_ep,
            "fold": fold,
            "active_eps": list(task_tensors.keys()),
            "phys_stats": phys_stats,
            "cond_encoders": cond_encoders,
            "cond_dims": cond_dims,
            "y_scalers": y_scalers,
            "task_tensors": task_tensors,
            "val": va_pack,
            "holdout": ho_pack,
            "audit": {
                "n_active_tasks": len(task_tensors),
                "n_train_clusters_all_46": len(train_clusters_all_46),
                "n_forbidden_clusters": len(forbidden_clusters),
                "n_leaked_clusters": 0,
                "n_leaked_compounds": 0,
                "n_target_train_rows": int(len(tr_rows)),
                "n_target_val_rows": int(len(va_rows)),
                "n_aux_rows_kept": n_aux_rows_kept,
                "n_aux_rows_dropped_for_leak": n_aux_rows_dropped_for_leak,
            },
        }


def compute_oof_tanimoto_46(bank: MultimodalGPUBank46, ep_data: Dict[str, Any]) -> np.ndarray:
    cv_rows = ep_data["cv_idx"]
    cidx_cv = ep_data["compound_idx"][cv_rows]
    fids_cv = ep_data["fold_id"][cv_rows]
    out = np.full(len(cv_rows), np.nan, dtype=np.float64)
    for f in range(5):
        va_m = fids_cv == f
        tr_m = fids_cv != f
        tr_u = np.unique(cidx_cv[tr_m])
        tr_u = tr_u[bank.smiles_valid[tr_u]]
        if len(tr_u) == 0 or not va_m.any():
            continue
        va_cidx = cidx_cv[va_m]
        max_t, _ = fast_tanimoto_max_and_top5(bank.morgan_packed[va_cidx], bank.morgan_packed[tr_u])
        max_t = max_t.astype(np.float64)
        max_t[~bank.smiles_valid[va_cidx]] = np.nan
        out[va_m] = max_t
    return out
