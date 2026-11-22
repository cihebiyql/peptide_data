#!/usr/bin/env python3
"""In-VRAM Multimodal Data Bank, Strict Zero-Leakage Cluster Masking, and Condition Encoding
for Peptide OmniPanel V33 Deep Multi-Task & Attention Neural Architectures.

Strictly reads from local cache (mirrored read-only from node23:/data/qlyu/v33_upgrade/).
Never modifies data/, features/, embeddings/, models_cpu/, or models_fusion/.
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

# Import read-only config & metrics helpers from models_cpu (never modifying them)
V33_ROOT = Path(os.environ.get("V33_BASE_DIR", str(Path(__file__).resolve().parent.parent)))
if str(V33_ROOT / "models_cpu") not in sys.path:
    sys.path.insert(0, str(V33_ROOT / "models_cpu"))

from v33_config import (  # noqa: E402
    CASE17_CLINICAL_ROUTE,
    CASE17_FDA_NUMERIC,
    ENDPOINT_SPECS,
    compute_halflife_mechanism_anchor,
)
from v33_features import TanimotoADIndex, fast_tanimoto_max_and_top5  # noqa: E402
from v33_metrics import (  # noqa: E402
    ConformalIntervalCalibrator,
    ProbabilityCalibrator,
    classification_metrics,
    compute_ad_stratification,
    regression_metrics,
)

csv.field_size_limit(10_000_000)

CACHE_DIR = V33_ROOT / "models_gpu_deep" / "_cache"
EMB_LOCAL_DIR = V33_ROOT / "models_gpu" / "_local_cache" / "embeddings"
OUT_ROOT = V33_ROOT / "models_gpu_deep"

ALL_ENDPOINTS: List[str] = sorted(ENDPOINT_SPECS.keys())
EP_TO_ID: Dict[str, int] = {ep: i for i, ep in enumerate(ALL_ENDPOINTS)}

CENSOR_TO_INT = {"exact": 0, "right": 1, "left": 2, "interval": 3}

# Domain-relatedness boost for balanced auxiliary sampling (all 23 aux tasks still active!)
ENDPOINT_FAMILY: Dict[str, str] = {
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
}

# Indices within physchem_core[:, :48] for explicit modification + PK-critical descriptors
# 38..47 are the 10 peptide modification flags:
# is_cyclic, macrocycle_ring_size, n_disulfide, n_nmethyl, n_d_aa_approx,
# fatty_diacid_len, fatty_monoacyl_len, has_peg, has_c_amidation, has_n_acetylation
MOD_FEATURE_INDICES = list(range(38, 48))
PK_CORE_INDICES = [0, 1, 3, 5, 6, 8, 9, 11, 13, 15, 23, 25, 32, 33, 34, 35, 36, 37] + MOD_FEATURE_INDICES


class DeepConditionEncoder:
    """Fold-internal categorical encoder for `cond_*` columns with optional pairwise interactions."""

    def __init__(self, cond_cols: List[str], min_count: int = 3, add_pairs: bool = False):
        self.cond_cols = list(cond_cols)
        self.min_count = min_count
        self.add_pairs = add_pairs
        self.active_cols_: List[str] = []
        self.categories_: Dict[str, List[str]] = {}
        self.pair_defs_: List[Tuple[str, str]] = []
        self.pair_categories_: Dict[str, List[str]] = {}
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
        self.pair_defs_ = []
        self.pair_categories_ = {}
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

        if self.add_pairs and len(self.active_cols_) >= 2:
            for i in range(len(self.active_cols_)):
                for j in range(i + 1, len(self.active_cols_)):
                    c1, c2 = self.active_cols_[i], self.active_cols_[j]
                    pkey = f"{c1}__{c2}"
                    vals = [f"{self._clean(r.get(c1))}|{self._clean(r.get(c2))}" for r in cond_rows]
                    counts = {}
                    for v in vals:
                        counts[v] = counts.get(v, 0) + 1
                    uniq = sorted([v for v, cnt in counts.items() if cnt >= max(8, self.min_count * 2)])
                    if len(uniq) >= 2:
                        self.pair_defs_.append((c1, c2))
                        self.pair_categories_[pkey] = uniq
                        for cat in uniq:
                            self.feature_names_.append(f"{pkey}={cat}")
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

        for c1, c2 in self.pair_defs_:
            pkey = f"{c1}__{c2}"
            cats = self.pair_categories_[pkey]
            cat_to_j = {c: j for j, c in enumerate(cats)}
            for i, r in enumerate(cond_rows):
                v = f"{self._clean(r.get(c1))}|{self._clean(r.get(c2))}"
                j = cat_to_j.get(v, -1)
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
            "add_pairs": self.add_pairs,
            "active_cols_": self.active_cols_,
            "categories_": self.categories_,
            "pair_defs_": self.pair_defs_,
            "pair_categories_": self.pair_categories_,
            "feature_names_": self.feature_names_,
        }

    @classmethod
    def from_state_dict(cls, sd: Dict[str, Any]) -> "DeepConditionEncoder":
        obj = cls(sd["cond_cols"], sd["min_count"], sd.get("add_pairs", False))
        obj.active_cols_ = sd["active_cols_"]
        obj.categories_ = sd["categories_"]
        obj.pair_defs_ = [tuple(x) for x in sd.get("pair_defs_", [])]
        obj.pair_categories_ = sd.get("pair_categories_", {})
        obj.feature_names_ = sd["feature_names_"]
        return obj


class MultimodalGPUBank:
    """Read-only multimodal feature + 4-embedding bank resident on GPU VRAM (~0.95 GB)."""

    def __init__(self, device: torch.device):
        self.device = device
        feat_dir = CACHE_DIR / "features"
        data_dir = CACHE_DIR / "data"

        self.manifest = json.loads((feat_dir / "feature_manifest.json").read_text(encoding="utf-8"))
        self.case17_indices: Dict[str, int] = self.manifest["case17_indices"]
        self.physchem_names: List[str] = self.manifest["physchem_core"]["feature_names"]

        # Compound index
        self.key_to_idx: Dict[str, int] = {}
        self.idx_to_key: List[str] = []
        with (feat_dir / "compound_index.tsv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                idx = int(r["compound_idx"])
                ck = r["compound_key"]
                self.key_to_idx[ck] = idx
                self.idx_to_key.append(ck)
        keys_arr = np.array(self.idx_to_key)

        # Physchem core + Morgan packed bits for AD
        pc = np.load(feat_dir / "physchem_core.npz")
        self.physchem_cpu: np.ndarray = np.ascontiguousarray(pc["X"], dtype=np.float32)
        self.smiles_valid: np.ndarray = np.ascontiguousarray(pc["smiles_valid_mask"], dtype=bool)
        self.seq_valid: np.ndarray = np.ascontiguousarray(pc["seq_valid_mask"], dtype=bool)
        pc.close()

        mb = np.load(feat_dir / "morgan_bits_1024.npz")
        self.morgan_packed: np.ndarray = np.ascontiguousarray(mb["morgan_bits_packed"], dtype=np.uint8)
        mb.close()

        # Load all 4 verified embeddings and assert exact alignment
        ncaa = np.load(EMB_LOCAL_DIR / "ncaa_repdistill_v2.npz")
        assert np.array_equal(ncaa["compound_keys"], keys_arr), "ncaa compound_keys mismatch"
        ncaa_smi = np.ascontiguousarray(ncaa["smiles_repr"], dtype=np.float16)
        ncaa_fus = np.ascontiguousarray(ncaa["fusion_hf"], dtype=np.float16)
        ncaa_prot = np.ascontiguousarray(ncaa["protein_proj"], dtype=np.float16)
        ncaa.close()

        pclm = np.load(EMB_LOCAL_DIR / "peptideclm2.npz")
        assert np.array_equal(pclm["compound_keys"], keys_arr), "peptideclm2 compound_keys mismatch"
        pclm_emb = np.ascontiguousarray(pclm["embeddings"], dtype=np.float16)
        pclm.close()

        cb = np.load(CACHE_DIR / "embeddings" / "chemberta_mtr.npz")
        assert np.array_equal(cb["compound_keys"], keys_arr), "chemberta compound_keys mismatch"
        cb_emb = np.ascontiguousarray(cb["embeddings"], dtype=np.float16)
        cb.close()

        esmc = np.load(EMB_LOCAL_DIR / "esmc_600m_multiregion.npz")
        assert np.array_equal(esmc["compound_keys"], keys_arr), "esmc compound_keys mismatch"
        esmc_4 = np.stack(
            [esmc["mean"], esmc["max"], esmc["nterm5"], esmc["cterm5"]], axis=1
        ).astype(np.float16, copy=False)
        esmc.close()

        # Zero out invalid rows explicitly as defense-in-depth
        ncaa_smi[~self.smiles_valid] = 0
        ncaa_fus[~self.smiles_valid] = 0
        ncaa_prot[~self.smiles_valid] = 0
        pclm_emb[~self.smiles_valid] = 0
        cb_emb[~self.smiles_valid] = 0
        esmc_4[~self.seq_valid] = 0

        # Move entire read-only bank to GPU VRAM (~0.95 GB total) in native bf16 for zero-copy gathers!
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
        with (data_dir / "case17_queries.tsv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                r["compound_idx"] = self.case17_indices[r["drug_name"]]
                self.case17_queries.append(r)
        assert len(self.case17_queries) == 17

        # Load all 24 endpoints + pre-encode auxiliary condition matrices (CV-only & non-Case17) ONCE
        self.endpoints: Dict[str, Dict[str, Any]] = {}
        self.all_case17_clusters: Set[str] = set()
        self.cluster_str_to_int: Dict[str, int] = {}
        for ep in ALL_ENDPOINTS:
            d = self._load_endpoint(ep, data_dir / "endpoints" / f"{ep}.tsv")
            self.endpoints[ep] = d
            c17_m = d["case17_excluded"] == 1
            self.all_case17_clusters.update(d["cluster_ids"][c17_m].tolist())
            for cl in d["cluster_ids"]:
                if cl not in self.cluster_str_to_int:
                    self.cluster_str_to_int[cl] = len(self.cluster_str_to_int)
        self.all_case17_clusters_int = np.array(
            [self.cluster_str_to_int[c] for c in self.all_case17_clusters], dtype=np.int64
        )
        for ep, d in self.endpoints.items():
            d["cluster_int"] = np.array([self.cluster_str_to_int[c] for c in d["cluster_ids"]], dtype=np.int64)
            # Pre-encode auxiliary condition tensors strictly on CV rows (never peeking at holdout!)
            cv_enc = DeepConditionEncoder(d["cond_cols"], min_count=3).fit([d["cond_rows"][i] for i in d["cv_idx"]])
            d["aux_cond_enc_cv"] = cv_enc
            d["aux_cond_mat_all_cvfit"] = torch.from_numpy(cv_enc.transform(d["cond_rows"])).to(self.device)
            # Pre-encode auxiliary condition tensors strictly on non-Case17 rows for Case17 runs
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
            "task_id": EP_TO_ID[endpoint],
            "spec": ENDPOINT_SPECS[endpoint],
            "task_type": ENDPOINT_SPECS[endpoint]["task_type"],
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
        """Compute strictly fold-internal mean/std on physchem blocks."""
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
        """Fast zero-copy GPU gather for a batch of compound indices (bf16 native embeddings).

        eval_mode:
          - 'dual_modality': uses real smiles_valid & seq_valid masks (invalid rows already zeroed!)
          - 'smiles_only': STRICTLY zeroes all sequence features & sets seq_mask=0
          - 'seq_only': STRICTLY zeroes all SMILES features & sets smi_mask=0
        """
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

        # seq_only
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

    def build_leak_free_multitask_split(
        self,
        target_ep: str,
        fold: int,
        aux_endpoints: Optional[List[str]] = None,
        add_cond_pairs_for: Optional[Set[str]] = None,
        leave_case17_out: bool = False,
        case17_val_fold: int = 0,
    ) -> Dict[str, Any]:
        """Build 100% ZERO-LEAKAGE multi-task training/val/holdout tensors for (target_ep, fold).

        HARD ZERO-LEAKAGE GUARANTEE:
        - Normal CV/HO mode:
          forbidden_clusters = HoldoutClusters(target_ep) | ValFoldClusters(target_ep, fold)
          For target_ep: train = CV rows with fold_id != fold; val = CV rows with fold_id == fold; ho = is_holdout == 1
          For EVERY auxiliary endpoint A != target_ep: ONLY CV rows (is_holdout == 0) whose cluster_id NOT IN forbidden_clusters!
          HARD RUNTIME ASSERTION: len(all_train_clusters_across_all_tasks & forbidden_clusters) == 0!
        - Leave-Case17-Out mode:
          Exclude ALL rows with case17_excluded == 1 OR cluster_id in ALL_CASE17_CLUSTERS across ALL 24 endpoints!
          Split remaining non-Case17 rows of target_ep into 5 cluster-grouped folds for early stopping & 5-fold ensemble.
          HARD RUNTIME ASSERTION: len(all_train_clusters & ALL_CASE17_CLUSTERS) == 0 AND 0 case17_excluded rows!
        """
        if aux_endpoints is None:
            aux_endpoints = [ep for ep in ALL_ENDPOINTS if ep != target_ep]
        else:
            aux_endpoints = [ep for ep in aux_endpoints if ep != target_ep]
        add_cond_pairs_for = add_cond_pairs_for or set()

        t_data = self.endpoints[target_ep]
        active_eps = [target_ep] + aux_endpoints

        if not leave_case17_out:
            ho_rows = t_data["ho_idx"]
            cv_rows = t_data["cv_idx"]
            va_rows = cv_rows[t_data["fold_id"][cv_rows] == fold]
            tr_rows = cv_rows[t_data["fold_id"][cv_rows] != fold]

            ho_clusters = set(t_data["cluster_ids"][ho_rows].tolist())
            va_clusters = set(t_data["cluster_ids"][va_rows].tolist())
            forbidden_clusters = ho_clusters | va_clusters
            eval_clusters = forbidden_clusters
        else:
            # Strict Leave-Case17-Out across ALL endpoints: exclude both flagged rows AND all Case17 clusters
            t_non_c17 = np.where(
                (t_data["case17_excluded"] == 0)
                & (~np.isin(t_data["cluster_int"], self.all_case17_clusters_int))
            )[0]
            # Deterministic cluster-based 5-fold split on non-Case17 clusters for early stopping
            uniq_cl = sorted(set(t_data["cluster_ids"][t_non_c17].tolist()))
            cl_to_f = {cl: (i % 5) for i, cl in enumerate(uniq_cl)}
            f_arr = np.array([cl_to_f[c] for c in t_data["cluster_ids"][t_non_c17]], dtype=np.int32)
            va_rows = t_non_c17[f_arr == case17_val_fold]
            tr_rows = t_non_c17[f_arr != case17_val_fold]
            if len(va_rows) < 2:
                va_rows = t_non_c17[: max(2, len(t_non_c17) // 5)]
                tr_rows = t_non_c17[max(2, len(t_non_c17) // 5) :]
            ho_rows = np.array([], dtype=np.int64)
            va_clusters = set(t_data["cluster_ids"][va_rows].tolist())
            forbidden_clusters = self.all_case17_clusters | va_clusters
            eval_clusters = self.all_case17_clusters | va_clusters

        # Assemble per-task clean training rows & strictly verify ZERO cluster leakage (vectorized!)
        forbidden_int = np.array([self.cluster_str_to_int[c] for c in forbidden_clusters], dtype=np.int64)
        task_train_rows: Dict[str, np.ndarray] = {target_ep: tr_rows}
        all_train_clusters: Set[str] = set(t_data["cluster_ids"][tr_rows].tolist())
        n_aux_rows_dropped_for_leak = 0
        n_aux_rows_kept = 0

        for a_ep in aux_endpoints:
            a_data = self.endpoints[a_ep]
            if not leave_case17_out:
                cand = a_data["cv_idx"]
            else:
                cand = np.where(a_data["case17_excluded"] == 0)[0]
            keep_mask = ~np.isin(a_data["cluster_int"][cand], forbidden_int)
            kept = cand[keep_mask]
            n_aux_rows_dropped_for_leak += int((~keep_mask).sum())
            n_aux_rows_kept += int(len(kept))
            if len(kept) >= 4:
                task_train_rows[a_ep] = kept
                all_train_clusters.update(a_data["cluster_ids"][kept].tolist())

        # =====================================================================
        # HARD ZERO-LEAKAGE RUNTIME ASSERTION (MATHEMATICALLY ENFORCED!)
        # =====================================================================
        leak_overlap = all_train_clusters & eval_clusters
        assert len(leak_overlap) == 0, (
            f"CRITICAL LEAKAGE DETECTED for {target_ep} fold={fold}: "
            f"{len(leak_overlap)} clusters leaked! Sample: {list(leak_overlap)[:5]}"
        )
        if leave_case17_out:
            c17_cidx_set = set(self.case17_indices.values())
            for ep_k, r_idx in task_train_rows.items():
                assert int((self.endpoints[ep_k]["case17_excluded"][r_idx] == 1).sum()) == 0, (
                    f"CRITICAL CASE17 LEAKAGE in {ep_k}!"
                )
                tr_cidx_ep = set(self.endpoints[ep_k]["compound_idx"][r_idx].tolist())
                assert tr_cidx_ep.isdisjoint(c17_cidx_set), f"CRITICAL CASE17 compound_idx LEAKAGE in {ep_k}!"
            assert len(all_train_clusters & self.all_case17_clusters) == 0, "Case17 cluster leaked!"

        # Fit strictly fold-internal physchem standardizer on ALL clean training compounds
        all_tr_cidx = np.concatenate([self.endpoints[ep_k]["compound_idx"][r_idx] for ep_k, r_idx in task_train_rows.items()])
        phys_stats = self.compute_phys_stats(all_tr_cidx)

        # Target endpoint: fit strictly fold-internal ConditionEncoder; Aux: reuse fast CV/non-Case17 pre-encoded tensor!
        cond_encoders: Dict[str, DeepConditionEncoder] = {}
        y_scalers: Dict[str, Tuple[float, float]] = {}  # (mean, std) for regression; (0, 1) for cls
        task_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
        cond_dims: Dict[str, int] = {}

        for ep_k in active_eps:
            d = self.endpoints[ep_k]
            r_idx = task_train_rows.get(ep_k, np.array([], dtype=np.int64))
            if len(r_idx) == 0:
                continue
            if ep_k == target_ep:
                enc = DeepConditionEncoder(
                    d["cond_cols"],
                    min_count=3,
                    add_pairs=(ep_k in add_cond_pairs_for),
                ).fit([d["cond_rows"][i] for i in r_idx])
                cond_encoders[ep_k] = enc
                cond_dims[ep_k] = len(enc.feature_names_)
                c_ten = torch.from_numpy(enc.transform([d["cond_rows"][i] for i in r_idx])).to(self.device)
            else:
                enc = d["aux_cond_enc_c17"] if leave_case17_out else d["aux_cond_enc_cv"]
                cond_encoders[ep_k] = enc
                cond_dims[ep_k] = len(enc.feature_names_)
                src_mat = d["aux_cond_mat_all_c17fit"] if leave_case17_out else d["aux_cond_mat_all_cvfit"]
                c_ten = src_mat[torch.from_numpy(r_idx).to(self.device)]

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

        # Target val & holdout tensors
        t_enc = cond_encoders[target_ep]
        va_cond = torch.from_numpy(t_enc.transform([t_data["cond_rows"][i] for i in va_rows])).to(self.device)
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
        ho_pack = None
        if len(ho_rows) > 0:
            ho_cond = torch.from_numpy(t_enc.transform([t_data["cond_rows"][i] for i in ho_rows])).to(self.device)
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
                "n_train_clusters_all_tasks": len(all_train_clusters),
                "n_forbidden_clusters": len(forbidden_clusters),
                "n_leaked_clusters": 0,
                "n_target_train_rows": int(len(tr_rows)),
                "n_target_val_rows": int(len(va_rows)),
                "n_aux_rows_kept": n_aux_rows_kept,
                "n_aux_rows_dropped_for_leak": n_aux_rows_dropped_for_leak,
            },
        }


def compute_oof_tanimoto(bank: MultimodalGPUBank, ep_data: Dict[str, Any]) -> np.ndarray:
    """Fold-internal OOF max Tanimoto matching v33 CPU suite."""
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
