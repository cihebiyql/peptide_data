#!/usr/bin/env python3
"""Read-only shared FusionBank loading frozen V33 features, all 4 verified GPU foundation
model embedding caches (float16 CoW-shared across forked workers), Wave-2C CPU predictions,
ConditionEncoder, and packed-bit Tanimoto Applicability Domain index.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fusion_config import CPU_MODELS_DIR, DATA_DIR, EMBEDDINGS_DIR, ENDPOINTS_DIR, FEATURES_DIR

csv.field_size_limit(10_000_000)


class FusionBank:
    """Read-only shared feature + pretrained embedding bank (~950 MB in float16/float32, CoW-safe)."""

    def __init__(
        self,
        features_dir: Path = FEATURES_DIR,
        embeddings_dir: Path = EMBEDDINGS_DIR,
        data_dir: Path = DATA_DIR,
        cpu_models_dir: Path = CPU_MODELS_DIR,
    ):
        self.features_dir = Path(features_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.data_dir = Path(data_dir)
        self.cpu_models_dir = Path(cpu_models_dir)

        manifest = json.loads((self.features_dir / "feature_manifest.json").read_text(encoding="utf-8"))
        self.manifest = manifest
        self.case17_indices: Dict[str, int] = manifest["case17_indices"]
        self.physchem_names: List[str] = manifest["physchem_core"]["feature_names"]

        self.key_to_idx: Dict[str, int] = {}
        self.idx_to_key: List[str] = []
        with (self.features_dir / "compound_index.tsv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                idx = int(r["compound_idx"])
                ck = r["compound_key"]
                self.key_to_idx[ck] = idx
                self.idx_to_key.append(ck)

        # Core physchem + packed Morgan bits for Tanimoto AD
        pc = np.load(self.features_dir / "physchem_core.npz")
        self.physchem: np.ndarray = np.ascontiguousarray(pc["X"], dtype=np.float32)
        self.smiles_valid: np.ndarray = np.ascontiguousarray(pc["smiles_valid_mask"], dtype=bool)
        self.seq_valid: np.ndarray = np.ascontiguousarray(pc["seq_valid_mask"], dtype=bool)
        pc.close()

        mb = np.load(self.features_dir / "morgan_bits_1024.npz")
        self.morgan_packed: np.ndarray = np.ascontiguousarray(mb["morgan_bits_packed"], dtype=np.uint8)
        mb.close()

        # Load all 4 verified GPU embedding caches in compact float16
        self.emb_blocks: Dict[str, np.ndarray] = {}
        self.emb_valid: Dict[str, np.ndarray] = {}

        d_ncaa = np.load(self.embeddings_dir / "ncaa_repdistill_v2.npz")
        v_ncaa = np.ascontiguousarray(d_ncaa["valid"], dtype=bool)
        assert np.array_equal(v_ncaa, self.smiles_valid), "ncaa_repdistill_v2 valid mask mismatch!"
        for k, bk in [
            ("smiles_repr", "ncaa_smiles_repr"),
            ("fusion_hf", "ncaa_fusion_hf"),
            ("protein_proj", "ncaa_protein_proj"),
        ]:
            arr = np.ascontiguousarray(d_ncaa[k], dtype=np.float16)
            assert arr.shape[0] == 46598 and np.isfinite(arr[v_ncaa]).all()
            self.emb_blocks[bk] = arr
            self.emb_valid[bk] = v_ncaa
        d_ncaa.close()

        d_pclm = np.load(self.embeddings_dir / "peptideclm2.npz")
        v_pclm = np.ascontiguousarray(d_pclm["valid"], dtype=bool)
        assert np.array_equal(v_pclm, self.smiles_valid), "peptideclm2 valid mask mismatch!"
        self.emb_blocks["peptideclm2"] = np.ascontiguousarray(d_pclm["embeddings"], dtype=np.float16)
        self.emb_valid["peptideclm2"] = v_pclm
        d_pclm.close()

        d_cb = np.load(self.embeddings_dir / "chemberta_mtr.npz")
        v_cb = np.ascontiguousarray(d_cb["valid"], dtype=bool)
        assert np.array_equal(v_cb, self.smiles_valid), "chemberta_mtr valid mask mismatch!"
        self.emb_blocks["chemberta_mtr"] = np.ascontiguousarray(d_cb["embeddings"], dtype=np.float16)
        self.emb_valid["chemberta_mtr"] = v_cb
        d_cb.close()

        d_esmc = np.load(self.embeddings_dir / "esmc_600m_multiregion.npz")
        v_esmc = np.ascontiguousarray(d_esmc["valid"], dtype=bool)
        assert np.array_equal(v_esmc, self.seq_valid), "esmc_600m_multiregion valid mask mismatch!"
        for k, bk in [
            ("mean", "esmc_mean"),
            ("max", "esmc_max"),
            ("nterm5", "esmc_nterm5"),
            ("cterm5", "esmc_cterm5"),
        ]:
            arr = np.ascontiguousarray(d_esmc[k], dtype=np.float16)
            assert arr.shape[0] == 46598 and np.isfinite(arr[v_esmc]).all()
            self.emb_blocks[bk] = arr
            self.emb_valid[bk] = v_esmc
        d_esmc.close()

        # Load Case17 query metadata
        self.case17_queries: List[Dict[str, Any]] = []
        with (self.data_dir / "case17_queries.tsv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                r["compound_idx"] = self.case17_indices[r["drug_name"]]
                self.case17_queries.append(r)

        assert self.physchem.shape == (46598, 83)
        assert self.morgan_packed.shape == (46598, 128)
        assert len(self.case17_queries) == 17

    def map_keys(self, compound_keys: List[str]) -> np.ndarray:
        return np.array([self.key_to_idx[k] for k in compound_keys], dtype=np.int32)

    def get_emb_f32(self, block_name: str, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.emb_blocks[block_name][idx].astype(np.float32),
            self.emb_valid[block_name][idx].copy(),
        )

    def load_endpoint_and_cpu(self, endpoint: str) -> Dict[str, Any]:
        """Load frozen endpoint TSV + strictly row-aligned Wave-2C CPU OOF/HO/Case17 predictions."""
        path = ENDPOINTS_DIR / f"{endpoint}.tsv"
        with path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            cols = list(reader.fieldnames or [])
            rows = list(reader)

        cond_cols = [c for c in cols if c.startswith("cond_")]
        n = len(rows)
        record_ids = [r["record_id"] for r in rows]
        compound_keys = [r["compound_key"] for r in rows]
        compound_idx = self.map_keys(compound_keys)
        cluster_ids = np.array([r["cluster_id"] for r in rows], dtype=object)
        is_holdout = np.array([int(r["is_holdout"]) for r in rows], dtype=np.int32)
        fold_id = np.array([int(r["fold_id"]) for r in rows], dtype=np.int32)
        case17_excluded = np.array([int(r["case17_excluded"]) for r in rows], dtype=np.int32)
        censor_type = np.array([r["censor_type"] for r in rows], dtype=object)
        y_value = np.array([float(r["y_value"]) for r in rows], dtype=np.float64)
        y_lower = np.array([float(r["y_lower"]) for r in rows], dtype=np.float64)
        y_upper = np.array([float(r["y_upper"]) for r in rows], dtype=np.float64)
        cond_rows = [{c: r.get(c, "unspecified") for c in cond_cols} for r in rows]

        cv_idx = np.where(is_holdout == 0)[0]
        ho_idx = np.where(is_holdout == 1)[0]
        assert len(cv_idx) + len(ho_idx) == n
        assert set(cluster_ids[cv_idx]).isdisjoint(set(cluster_ids[ho_idx])), f"{endpoint}: cluster leakage!"

        # Load strictly aligned Wave-2C CPU predictions
        cpu_dir = self.cpu_models_dir / endpoint

        def _parse_col(tbl_rows: List[Dict[str, str]], col: str) -> Optional[np.ndarray]:
            if not tbl_rows or col not in tbl_rows[0]:
                return None
            out = np.full(len(tbl_rows), np.nan, dtype=np.float64)
            for i, r in enumerate(tbl_rows):
                v = r.get(col, "")
                if v != "":
                    out[i] = float(v)
            return out

        with (cpu_dir / "oof_predictions.tsv").open(encoding="utf-8") as fh:
            cpu_oof_rows = list(csv.DictReader(fh, delimiter="\t"))
        assert len(cpu_oof_rows) == len(cv_idx), f"{endpoint}: CPU OOF count mismatch!"
        for loc_i, orig_i in enumerate(cv_idx):
            assert cpu_oof_rows[loc_i]["record_id"] == record_ids[orig_i]
            assert cpu_oof_rows[loc_i]["compound_key"] == compound_keys[orig_i]

        with (cpu_dir / "holdout_predictions.tsv").open(encoding="utf-8") as fh:
            cpu_ho_rows = list(csv.DictReader(fh, delimiter="\t"))
        assert len(cpu_ho_rows) == len(ho_idx), f"{endpoint}: CPU HO count mismatch!"
        for loc_i, orig_i in enumerate(ho_idx):
            assert cpu_ho_rows[loc_i]["record_id"] == record_ids[orig_i]
            assert cpu_ho_rows[loc_i]["compound_key"] == compound_keys[orig_i]

        with (cpu_dir / "case17_predictions.tsv").open(encoding="utf-8") as fh:
            cpu_c17_rows = list(csv.DictReader(fh, delimiter="\t"))
        assert len(cpu_c17_rows) == 17, f"{endpoint}: CPU Case17 count mismatch!"
        for i, q in enumerate(self.case17_queries):
            assert cpu_c17_rows[i]["drug_name"] == q["drug_name"]

        cpu_metrics = json.loads((cpu_dir / "metrics.json").read_text(encoding="utf-8"))

        cpu_preds: Dict[str, Any] = {
            "oof_smiles_only": _parse_col(cpu_oof_rows, "pred_smiles_only"),
            "oof_dual": _parse_col(cpu_oof_rows, "pred_dual"),
            "oof_aft": _parse_col(cpu_oof_rows, "pred_aft"),
            "oof_aft_smiles_only": _parse_col(cpu_oof_rows, "pred_aft_smiles_only"),
            "oof_seq_only": _parse_col(cpu_oof_rows, "pred_seq_only"),
            "oof_max_tanimoto": _parse_col(cpu_oof_rows, "max_tanimoto_oof"),
            "ho_smiles_only": _parse_col(cpu_ho_rows, "pred_smiles_only"),
            "ho_dual": _parse_col(cpu_ho_rows, "pred_dual"),
            "ho_aft": _parse_col(cpu_ho_rows, "pred_aft"),
            "ho_aft_smiles_only": _parse_col(cpu_ho_rows, "pred_aft_smiles_only"),
            "ho_seq_only": _parse_col(cpu_ho_rows, "pred_seq_only"),
            "c17_smiles_only": _parse_col(cpu_c17_rows, "pred_smiles_only"),
            "c17_dual": _parse_col(cpu_c17_rows, "pred_dual"),
            "c17_aft": _parse_col(cpu_c17_rows, "pred_aft"),
            "c17_seq_only": _parse_col(cpu_c17_rows, "pred_seq_only"),
            "c17_alt": {},
            "metrics": cpu_metrics,
        }
        # Capture any alternate clinical route columns from Wave-2C Case17 (e.g. _SC, _SC_CL_over_F)
        c17_headers = list(cpu_c17_rows[0].keys())
        for h in c17_headers:
            if h.startswith("pred_smiles_only_") and h != "pred_smiles_only_cal":
                alt_name = h[len("pred_smiles_only_") :]
                cpu_preds["c17_alt"][alt_name] = {
                    "smiles_only": _parse_col(cpu_c17_rows, f"pred_smiles_only_{alt_name}"),
                    "dual": _parse_col(cpu_c17_rows, f"pred_dual_{alt_name}"),
                    "aft": _parse_col(cpu_c17_rows, f"pred_aft_{alt_name}"),
                }

        return {
            "endpoint": endpoint,
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
            "y_value": y_value,
            "y_lower": y_lower,
            "y_upper": y_upper,
            "cond_rows": cond_rows,
            "cv_idx": cv_idx,
            "ho_idx": ho_idx,
            "cpu": cpu_preds,
        }


class ConditionEncoder:
    """Fold-internal categorical encoder for `cond_*` columns (identical to Wave-2C)."""

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
        if s == "" or s.lower() == "nan" or s.lower() == "none":
            return "unspecified"
        return s

    def fit(self, cond_rows: List[Dict[str, str]]) -> "ConditionEncoder":
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
            has_other = "other" in cat_to_j
            other_j = cat_to_j.get("other", -1)
            for i, r in enumerate(cond_rows):
                v = self._clean(r.get(col, "unspecified"))
                j = cat_to_j.get(v, other_j if has_other else -1)
                if j >= 0:
                    out[i, offset + j] = 1.0
            offset += len(cats)
        return out

    def transform_constant(self, n: int, default_cond: Dict[str, str]) -> np.ndarray:
        if len(self.feature_names_) == 0:
            return np.zeros((n, 0), dtype=np.float32)
        row = self.transform([default_cond])[0]
        return np.repeat(row[None, :], n, axis=0)


def fast_tanimoto_max_and_top5(
    query_packed: np.ndarray,
    ref_packed: np.ndarray,
    chunk_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    n_q = query_packed.shape[0]
    n_r = ref_packed.shape[0]
    if n_q == 0 or n_r == 0:
        return np.zeros(n_q, dtype=np.float32), np.zeros(n_q, dtype=np.float32)

    ref_bits = np.unpackbits(ref_packed, axis=1).astype(np.float32)
    ref_pop = ref_bits.sum(axis=1)
    k_top = min(5, max(1, n_r))

    max_sim = np.zeros(n_q, dtype=np.float32)
    top5_sim = np.zeros(n_q, dtype=np.float32)

    for s in range(0, n_q, chunk_size):
        e = min(n_q, s + chunk_size)
        q_bits = np.unpackbits(query_packed[s:e], axis=1).astype(np.float32)
        q_pop = q_bits.sum(axis=1)
        inter = q_bits @ ref_bits.T
        union = q_pop[:, None] + ref_pop[None, :] - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        zero_q = q_pop <= 0
        if zero_q.any():
            sim[zero_q] = 0.0
        max_sim[s:e] = np.maximum(sim.max(axis=1), 0.0)
        if k_top >= n_r:
            top5_sim[s:e] = np.maximum(sim, 0.0).mean(axis=1)
        else:
            part = np.partition(sim, n_r - k_top, axis=1)[:, n_r - k_top :]
            top5_sim[s:e] = np.maximum(part, 0.0).mean(axis=1)
    return max_sim, top5_sim


@dataclass
class TanimotoADIndex:
    """Compact serialized Applicability Domain index storing unique packed Morgan bits."""

    ref_packed: np.ndarray
    in_domain_thr: float = 0.50
    borderline_thr: float = 0.30
    tier_calibration: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_indices(cls, bank: FusionBank, train_idx: np.ndarray) -> "TanimotoADIndex":
        uniq_idx = np.unique(train_idx)
        valid_uniq = uniq_idx[bank.smiles_valid[uniq_idx]]
        ref_packed = np.ascontiguousarray(bank.morgan_packed[valid_uniq], dtype=np.uint8)
        return cls(ref_packed=ref_packed)

    def query(self, bank: FusionBank, query_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        q_packed = bank.morgan_packed[query_idx]
        max_t, top5_t = fast_tanimoto_max_and_top5(q_packed, self.ref_packed)
        smi_ok = bank.smiles_valid[query_idx]
        tiers: List[str] = []
        for i in range(len(query_idx)):
            if not smi_ok[i]:
                max_t[i] = np.nan
                top5_t[i] = np.nan
                tiers.append("SeqOnly")
            elif max_t[i] >= self.in_domain_thr:
                tiers.append("In-Domain")
            elif max_t[i] >= self.borderline_thr:
                tiers.append("Borderline")
            else:
                tiers.append("OOD")
        return max_t, top5_t, tiers
