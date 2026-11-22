#!/usr/bin/env python3
"""Frozen Feature Bank loader, leak-free fold-internal preprocessors, condition encoder,
and packed-bit Tanimoto Applicability Domain index for V33 CPU modeling.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD

from v33_config import DATA_DIR, FEATURES_DIR

csv.field_size_limit(10_000_000)


class FeatureBank:
    """Read-only frozen V33 feature bank shared across forked workers (Copy-on-Write)."""

    def __init__(self, features_dir: Path = FEATURES_DIR, data_dir: Path = DATA_DIR):
        self.features_dir = Path(features_dir)
        self.data_dir = Path(data_dir)

        manifest = json.loads((self.features_dir / "feature_manifest.json").read_text(encoding="utf-8"))
        self.manifest = manifest
        self.case17_indices: Dict[str, int] = manifest["case17_indices"]
        self.physchem_names: List[str] = manifest["physchem_core"]["feature_names"]

        # Map compound_key -> integer row index 0..46597
        self.key_to_idx: Dict[str, int] = {}
        self.idx_to_key: List[str] = []
        with (self.features_dir / "compound_index.tsv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                idx = int(r["compound_idx"])
                ck = r["compound_key"]
                self.key_to_idx[ck] = idx
                self.idx_to_key.append(ck)

        # Load frozen numpy arrays cleanly into contiguous memory
        pc = np.load(self.features_dir / "physchem_core.npz")
        self.physchem: np.ndarray = np.ascontiguousarray(pc["X"], dtype=np.float32)
        self.smiles_valid: np.ndarray = np.ascontiguousarray(pc["smiles_valid_mask"], dtype=bool)
        self.seq_valid: np.ndarray = np.ascontiguousarray(pc["seq_valid_mask"], dtype=bool)
        pc.close()

        fp = np.load(self.features_dir / "fp_compact.npz")
        self.morgan_count: np.ndarray = np.ascontiguousarray(fp["morgan_count_r2_1024"], dtype=np.uint16)
        self.atompair: np.ndarray = np.ascontiguousarray(fp["atompair_1024"], dtype=np.uint16)
        self.maccs: np.ndarray = np.ascontiguousarray(fp["maccs_167"], dtype=np.uint8)
        self.torsion: np.ndarray = np.ascontiguousarray(fp["torsion_512"], dtype=np.uint16)
        fp.close()

        mb = np.load(self.features_dir / "morgan_bits_1024.npz")
        self.morgan_bits: np.ndarray = np.ascontiguousarray(mb["morgan_bits_1024"], dtype=np.uint8)
        self.morgan_packed: np.ndarray = np.ascontiguousarray(mb["morgan_bits_packed"], dtype=np.uint8)
        mb.close()

        # Load Case17 query metadata (SMILES, drug_name, reference strings)
        self.case17_queries: List[Dict[str, str]] = []
        with (self.data_dir / "case17_queries.tsv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                r["compound_idx"] = self.case17_indices[r["drug_name"]]
                self.case17_queries.append(r)

        assert self.physchem.shape == (46598, 83)
        assert self.morgan_bits.shape == (46598, 1024)
        assert self.morgan_packed.shape == (46598, 128)
        assert len(self.case17_queries) == 17

    def map_keys(self, compound_keys: List[str]) -> np.ndarray:
        return np.array([self.key_to_idx[k] for k in compound_keys], dtype=np.int32)

    def get_fp_block(self, idx: np.ndarray, kind: str) -> np.ndarray:
        if kind == "morgan":
            return np.log1p(self.morgan_count[idx].astype(np.float32))
        if kind == "svd_src":
            return np.log1p(
                np.concatenate(
                    [
                        self.morgan_count[idx].astype(np.float32),
                        self.atompair[idx].astype(np.float32),
                        self.maccs[idx].astype(np.float32),
                    ],
                    axis=1,
                )
            )
        if kind == "fp_all":
            return np.log1p(
                np.concatenate(
                    [
                        self.morgan_count[idx].astype(np.float32),
                        self.atompair[idx].astype(np.float32),
                        self.maccs[idx].astype(np.float32),
                        self.torsion[idx].astype(np.float32),
                    ],
                    axis=1,
                )
            )
        raise ValueError(f"Unknown fp block kind: {kind}")


class ConditionEncoder:
    """Fold-internal categorical encoder for `cond_*` columns.

    - Empty/missing values -> 'unspecified'
    - Training categories with count < min_count (default 3) merged into 'other'
    - Constant columns (<=1 distinct category after rare-collapsing) emit 0 features
    - Unseen categories at transform time map to 'other' if present, else all-zeros
    """

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


class FoldPreprocessor:
    """Strictly fold-internal feature builder and preprocessor.

    Tracks:
      - 'smiles_only': physchem[:, :48] + FPs + conds (+ optional meta)
      - 'dual_modality': physchem[:, :83] (valid-masked median impute) + [has_smiles, has_seq] + FPs + conds (+ optional meta)
      - 'seq_only': physchem[:, 48:83] + conds

    Routes:
      - 'physchem_core'
      - 'physchem_svd'       (32D fold-internal TruncatedSVD on log1p(morgan+atompair+maccs))
      - 'physchem_morgan'    (fold-internal support-filtered log1p(morgan_count_r2_1024))
      - 'physchem_fp_all'    (fold-internal support-filtered log1p(morgan+atompair+maccs+torsion))
      - 'tanimoto_hybrid'    (dict with scaled physchem+meta, binary morgan bits, and one-hot conds)
    """

    def __init__(
        self,
        track: str,
        route: str,
        cond_cols: List[str],
        scale_for_linear: bool = True,
        svd_dim: int = 32,
        min_fp_support_frac: float = 0.01,
        random_state: int = 42,
    ):
        assert track in ("smiles_only", "dual_modality", "seq_only")
        assert route in ("physchem_core", "physchem_svd", "physchem_morgan", "physchem_fp_all", "tanimoto_hybrid")
        self.track = track
        self.route = route
        self.cond_cols = list(cond_cols)
        self.scale_for_linear = scale_for_linear
        self.svd_dim = svd_dim
        self.min_fp_support_frac = min_fp_support_frac
        self.random_state = random_state

        self.cond_enc_: Optional[ConditionEncoder] = None
        self.smi_medians_: Optional[np.ndarray] = None
        self.seq_medians_: Optional[np.ndarray] = None
        self.meta_medians_: Optional[np.ndarray] = None
        self.phys_mean_: Optional[np.ndarray] = None
        self.phys_std_: Optional[np.ndarray] = None
        self.fp_keep_idx_: Optional[np.ndarray] = None
        self.fp_mean_: Optional[np.ndarray] = None
        self.fp_std_: Optional[np.ndarray] = None
        self.svd_: Optional[TruncatedSVD] = None
        self.svd_mean_: Optional[np.ndarray] = None
        self.svd_std_: Optional[np.ndarray] = None
        self.n_cond_active_: int = 0
        self.has_meta_: bool = False

    def _extract_phys_raw(
        self, bank: FeatureBank, idx: np.ndarray, fit: bool
    ) -> np.ndarray:
        if self.track == "smiles_only":
            X = bank.physchem[idx, :48].copy()
            smi_ok = bank.smiles_valid[idx]
            if fit:
                if smi_ok.any():
                    self.smi_medians_ = np.median(X[smi_ok], axis=0).astype(np.float32)
                else:
                    self.smi_medians_ = np.zeros(48, dtype=np.float32)
            if (~smi_ok).any():
                X[~smi_ok] = self.smi_medians_
            return X

        if self.track == "seq_only":
            X = bank.physchem[idx, 48:83].copy()
            seq_ok = bank.seq_valid[idx]
            if fit:
                if seq_ok.any():
                    self.seq_medians_ = np.median(X[seq_ok], axis=0).astype(np.float32)
                else:
                    self.seq_medians_ = np.zeros(35, dtype=np.float32)
            if (~seq_ok).any():
                X[~seq_ok] = self.seq_medians_
            return X

        # dual_modality: 48 SMILES + 35 seq + 2 availability flags = 85D
        X_smi = bank.physchem[idx, :48].copy()
        X_seq = bank.physchem[idx, 48:83].copy()
        smi_ok = bank.smiles_valid[idx]
        seq_ok = bank.seq_valid[idx]
        if fit:
            self.smi_medians_ = (
                np.median(X_smi[smi_ok], axis=0).astype(np.float32)
                if smi_ok.any()
                else np.zeros(48, dtype=np.float32)
            )
            self.seq_medians_ = (
                np.median(X_seq[seq_ok], axis=0).astype(np.float32)
                if seq_ok.any()
                else np.zeros(35, dtype=np.float32)
            )
        if (~smi_ok).any():
            X_smi[~smi_ok] = self.smi_medians_
        if (~seq_ok).any():
            X_seq[~seq_ok] = self.seq_medians_
        flags = np.stack([smi_ok.astype(np.float32), seq_ok.astype(np.float32)], axis=1)
        return np.concatenate([X_smi, X_seq, flags], axis=1)

    def _clean_meta(self, meta: Optional[np.ndarray], fit: bool) -> Optional[np.ndarray]:
        if meta is None:
            return None
        M = np.asarray(meta, dtype=np.float32).copy()
        if M.ndim == 1:
            M = M[:, None]
        if fit:
            self.has_meta_ = True
            meds = np.zeros(M.shape[1], dtype=np.float32)
            for j in range(M.shape[1]):
                ok = np.isfinite(M[:, j])
                meds[j] = float(np.median(M[ok, j])) if ok.any() else 0.0
            self.meta_medians_ = meds
        bad = ~np.isfinite(M)
        if bad.any():
            for j in range(M.shape[1]):
                if bad[:, j].any():
                    M[bad[:, j], j] = self.meta_medians_[j]
        return M

    def fit_transform(
        self,
        bank: FeatureBank,
        idx: np.ndarray,
        cond_rows: List[Dict[str, str]],
        meta: Optional[np.ndarray] = None,
    ) -> Any:
        self.cond_enc_ = ConditionEncoder(self.cond_cols, min_count=3).fit(cond_rows)
        X_cond = self.cond_enc_.transform(cond_rows)
        self.n_cond_active_ = len(self.cond_enc_.active_cols_)

        X_phys = self._extract_phys_raw(bank, idx, fit=True)
        M = self._clean_meta(meta, fit=True)
        if M is not None:
            X_phys = np.concatenate([X_phys, M], axis=1)

        # Standardize physchem+meta block
        self.phys_mean_ = X_phys.mean(axis=0).astype(np.float32)
        sd = X_phys.std(axis=0).astype(np.float32)
        sd[sd < 1e-6] = 1.0
        self.phys_std_ = sd

        if self.route == "tanimoto_hybrid":
            X_phys_s = np.clip((X_phys - self.phys_mean_) / self.phys_std_, -8.0, 8.0).astype(np.float32)
            X_bits = bank.morgan_bits[idx].astype(np.float32)
            # Zero bits explicitly on smiles_valid==0 rows (already 0, kept safe)
            if (~bank.smiles_valid[idx]).any():
                X_bits[~bank.smiles_valid[idx]] = 0.0
            return {
                "X_phys": X_phys_s,
                "X_bits": X_bits,
                "X_cond": X_cond,
                "n_cond_cols": max(1, self.n_cond_active_),
            }

        X_p_out = (
            np.clip((X_phys - self.phys_mean_) / self.phys_std_, -10.0, 10.0).astype(np.float32)
            if self.scale_for_linear
            else X_phys
        )

        if self.route == "physchem_core":
            return np.concatenate([X_p_out, X_cond], axis=1)

        if self.route == "physchem_svd":
            raw_fp = bank.get_fp_block(idx, "svd_src")
            n_comp = max(2, min(self.svd_dim, raw_fp.shape[0] - 2, raw_fp.shape[1] - 1))
            self.svd_ = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
            X_svd = self.svd_.fit_transform(raw_fp).astype(np.float32)
            self.svd_mean_ = X_svd.mean(axis=0).astype(np.float32)
            svd_sd = X_svd.std(axis=0).astype(np.float32)
            svd_sd[svd_sd < 1e-6] = 1.0
            self.svd_std_ = svd_sd
            if self.scale_for_linear:
                X_svd = np.clip((X_svd - self.svd_mean_) / self.svd_std_, -10.0, 10.0).astype(np.float32)
            return np.concatenate([X_p_out, X_svd, X_cond], axis=1)

        if self.route in ("physchem_morgan", "physchem_fp_all"):
            kind = "morgan" if self.route == "physchem_morgan" else "fp_all"
            raw_fp = bank.get_fp_block(idx, kind)
            n_tr = raw_fp.shape[0]
            min_sup = max(3, int(self.min_fp_support_frac * n_tr))
            nnz = (raw_fp > 0).sum(axis=0)
            var = raw_fp.var(axis=0)
            keep = np.where((nnz >= min_sup) & (var > 1e-8))[0]
            if len(keep) == 0:
                keep = np.where(var > 0)[0]
            self.fp_keep_idx_ = keep
            X_fp = raw_fp[:, keep]
            if self.scale_for_linear:
                self.fp_mean_ = X_fp.mean(axis=0).astype(np.float32)
                fp_sd = X_fp.std(axis=0).astype(np.float32)
                fp_sd[fp_sd < 1e-6] = 1.0
                self.fp_std_ = fp_sd
                X_fp = np.clip((X_fp - self.fp_mean_) / self.fp_std_, -10.0, 10.0).astype(np.float32)
            return np.concatenate([X_p_out, X_fp, X_cond], axis=1)

        raise ValueError(f"Unsupported route: {self.route}")

    def transform(
        self,
        bank: FeatureBank,
        idx: np.ndarray,
        cond_rows: Optional[List[Dict[str, str]]] = None,
        default_cond: Optional[Dict[str, str]] = None,
        meta: Optional[np.ndarray] = None,
    ) -> Any:
        assert self.cond_enc_ is not None
        if cond_rows is not None:
            X_cond = self.cond_enc_.transform(cond_rows)
        elif default_cond is not None:
            X_cond = self.cond_enc_.transform_constant(len(idx), default_cond)
        else:
            X_cond = self.cond_enc_.transform_constant(len(idx), {})

        X_phys = self._extract_phys_raw(bank, idx, fit=False)
        if self.has_meta_:
            assert meta is not None, "Preprocessor fitted with meta-features requires meta at transform time"
            M = self._clean_meta(meta, fit=False)
            X_phys = np.concatenate([X_phys, M], axis=1)

        if self.route == "tanimoto_hybrid":
            X_phys_s = np.clip((X_phys - self.phys_mean_) / self.phys_std_, -8.0, 8.0).astype(np.float32)
            X_bits = bank.morgan_bits[idx].astype(np.float32)
            if (~bank.smiles_valid[idx]).any():
                X_bits[~bank.smiles_valid[idx]] = 0.0
            return {
                "X_phys": X_phys_s,
                "X_bits": X_bits,
                "X_cond": X_cond,
                "n_cond_cols": max(1, self.n_cond_active_),
            }

        X_p_out = (
            np.clip((X_phys - self.phys_mean_) / self.phys_std_, -10.0, 10.0).astype(np.float32)
            if self.scale_for_linear
            else X_phys
        )

        if self.route == "physchem_core":
            return np.concatenate([X_p_out, X_cond], axis=1)

        if self.route == "physchem_svd":
            raw_fp = bank.get_fp_block(idx, "svd_src")
            X_svd = self.svd_.transform(raw_fp).astype(np.float32)
            if self.scale_for_linear:
                X_svd = np.clip((X_svd - self.svd_mean_) / self.svd_std_, -10.0, 10.0).astype(np.float32)
            return np.concatenate([X_p_out, X_svd, X_cond], axis=1)

        if self.route in ("physchem_morgan", "physchem_fp_all"):
            kind = "morgan" if self.route == "physchem_morgan" else "fp_all"
            raw_fp = bank.get_fp_block(idx, kind)
            X_fp = raw_fp[:, self.fp_keep_idx_]
            if self.scale_for_linear:
                X_fp = np.clip((X_fp - self.fp_mean_) / self.fp_std_, -10.0, 10.0).astype(np.float32)
            return np.concatenate([X_p_out, X_fp, X_cond], axis=1)

        raise ValueError(f"Unsupported route: {self.route}")


def fast_tanimoto_max_and_top5(
    query_packed: np.ndarray,
    ref_packed: np.ndarray,
    chunk_size: int = 512,
    exclude_self: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact Tanimoto max & top-5 mean similarity from packed (N, 128) uint8 Morgan bits.

    Unpacks to float32 inside bounded chunks for fast BLAS dot products.
    """
    n_q = query_packed.shape[0]
    n_r = ref_packed.shape[0]
    if n_q == 0 or n_r == 0:
        return np.zeros(n_q, dtype=np.float32), np.zeros(n_q, dtype=np.float32)

    ref_bits = np.unpackbits(ref_packed, axis=1).astype(np.float32)
    ref_pop = ref_bits.sum(axis=1)  # (n_r,)
    k_top = min(5, max(1, n_r - (1 if exclude_self else 0)))

    max_sim = np.zeros(n_q, dtype=np.float32)
    top5_sim = np.zeros(n_q, dtype=np.float32)

    for s in range(0, n_q, chunk_size):
        e = min(n_q, s + chunk_size)
        q_bits = np.unpackbits(query_packed[s:e], axis=1).astype(np.float32)
        q_pop = q_bits.sum(axis=1)  # (b,)
        inter = q_bits @ ref_bits.T  # (b, n_r)
        union = q_pop[:, None] + ref_pop[None, :] - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        if exclude_self and n_q == n_r:
            for local_i in range(e - s):
                sim[local_i, s + local_i] = -1.0
        # Zero-pop queries get 0.0
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

    ref_packed: np.ndarray  # (n_unique_ref, 128) uint8
    in_domain_thr: float = 0.50
    borderline_thr: float = 0.30
    tier_calibration: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_indices(cls, bank: FeatureBank, train_idx: np.ndarray) -> "TanimotoADIndex":
        uniq_idx = np.unique(train_idx)
        valid_uniq = uniq_idx[bank.smiles_valid[uniq_idx]]
        ref_packed = np.ascontiguousarray(bank.morgan_packed[valid_uniq], dtype=np.uint8)
        return cls(ref_packed=ref_packed)

    def query(
        self, bank: FeatureBank, query_idx: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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
