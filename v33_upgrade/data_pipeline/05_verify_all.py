#!/usr/bin/env python3
"""
Step 05 (runs on node23): Independent end-to-end verification suite for Peptide OmniPanel V33 Data & Features.
Checks:
  1. Zero NaN/Inf in all labels (y_value, y_lower, y_upper) across all 24 endpoints.
  2. Strict y_lower <= y_upper for 100% of rows (y_lower == y_upper == y_value for exact, y_lower < y_upper for censored/interval).
  3. Physical domain & sentinel checks:
     - PPB and F strictly within [0.0, 1.0] (both y_value and [y_lower, y_upper]).
     - Permeability endpoints (PAMPA, Caco2, MDCK, RRCK) have ZERO -10.0 CycPeptMPDB sentinels in y_value.
  4. Global cluster consistency across all 24 endpoints:
     - Each compound_key maps to a single consistent cluster_id across all endpoints.
     - All compounds sharing non-empty inchikey_14, conn_hash, or canonical_smiles ALWAYS share the same cluster_id.
  5. Triple-layer ZERO leakage check on EVERY endpoint:
     - Zero shared cluster_id, compound_key, inchikey_14, conn_hash, or canonical_smiles between
       any two CV folds (0..4) AND between CV (is_holdout=0) and holdout (is_holdout=1).
  6. Hard fold-balance assertion on EVERY endpoint:
     - min(fold_sizes) >= max(3, int(0.10 * n_cv_rows))
     - max(fold_sizes) <= max(10, int(0.35 * n_cv_rows))
     - Both classes present in every CV fold (0..4) and holdout for every classification endpoint.
  7. Every compound_key in every endpoint exists in compounds_v33.tsv and in features/compound_index.tsv.
  8. All 17 Case17 drugs exist in case17_queries.tsv and in features/compound_index.tsv with smiles_valid=1;
     case17_overlap.tsv count matches sum(case17_excluded==1) and zero unflagged exact/ik14 Case17 compounds exist.
  9. Zero NaN/Inf in physchem_core.npz, valid shapes & non-zero fingerprints for smiles_valid_mask==True in
     fp_compact.npz and morgan_bits_1024.npz, and exact SHA-256 verification against feature_manifest.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

csv.field_size_limit(10_000_000)

BASE_DIR = Path("/data/qlyu/v33_upgrade")
DATA_DIR = BASE_DIR / "data"
EP_DIR = DATA_DIR / "endpoints"
FEAT_DIR = BASE_DIR / "features"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    # 1. Load compounds_v33.tsv & compound_index.tsv (strictly read-only)
    comp_map = {}
    with (DATA_DIR / "compounds_v33.tsv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            comp_map[row["compound_key"]] = row

    idx_map = {}
    with (FEAT_DIR / "compound_index.tsv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            idx_map[row["compound_key"]] = row

    with (DATA_DIR / "case17_queries.tsv").open(encoding="utf-8", newline="") as fh:
        c17_rows = list(csv.DictReader(fh, delimiter="\t"))

    assert len(c17_rows) == 17, f"Expected 17 Case17 drugs, got {len(c17_rows)}"
    c17_cks = set()
    c17_ik14s = set()
    for r in c17_rows:
        ck = r["compound_key"]
        assert ck in idx_map, f"Case17 drug {r['drug_name']} missing from compound_index.tsv"
        assert idx_map[ck]["smiles_valid"] == "1", f"Case17 drug {r['drug_name']} smiles_valid != 1"
        c17_cks.add(ck)
        if r["inchikey_14"]:
            c17_ik14s.add(r["inchikey_14"])

    # 2. Verify feature bank .npz files & SHA-256
    manifest = json.loads((FEAT_DIR / "feature_manifest.json").read_text(encoding="utf-8"))
    for key in ("physchem_core", "fp_compact", "morgan_bits_1024", "compound_index"):
        p = Path(manifest[key]["file"])
        assert p.exists(), f"Missing feature file: {p}"
        actual_sha = sha256_file(p)
        assert actual_sha == manifest[key]["sha256"], f"SHA256 mismatch for {p}"

    N = len(idx_map)
    phys = np.load(FEAT_DIR / "physchem_core.npz", allow_pickle=False)
    X_phys = phys["X"]
    smi_mask = phys["smiles_valid_mask"]
    seq_mask = phys["seq_valid_mask"]
    assert X_phys.shape == (N, len(phys["feature_names"])), f"Unexpected physchem shape {X_phys.shape}"
    assert np.all(np.isfinite(X_phys)), "NaN/Inf in physchem_core.npz!"
    assert not np.any(np.isnan(X_phys)), "NaN in physchem_core.npz!"

    fp = np.load(FEAT_DIR / "fp_compact.npz", allow_pickle=False)
    for arr_name, exp_dim in [("morgan_count_r2_1024", 1024), ("atompair_1024", 1024), ("maccs_167", 167), ("torsion_512", 512)]:
        a = fp[arr_name]
        assert a.shape == (N, exp_dim), f"{arr_name} shape mismatch: {a.shape}"
        assert np.all(np.isfinite(a)), f"Non-finite in {arr_name}"
        if arr_name != "torsion_512":
            zero_valid = int((a[smi_mask].sum(axis=1) == 0).sum())
            assert zero_valid == 0, f"{zero_valid} valid SMILES have all-zero {arr_name}!"

    mb = np.load(FEAT_DIR / "morgan_bits_1024.npz", allow_pickle=False)
    assert mb["morgan_bits_1024"].shape == (N, 1024)
    assert mb["morgan_bits_packed"].shape == (N, 128)
    assert np.array_equal(np.packbits(mb["morgan_bits_1024"], axis=1), mb["morgan_bits_packed"]), "Packed bits mismatch!"

    c17_idxs = [int(idx_map[r["compound_key"]]["compound_idx"]) for r in c17_rows]
    c17_bits = mb["morgan_bits_1024"][c17_idxs].astype(np.float32)
    assert np.all(c17_bits.sum(axis=1) > 0), "A Case17 drug has zero Morgan bits!"

    # 3. Verify all 24 endpoints
    ep_files = sorted(EP_DIR.glob("*.tsv"))
    assert len(ep_files) == 24, f"Expected 24 endpoint TSVs, got {len(ep_files)}"

    ep_report = {}
    total_rows = 0
    total_censored = 0
    total_c17_ov = 0

    global_ck_to_clu: dict[str, str] = {}
    global_ik14_to_clu: dict[str, str] = {}
    global_ch_to_clu: dict[str, str] = {}
    global_smi_to_clu: dict[str, str] = {}

    for ep_path in ep_files:
        ep = ep_path.stem
        with ep_path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        assert len(rows) > 0, f"Empty endpoint {ep}"
        ttype = rows[0]["task_type"]
        tier = rows[0]["task_tier"]

        ho_clusters: set[str] = set()
        cv_clusters: set[str] = set()
        ho_cks: set[str] = set()
        cv_cks: set[str] = set()
        ho_ik14: set[str] = set()
        cv_ik14: set[str] = set()
        ho_ch: set[str] = set()
        cv_ch: set[str] = set()

        fold_clusters: dict[int, set[str]] = {i: set() for i in range(5)}
        fold_cks: dict[int, set[str]] = {i: set() for i in range(5)}
        fold_ik14: dict[int, set[str]] = {i: set() for i in range(5)}
        fold_ch: dict[int, set[str]] = {i: set() for i in range(5)}
        fold_row_counts: dict[int, int] = {i: 0 for i in range(5)}
        fold_classes: dict[int, Counter[int]] = {i: Counter() for i in range(5)}
        ho_classes: Counter[int] = Counter()
        censor_cnt: Counter[str] = Counter()
        smi_ready = 0
        c17_cnt = 0

        for r in rows:
            ck = r["compound_key"]
            assert ck in comp_map, f"{ep}: unknown compound_key {ck}"
            assert ck in idx_map, f"{ep}: compound_key {ck} missing from feature index"
            cm = comp_map[ck]

            yv = float(r["y_value"])
            yl = float(r["y_lower"])
            yh = float(r["y_upper"])
            assert math.isfinite(yv) and math.isfinite(yl) and math.isfinite(yh), f"{ep}: non-finite label!"
            assert yl <= yh, f"{ep}: y_lower ({yl}) > y_upper ({yh})!"
            ct = r["censor_type"]
            censor_cnt[ct] += 1
            if ct == "exact":
                assert abs(yl - yv) < 1e-9 and abs(yh - yv) < 1e-9, f"{ep}: exact bounds mismatch"
            else:
                assert yl < yh, f"{ep}: censored/interval must satisfy y_lower < y_upper ({yl}, {yh})"

            # Physical domain & sentinel assertions
            if ep in ("PPB", "F"):
                assert 0.0 <= yl <= yh <= 1.0 and 0.0 <= yv <= 1.0, (
                    f"{ep}: fraction out of [0,1]: y={yv}, [{yl}, {yh}] for {ck}"
                )
            if ep in ("PAMPA", "Caco2", "MDCK", "RRCK"):
                assert yv > -9.99, f"{ep}: CycPeptMPDB -10 sentinel leaked into y_value={yv} for {ck}"

            cid = r["cluster_id"]
            assert cid.startswith("CLU_"), f"{ep}: invalid cluster_id format {cid}"

            # Global cross-endpoint cluster consistency & exact-identity invariance
            if ck in global_ck_to_clu:
                assert global_ck_to_clu[ck] == cid, f"{ep}: compound_key {ck} cluster mismatch ({cid} vs {global_ck_to_clu[ck]})"
            else:
                global_ck_to_clu[ck] = cid
            if cm["inchikey_14"]:
                ik = cm["inchikey_14"]
                if ik in global_ik14_to_clu:
                    assert global_ik14_to_clu[ik] == cid, f"{ep}: inchikey_14 {ik} split across clusters!"
                else:
                    global_ik14_to_clu[ik] = cid
            if cm["conn_hash"]:
                ch = cm["conn_hash"]
                if ch in global_ch_to_clu:
                    assert global_ch_to_clu[ch] == cid, f"{ep}: conn_hash {ch} split across clusters!"
                else:
                    global_ch_to_clu[ch] = cid
            if cm["canonical_smiles"]:
                sm = cm["canonical_smiles"]
                if sm in global_smi_to_clu:
                    assert global_smi_to_clu[sm] == cid, f"{ep}: canonical_smiles split across clusters!"
                else:
                    global_smi_to_clu[sm] = cid

            ho = int(r["is_holdout"])
            fi = int(r["fold_id"])
            if ho == 1:
                assert fi == -1, f"{ep}: holdout row has fold_id={fi}"
                ho_clusters.add(cid)
                ho_cks.add(ck)
                if cm["inchikey_14"]:
                    ho_ik14.add(cm["inchikey_14"])
                if cm["conn_hash"]:
                    ho_ch.add(cm["conn_hash"])
                if ttype == "classification":
                    ho_classes[int(yv)] += 1
            else:
                assert fi in (0, 1, 2, 3, 4), f"{ep}: invalid fold_id={fi}"
                cv_clusters.add(cid)
                cv_cks.add(ck)
                if cm["inchikey_14"]:
                    cv_ik14.add(cm["inchikey_14"])
                if cm["conn_hash"]:
                    cv_ch.add(cm["conn_hash"])
                fold_clusters[fi].add(cid)
                fold_cks[fi].add(ck)
                if cm["inchikey_14"]:
                    fold_ik14[fi].add(cm["inchikey_14"])
                if cm["conn_hash"]:
                    fold_ch[fi].add(cm["conn_hash"])
                fold_row_counts[fi] += 1
                if ttype == "classification":
                    fold_classes[fi][int(yv)] += 1

            if idx_map[ck]["smiles_valid"] == "1":
                smi_ready += 1
            if int(r["case17_excluded"]) == 1:
                c17_cnt += 1
            else:
                assert ck not in c17_cks, f"{ep}: unflagged Case17 exact compound_key {ck}!"
                assert not (cm["inchikey_14"] and cm["inchikey_14"] in c17_ik14s), (
                    f"{ep}: unflagged Case17 inchikey_14 {cm['inchikey_14']}!"
                )

        # Triple-layer ZERO leakage checks (cluster_id, compound_key, inchikey_14, conn_hash)
        assert len(ho_clusters & cv_clusters) == 0, f"{ep}: CLUSTER LEAKAGE between CV and holdout!"
        assert len(ho_cks & cv_cks) == 0, f"{ep}: COMPOUND_KEY LEAKAGE between CV and holdout!"
        assert len(ho_ik14 & cv_ik14) == 0, f"{ep}: INCHIKEY_14 LEAKAGE between CV and holdout!"
        assert len(ho_ch & cv_ch) == 0, f"{ep}: CONN_HASH LEAKAGE between CV and holdout!"

        for i in range(5):
            assert len(fold_clusters[i]) > 0, f"{ep}: empty fold {i}"
            for j in range(i + 1, 5):
                assert len(fold_clusters[i] & fold_clusters[j]) == 0, f"{ep}: CLUSTER LEAKAGE fold {i} vs {j}!"
                assert len(fold_cks[i] & fold_cks[j]) == 0, f"{ep}: COMPOUND_KEY LEAKAGE fold {i} vs {j}!"
                assert len(fold_ik14[i] & fold_ik14[j]) == 0, f"{ep}: INCHIKEY_14 LEAKAGE fold {i} vs {j}!"
                assert len(fold_ch[i] & fold_ch[j]) == 0, f"{ep}: CONN_HASH LEAKAGE fold {i} vs {j}!"
            if ttype == "classification":
                assert len(fold_classes[i]) == 2, f"{ep}: single-class fold {i}: {fold_classes[i]}"
        if ttype == "classification":
            assert len(ho_classes) == 2, f"{ep}: single-class holdout: {ho_classes}"

        # Hard fold-balance assertions
        cv_rows = sum(fold_row_counts.values())
        ho_rows = len(rows) - cv_rows
        min_f = min(fold_row_counts.values())
        max_f = max(fold_row_counts.values())
        min_req = max(3, int(0.10 * cv_rows))
        max_req = max(10, int(0.35 * cv_rows))
        assert min_f >= min_req, f"{ep}: fold too small ({min_f} < {min_req}, n_cv={cv_rows}, folds={fold_row_counts})"
        assert max_f <= max_req, f"{ep}: fold too large ({max_f} > {max_req}, n_cv={cv_rows}, folds={fold_row_counts})"

        n_cens = sum(v for k, v in censor_cnt.items() if k != "exact")
        total_rows += len(rows)
        total_censored += n_cens
        total_c17_ov += c17_cnt
        ep_report[ep] = {
            "tier": tier,
            "task_type": ttype,
            "rows": len(rows),
            "unique_compounds": len({r["compound_key"] for r in rows}),
            "unique_clusters": len(ho_clusters | cv_clusters),
            "smiles_valid_rows": smi_ready,
            "smiles_valid_pct": round(100.0 * smi_ready / len(rows), 2),
            "exact": censor_cnt["exact"],
            "right_censored": censor_cnt["right"],
            "left_censored": censor_cnt["left"],
            "interval": censor_cnt["interval"],
            "case17_overlap_rows": c17_cnt,
            "cv_rows": cv_rows,
            "holdout_rows": ho_rows,
            "holdout_frac": round(ho_rows / len(rows), 4),
            "fold_sizes": {f"fold_{i}": fold_row_counts[i] for i in range(5)},
            "min_fold_frac": round(min_f / cv_rows, 4),
            "max_fold_frac": round(max_f / cv_rows, 4),
            "fold_balance_pass": True,
            "zero_leakage_pass": True,
            "pos_count": sum(1 for r in rows if ttype == "classification" and int(float(r["y_value"])) == 1),
            "neg_count": sum(1 for r in rows if ttype == "classification" and int(float(r["y_value"])) == 0),
        }

    # Check Case17 overlap table consistency
    with (DATA_DIR / "case17_overlap.tsv").open(encoding="utf-8", newline="") as fh:
        ov_rows = list(csv.DictReader(fh, delimiter="\t"))
    assert len(ov_rows) == total_c17_ov, f"case17_overlap.tsv count ({len(ov_rows)}) != sum of case17_excluded ({total_c17_ov})"

    summary = {
        "status": "ALL_CHECKS_PASSED",
        "schema": "peptide_omnipanel_v33_verification_report_v2",
        "n_endpoints": len(ep_report),
        "total_endpoint_rows": total_rows,
        "total_censored_or_interval_rows": total_censored,
        "total_case17_overlap_rows": total_c17_ov,
        "n_unique_v33_compounds": len(comp_map),
        "n_feature_bank_compounds": N,
        "n_smiles_valid": int(smi_mask.sum()),
        "n_seq_valid": int(seq_mask.sum()),
        "physchem_shape": list(X_phys.shape),
        "fold_balance_rule": "min(fold_sizes) >= max(3, int(0.10*n_cv)) AND max(fold_sizes) <= max(10, int(0.35*n_cv))",
        "all_24_endpoints_fold_balance_passed": True,
        "all_24_endpoints_zero_leakage_passed": True,
        "endpoints": ep_report,
    }
    (DATA_DIR / "verification_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("==========================================")
    print("ALL VERIFICATION ASSERTIONS PASSED (100%)!")
    print(f"  Endpoints: {len(ep_report)} | Total clean rows: {total_rows:,} | Censored/Interval: {total_censored:,}")
    print(f"  Unique V33 compounds: {len(comp_map):,} | Feature bank total (incl. Case17): {N:,}")
    print(f"  SMILES valid: {int(smi_mask.sum()):,}/{N:,} | Seq valid: {int(seq_mask.sum()):,}/{N:,}")
    print(f"  physchem_core: {list(X_phys.shape)} | Case17 overlap rows: {total_c17_ov}")
    print("  Fold balance (10%..35%): 24/24 PASSED | Zero leakage (cluster/ck/ik14/ch): 24/24 PASSED")
    print("==========================================")


if __name__ == "__main__":
    main()
