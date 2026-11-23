#!/usr/bin/env python3
"""
Step 03 (runs on node23, 64 CPU cores):
- Builds ONE unified, deterministic, leakage-proof, NON-CHAINING compound-level cluster assignment (`cluster_id`):
  Stage (a): Exact `canonical_identity_group` (Union-Find ONLY on identical `inchikey_14`, `conn_hash`,
             or `canonical_smiles`). Guarantees stereoisomers/tautomers/duplicate keys of the same
             chemical connectivity ALWAYS stay in the same cluster.
  Stage (b): Non-chaining Dual-Track Leader Clustering (NO Murcko/generic ring collapse, NO unbounded
             transitive single-linkage chaining!):
             - Track S (Linear sequence-defined peptides, `is_cyclic == 0` with genuine canonical
               `parent_sequence` or missing SMILES): greedy length-sorted CD-HIT-style leader clustering
               at >= 70% global normalized edit identity (k-mer indexed + exact banded Levenshtein).
               Prevents the binary-Morgan bag-of-20AA collision that lumps unrelated canonical linear
               peptides together, while keeping homologous mutant/truncation series together.
             - Track M (Cyclic / macrocyclic / short / chemically modified SMILES peptides):
               Taylor-Butina degree-sorted leader clustering on `features/morgan_bits_1024.npz` at
               Tanimoto >= 0.70 (every member has Tanimoto >= 0.70 to its cluster centroid/leader).
- Strictly READ-ONLY on `compounds_v33.tsv`, `endpoints_raw_clean/`, and `features/*` (NEVER modifies them!).
- Freezes per-endpoint outer group holdout (~20% of rows, `is_holdout=1, fold_id=-1`, never splitting a cluster)
  and inner 5-fold grouped CV (`is_holdout=0, fold_id=0..4`) via stratified marginal-variance LPT
  size-balanced cluster bin-packing + local refinement.
- Applies exact physical domain clamping on `endpoints/PPB.tsv` and `endpoints/F.tsv` (`y_upper <= 1.0`,
  `y_lower >= 0.0`, preserving strict `y_lower < y_upper` for censored rows).
- Asserts ZERO leakage (`cluster_id`, `compound_key`, `inchikey_14`, `conn_hash`, `canonical_smiles`)
  and strict fold-size balance (`min >= max(3, int(0.10*n_cv))`, `max <= max(10, int(0.35*n_cv))`)
  across every single one of the 24 endpoints!
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

csv.field_size_limit(10_000_000)

BASE_DIR = Path("<WORK_DIR>/v33_upgrade")
DATA_DIR = BASE_DIR / "data"
FEAT_DIR = BASE_DIR / "features"
RAW_EP_DIR = DATA_DIR / "endpoints_raw_clean"
FINAL_EP_DIR = DATA_DIR / "endpoints"

TANIMOTO_THRESHOLD = 0.70
SEQ_IDENTITY_THRESHOLD = 0.70


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def levenshtein_bounded(s1: str, s2: str, max_dist: int) -> int:
    """Fast banded Levenshtein distance; returns max_dist + 1 if distance exceeds max_dist."""
    n, m = len(s1), len(s2)
    if abs(n - m) > max_dist:
        return max_dist + 1
    if s1 == s2:
        return 0
    if n > m:
        s1, s2 = s2, s1
        n, m = m, n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        c1 = s1[i - 1]
        lo = max(1, i - max_dist)
        hi = min(m, i + max_dist)
        curr = [max_dist + 1] * (m + 1)
        if lo == 1:
            curr[0] = i
        row_min = curr[0]
        for j in range(lo, hi + 1):
            cost = 0 if c1 == s2[j - 1] else 1
            v = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            curr[j] = v
            if v < row_min:
                row_min = v
        if row_min > max_dist:
            return max_dist + 1
        prev = curr
    return prev[m]


def cluster_sequences_leader(unique_seqs: list[str], thresh: float = SEQ_IDENTITY_THRESHOLD) -> dict[str, int]:
    """
    Non-chaining greedy length-sorted CD-HIT-style leader clustering at >= `thresh`
    global normalized edit identity (1 - edit_dist / len(leader)).
    - Sequences of length < 4: exact match only.
    - Sequences of length >= 4: k-mer inverted index (k=2 for len<10, k=3 for len>=10).
    """
    seq_to_cid: dict[str, int] = {}
    next_cid = 0

    short_seqs = [s for s in unique_seqs if len(s) < 4]
    for s in sorted(short_seqs):
        seq_to_cid[s] = next_cid
        next_cid += 1

    long_seqs = sorted([s for s in unique_seqs if len(s) >= 4], key=lambda s: (-len(s), s))
    reps: list[tuple[str, int]] = []
    kmer_index_2: dict[str, list[int]] = defaultdict(list)
    kmer_index_3: dict[str, list[int]] = defaultdict(list)

    for seq in long_seqs:
        L = len(seq)
        max_len_rep = int(math.floor(L / thresh))

        if L < 10:
            kmers = {seq[i : i + 2] for i in range(L - 1)}
            cand_counts: Counter[int] = Counter()
            for km in kmers:
                for rid in kmer_index_2.get(km, ()):
                    cand_counts[rid] += 1
            min_hits = 1
        else:
            kmers = {seq[i : i + 3] for i in range(L - 2)}
            cand_counts = Counter()
            for km in kmers:
                for rid in kmer_index_3.get(km, ()):
                    cand_counts[rid] += 1
            min_hits = 1 if L < 15 else 2

        best_cid = None
        best_sim = thresh - 1e-6
        for rid, hits in cand_counts.most_common(64):
            if hits < min_hits:
                break
            rseq, rcid = reps[rid]
            rL = len(rseq)
            if rL > max_len_rep:
                continue
            max_ed = int(math.floor((1.0 - thresh) * rL + 1e-9))
            ed = levenshtein_bounded(seq, rseq, max_ed)
            if ed <= max_ed:
                sim = 1.0 - (ed / rL)
                if sim >= thresh and sim > best_sim:
                    best_sim = sim
                    best_cid = rcid
                    if sim >= 0.95:
                        break

        if best_cid is not None:
            seq_to_cid[seq] = best_cid
        else:
            rid = len(reps)
            cid = next_cid
            next_cid += 1
            reps.append((seq, cid))
            seq_to_cid[seq] = cid
            if L < 10:
                for km in {seq[i : i + 2] for i in range(L - 1)}:
                    kmer_index_2[km].append(rid)
            else:
                for km in {seq[i : i + 3] for i in range(L - 2)}:
                    kmer_index_3[km].append(rid)
                if L <= 13:
                    for km in {seq[i : i + 2] for i in range(L - 1)}:
                        kmer_index_2[km].append(rid)

    return seq_to_cid


def cluster_morgan_butina(bits: np.ndarray, thresh: float = TANIMOTO_THRESHOLD, block_size: int = 2048) -> list[int]:
    """
    Exact non-chaining Taylor-Butina leader clustering on binary Morgan 1024-bit fingerprints.
    Every cluster member has Tanimoto >= `thresh` to its cluster centroid/leader.
    """
    n = bits.shape[0]
    if n == 0:
        return []
    B = bits.astype(np.float32, copy=False)
    P = B.sum(axis=1)
    nbrs: list[list[int]] = [[] for _ in range(n)]

    for i0 in range(0, n, block_size):
        i1 = min(n, i0 + block_size)
        dot = B[i0:i1] @ B.T
        denom = P[i0:i1, None] + P[None, :] - dot
        sim = np.where(denom > 0.0, dot / denom, 0.0)
        rs, cs = np.where(sim >= thresh)
        for r_loc, c_loc in zip(rs, cs):
            nbrs[i0 + int(r_loc)].append(int(c_loc))

    # Taylor-Butina: sort leaders descending by neighbor count (tie-break deterministically by index)
    order = sorted(range(n), key=lambda u: (-len(nbrs[u]), u))
    assigned = [-1] * n
    next_cid = 0
    for u in order:
        if assigned[u] != -1:
            continue
        cid = next_cid
        next_cid += 1
        for v in nbrs[u]:
            if assigned[v] == -1:
                assigned[v] = cid
    return assigned


def _det_tie(ep_name: str, cid: str) -> int:
    return int(hashlib.blake2b(f"{ep_name}|{cid}".encode("utf-8"), digest_size=8).hexdigest(), 16)


def _bin_pack_clusters(
    cluster_ids: list[str],
    cluster_sizes: dict[str, int],
    cluster_strata: dict[str, np.ndarray],
    n_bins: int,
    ep_name: str,
    is_classification: bool,
    max_bin_cap: int | None = None,
) -> dict[str, int]:
    """
    Stratified Longest-Processing-Time (LPT) size-balanced cluster bin-packing + local 1-move refinement.
    - Sorts clusters descending by row count (with deterministic tie-break).
    - Assigns each cluster to the eligible bin minimizing marginal global variance increase
      over both row count (dominant) and class/quantile strata counts.
    - Guarantees every bin receives >= 1 positive and >= 1 negative sample for classification endpoints.
    """
    n_strata = len(next(iter(cluster_strata.values())))
    total_rows = sum(cluster_sizes[c] for c in cluster_ids)
    total_strata = np.sum([cluster_strata[c] for c in cluster_ids], axis=0).astype(float)
    target_row = max(1.0, total_rows / n_bins)
    target_strata = np.maximum(1.0, total_strata / n_bins)

    bin_rows = np.zeros(n_bins, dtype=float)
    bin_strata = np.zeros((n_bins, n_strata), dtype=float)
    assignment: dict[str, int] = {}

    # Sort descending by cluster size, tie-breaking deterministically
    ordered = sorted(cluster_ids, key=lambda c: (-cluster_sizes[c], _det_tie(ep_name, c)))

    for cid in ordered:
        s = float(cluster_sizes[cid])
        q = cluster_strata[cid].astype(float)

        # For classification, check if any bin is currently missing a class present in this cluster
        eligible = list(range(n_bins))
        if is_classification:
            missing_bins = [
                b for b in eligible
                if (q[1] > 0 and bin_strata[b, 1] == 0) or (q[0] > 0 and bin_strata[b, 0] == 0)
            ]
            if missing_bins:
                # Prefer bins missing a class, unless they would blow past cap while another missing bin fits
                if max_bin_cap is not None:
                    fit = [b for b in missing_bins if bin_rows[b] + s <= max_bin_cap]
                    if fit:
                        eligible = fit
                    else:
                        eligible = missing_bins
                else:
                    eligible = missing_bins

        if max_bin_cap is not None and len(eligible) > 1:
            under_cap = [b for b in eligible if bin_rows[b] + s <= max_bin_cap]
            if under_cap:
                eligible = under_cap

        best_b = eligible[0]
        best_cost = float("inf")
        for b in eligible:
            # Marginal increase in normalized squared load (strictly prefers lower current load!)
            dv_row = ((bin_rows[b] + s) ** 2 - bin_rows[b] ** 2) / (target_row ** 2)
            dv_str = float(np.sum(((bin_strata[b] + q) ** 2 - bin_strata[b] ** 2) / (target_strata ** 2)))
            # Mild extra penalty if bin already exceeds target_row while others don't
            over_pen = 5.0 * max(0.0, (bin_rows[b] + s - target_row) / target_row) ** 2 if bin_rows[b] >= target_row else 0.0
            cost = 4.0 * dv_row + 1.0 * dv_str + over_pen
            if cost < best_cost - 1e-12:
                best_cost = cost
                best_b = b

        assignment[cid] = best_b
        bin_rows[best_b] += s
        bin_strata[best_b] += q

    # Local 1-move refinement: strictly improve total variance objective while preserving hard invariants
    def total_obj(br: np.ndarray, bs: np.ndarray) -> float:
        row_var = float(np.sum(((br - target_row) / target_row) ** 2))
        str_var = float(np.sum(((bs - target_strata[None, :]) / target_strata[None, :]) ** 2))
        return 4.0 * row_var + 1.0 * str_var

    cur_obj = total_obj(bin_rows, bin_strata)
    for _ in range(30):
        improved = False
        for cid in ordered:
            s = float(cluster_sizes[cid])
            q = cluster_strata[cid].astype(float)
            b_from = assignment[cid]
            if bin_rows[b_from] - s <= 0:
                continue
            if is_classification:
                if (bin_strata[b_from, 0] - q[0] <= 0) or (bin_strata[b_from, 1] - q[1] <= 0):
                    continue
            best_to = b_from
            best_obj = cur_obj
            for b_to in range(n_bins):
                if b_to == b_from:
                    continue
                if max_bin_cap is not None and bin_rows[b_to] + s > max_bin_cap and bin_rows[b_from] <= max_bin_cap:
                    continue
                br_new = bin_rows.copy()
                bs_new = bin_strata.copy()
                br_new[b_from] -= s
                br_new[b_to] += s
                bs_new[b_from] -= q
                bs_new[b_to] += q
                cand_obj = total_obj(br_new, bs_new)
                if cand_obj < best_obj - 1e-9:
                    best_obj = cand_obj
                    best_to = b_to
            if best_to != b_from:
                bin_rows[b_from] -= s
                bin_rows[best_to] += s
                bin_strata[b_from] -= q
                bin_strata[best_to] += q
                assignment[cid] = best_to
                cur_obj = best_obj
                improved = True
        if not improved:
            break

    return assignment


def assign_splits(
    rows: list[dict],
    comp_cluster: dict[str, str],
    comp_map: dict[str, dict],
    ep_name: str,
) -> tuple[list[dict], dict]:
    """
    Freeze per-endpoint ~20% outer group holdout (`is_holdout=1, fold_id=-1`)
    and size-balanced 5-fold grouped CV (`is_holdout=0, fold_id=0..4`) with ZERO leakage.
    """
    n = len(rows)
    ttype = rows[0]["task_type"]
    is_cls = ttype == "classification"

    # Optional physical domain clamping for fraction endpoints (PPB, F) on output copy only
    clean_rows: list[dict] = []
    n_clamped = 0
    for r in rows:
        nr = dict(r)
        if ep_name in ("PPB", "F"):
            yv = float(nr["y_value"])
            yl = float(nr["y_lower"])
            yh = float(nr["y_upper"])
            new_yl = max(0.0, yl)
            new_yh = min(1.0, yh)
            if nr["censor_type"] != "exact" and new_yl >= new_yh:
                new_yl, new_yh = yl, yh  # never invert bounds
            if abs(new_yl - yl) > 1e-12 or abs(new_yh - yh) > 1e-12:
                nr["y_lower"] = str(round(new_yl, 6))
                nr["y_upper"] = str(round(new_yh, 6))
                n_clamped += 1
        clean_rows.append(nr)

    groups = np.array([comp_cluster[r["compound_key"]] for r in clean_rows])
    uniq_groups = sorted(set(groups.tolist()))
    n_groups = len(uniq_groups)
    assert n_groups >= 10, f"Endpoint {ep_name} has too few clusters ({n_groups}) for 20% holdout + 5-fold CV!"

    # Build stratification vector (class 0/1 or adaptive quantile bins)
    if is_cls:
        y_strat = np.array([int(float(r["y_value"])) for r in clean_rows], dtype=int)
        n_strata = 2
    else:
        yv = np.array([float(r["y_value"]) for r in clean_rows], dtype=float)
        n_bins_q = 5 if n_groups >= 40 else (3 if n_groups >= 18 else 2)
        qs = np.unique(np.quantile(yv, np.linspace(0, 1, n_bins_q + 1)[1:-1]))
        y_strat = np.digitize(yv, qs, right=False).astype(int)
        n_strata = len(qs) + 1

    # Aggregate cluster sizes and strata histograms
    cluster_sizes: dict[str, int] = Counter()
    cluster_strata: dict[str, np.ndarray] = {c: np.zeros(n_strata, dtype=int) for c in uniq_groups}
    for i, c in enumerate(groups):
        cluster_sizes[c] += 1
        cluster_strata[c][y_strat[i]] += 1

    # 1. Outer ~20% holdout: stratified 5-bin LPT packing, then pick best holdout bin
    outer_asg = _bin_pack_clusters(
        uniq_groups, cluster_sizes, cluster_strata, n_bins=5, ep_name=f"{ep_name}|outer", is_classification=is_cls
    )
    global_pos_rate = float(y_strat.mean()) if is_cls else 0.0

    best_ho_bin = None
    best_ho_score = float("inf")
    for b in range(5):
        ho_c = {c for c, ab in outer_asg.items() if ab == b}
        cv_c = set(uniq_groups) - ho_c
        ho_n = sum(cluster_sizes[c] for c in ho_c)
        cv_n = n - ho_n
        if ho_n == 0 or cv_n == 0 or len(cv_c) < 5:
            continue
        if is_cls:
            ho_pos = sum(int(cluster_strata[c][1]) for c in ho_c)
            ho_neg = sum(int(cluster_strata[c][0]) for c in ho_c)
            cv_pos = sum(int(cluster_strata[c][1]) for c in cv_c)
            cv_neg = sum(int(cluster_strata[c][0]) for c in cv_c)
            if min(ho_pos, ho_neg, cv_pos, cv_neg) == 0:
                continue
            if cv_pos < 5 or cv_neg < 5:
                continue
        max_cv_clu = max(cluster_sizes[c] for c in cv_c)
        cv_cap = max(10, int(0.35 * cv_n))
        # Strongly require remaining CV clusters to fit within 35% fold cap!
        cap_violation = max(0, max_cv_clu - cv_cap)
        frac = ho_n / n
        frac_err = abs(frac - 0.20)
        cls_err = abs((ho_pos / ho_n) - global_pos_rate) if is_cls else 0.0
        # Prefer bins within [0.15, 0.25] that satisfy max_cv_clu <= cv_cap
        score = 100.0 * cap_violation + 2.0 * frac_err + 1.0 * cls_err
        if score < best_ho_score:
            best_ho_score = score
            best_ho_bin = b

    assert best_ho_bin is not None, f"Failed to select valid outer holdout bin for {ep_name}!"
    ho_clusters = {c for c, ab in outer_asg.items() if ab == best_ho_bin}
    cv_clusters = sorted(set(uniq_groups) - ho_clusters)

    n_cv_rows = sum(cluster_sizes[c] for c in cv_clusters)
    cv_min_req = max(3, int(0.10 * n_cv_rows))
    cv_max_req = max(10, int(0.35 * n_cv_rows))

    # 2. Inner 5-fold grouped CV via size-balanced stratified cluster bin-packing
    cv_asg = _bin_pack_clusters(
        cv_clusters,
        cluster_sizes,
        cluster_strata,
        n_bins=5,
        ep_name=f"{ep_name}|inner",
        is_classification=is_cls,
        max_bin_cap=cv_max_req,
    )

    # Map assignments back to rows
    is_holdout = np.zeros(n, dtype=int)
    fold_id = np.full(n, -1, dtype=int)
    for i, c in enumerate(groups):
        if c in ho_clusters:
            is_holdout[i] = 1
            fold_id[i] = -1
        else:
            is_holdout[i] = 0
            fold_id[i] = cv_asg[c]

    # 3. HARD ANTI-LEAKAGE & FOLD-BALANCE ASSERTIONS
    assert len(ho_clusters & set(cv_clusters)) == 0, f"LEAKAGE holdout vs CV in {ep_name}!"

    def _id_tokens(mask: np.ndarray) -> tuple[set[str], set[str], set[str]]:
        cks, ik14s, chs = set(), set(), set()
        for idx in np.where(mask)[0]:
            ck = clean_rows[idx]["compound_key"]
            cks.add(ck)
            cm = comp_map[ck]
            if cm["inchikey_14"]:
                ik14s.add(cm["inchikey_14"])
            if cm["conn_hash"]:
                chs.add(cm["conn_hash"])
        return cks, ik14s, chs

    ho_mask = is_holdout == 1
    cv_mask = is_holdout == 0
    ho_cks, ho_ik14, ho_ch = _id_tokens(ho_mask)
    cv_cks, cv_ik14, cv_ch = _id_tokens(cv_mask)
    assert len(ho_cks & cv_cks) == 0, f"COMPOUND_KEY LEAKAGE holdout vs CV in {ep_name}!"
    assert len(ho_ik14 & cv_ik14) == 0, f"INCHIKEY_14 LEAKAGE holdout vs CV in {ep_name}!"
    assert len(ho_ch & cv_ch) == 0, f"CONN_HASH LEAKAGE holdout vs CV in {ep_name}!"

    fold_cluster_sets = []
    fold_id_sets = []
    fold_sizes = {}
    fold_class_counts = {}
    fold_y_medians = {}

    for fi in range(5):
        mask = (is_holdout == 0) & (fold_id == fi)
        cnt = int(mask.sum())
        assert cnt > 0, f"Empty CV fold {fi} in {ep_name}!"
        if is_cls:
            pos_f = int((y_strat[mask] == 1).sum())
            neg_f = int((y_strat[mask] == 0).sum())
            assert pos_f > 0 and neg_f > 0, f"Single-class CV fold {fi} in {ep_name}: pos={pos_f}, neg={neg_f}"
            fold_class_counts[f"fold_{fi}"] = {"pos": pos_f, "neg": neg_f}
        else:
            fold_y_medians[f"fold_{fi}"] = round(float(np.median([float(clean_rows[k]["y_value"]) for k in np.where(mask)[0]])), 4)

        fc = set(groups[mask])
        ftoks = _id_tokens(mask)
        for prev_i, (prev_c, prev_toks) in enumerate(zip(fold_cluster_sets, fold_id_sets)):
            assert len(fc & prev_c) == 0, f"CLUSTER LEAKAGE fold {fi} vs {prev_i} in {ep_name}!"
            assert len(ftoks[0] & prev_toks[0]) == 0, f"COMPOUND_KEY LEAKAGE fold {fi} vs {prev_i} in {ep_name}!"
            assert len(ftoks[1] & prev_toks[1]) == 0, f"INCHIKEY_14 LEAKAGE fold {fi} vs {prev_i} in {ep_name}!"
            assert len(ftoks[2] & prev_toks[2]) == 0, f"CONN_HASH LEAKAGE fold {fi} vs {prev_i} in {ep_name}!"
        fold_cluster_sets.append(fc)
        fold_id_sets.append(ftoks)
        fold_sizes[f"fold_{fi}"] = cnt

    assert np.all((is_holdout == 1) == (fold_id == -1)), f"fold_id/-1 mismatch in {ep_name}!"
    if is_cls:
        ho_pos = int((y_strat[ho_mask] == 1).sum())
        ho_neg = int((y_strat[ho_mask] == 0).sum())
        assert ho_pos > 0 and ho_neg > 0, f"Single-class holdout in {ep_name}!"

    min_f = min(fold_sizes.values())
    max_f = max(fold_sizes.values())
    assert min_f >= cv_min_req, f"{ep_name}: min fold size {min_f} < required {cv_min_req} (n_cv={n_cv_rows}, sizes={fold_sizes})"
    assert max_f <= cv_max_req, f"{ep_name}: max fold size {max_f} > required {cv_max_req} (n_cv={n_cv_rows}, sizes={fold_sizes})"

    # Attach cluster_id, is_holdout, fold_id to output rows
    out_rows = []
    for i, r in enumerate(clean_rows):
        nr = dict(r)
        nr["cluster_id"] = str(groups[i])
        nr["is_holdout"] = int(is_holdout[i])
        nr["fold_id"] = int(fold_id[i])
        out_rows.append(nr)

    info = {
        "n_rows": n,
        "n_clusters": int(n_groups),
        "n_cv_rows": int(n_cv_rows),
        "n_holdout_rows": int(ho_mask.sum()),
        "holdout_frac": round(float(ho_mask.sum() / n), 4),
        "n_cv_clusters": len(cv_clusters),
        "n_holdout_clusters": len(ho_clusters),
        "max_cluster_rows": int(max(cluster_sizes.values())),
        "fold_sizes": fold_sizes,
        "min_fold_size": int(min_f),
        "max_fold_size": int(max_f),
        "min_fold_frac": round(float(min_f / n_cv_rows), 4),
        "max_fold_frac": round(float(max_f / n_cv_rows), 4),
        "fold_balance_pass": True,
        "clamped_oob_bounds": int(n_clamped),
    }
    if is_cls:
        info["fold_class_counts"] = fold_class_counts
        info["holdout_class_counts"] = {"pos": ho_pos, "neg": ho_neg}
    else:
        info["fold_y_medians"] = fold_y_medians
        info["holdout_y_median"] = round(float(np.median([float(clean_rows[k]["y_value"]) for k in np.where(ho_mask)[0]])), 4)
    return out_rows, info


def main():
    t0 = time.time()
    FINAL_EP_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading frozen compounds_v33.tsv & features/morgan_bits_1024.npz (READ-ONLY)...", flush=True)
    comp_rows = []
    with (DATA_DIR / "compounds_v33.tsv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            comp_rows.append(row)
    n_comp = len(comp_rows)
    comp_map = {c["compound_key"]: c for c in comp_rows}

    ck_to_fidx: dict[str, int] = {}
    with (FEAT_DIR / "compound_index.tsv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            ck_to_fidx[row["compound_key"]] = int(row["compound_idx"])

    mb = np.load(FEAT_DIR / "morgan_bits_1024.npz", allow_pickle=False)["morgan_bits_1024"]
    print(f"  Loaded {n_comp:,} unique compounds and Morgan 1024-bit matrix {mb.shape}.", flush=True)

    # Stage (a): Exact canonical_identity_group (ONLY inchikey_14 / conn_hash / canonical_smiles)
    print("[2/4] Stage (a) exact identity groups + Stage (b) non-chaining Dual-Track leader clustering...", flush=True)
    uf = UnionFind(n_comp)
    by_conn: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, c in enumerate(comp_rows):
        if c["inchikey_14"]:
            by_conn[("ik14", c["inchikey_14"])].append(i)
        if c["conn_hash"]:
            by_conn[("ch", c["conn_hash"])].append(i)
        if c["canonical_smiles"]:
            by_conn[("smi", c["canonical_smiles"])].append(i)
    for idxs in by_conn.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)

    idg_members: dict[int, list[int]] = defaultdict(list)
    for i in range(n_comp):
        idg_members[uf.find(i)].append(i)
    idg_roots = sorted(idg_members.keys())
    n_idg = len(idg_roots)
    root_to_gid = {r: gi for gi, r in enumerate(idg_roots)}

    # Partition atomic identity groups into Track S (linear sequence-defined) vs Track M (Butina Morgan)
    track_s_gids: dict[int, str] = {}
    track_m_gids: list[tuple[int, int]] = []
    track_o_gids: list[int] = []

    for gi, r in enumerate(idg_roots):
        ms = idg_members[r]
        any_cyc = any(int(comp_rows[i]["is_cyclic"]) == 1 for i in ms)

        best_seq = ""
        best_sorig = "none"
        for i in ms:
            c = comp_rows[i]
            ps = c["parent_sequence"]
            so = c["seq_origin"]
            if ps:
                if so == "source_bare_20aa":
                    best_seq, best_sorig = ps, so
                    break
                elif so == "source_parent_projection" and best_sorig != "source_parent_projection":
                    best_seq, best_sorig = ps, so
                elif not best_seq:
                    best_seq, best_sorig = ps, so

        ms_smi = [i for i in ms if mb[ck_to_fidx[comp_rows[i]["compound_key"]]].sum() > 0]
        n_known = sum(1 for ch in best_seq if ch != "X")
        has_good_linear_seq = (
            (not any_cyc)
            and bool(best_seq)
            and (
                (not ms_smi)
                or (
                    best_sorig in ("source_bare_20aa", "source_parent_projection")
                    and len(best_seq) >= 6
                    and n_known >= 5
                    and (n_known / len(best_seq)) >= 0.75
                )
                or (
                    best_sorig == "rdkit_backbone_extracted"
                    and len(best_seq) >= 8
                    and n_known >= 6
                    and (n_known / len(best_seq)) >= 0.80
                )
            )
        )
        if has_good_linear_seq:
            track_s_gids[gi] = best_seq
        elif ms_smi:
            ms_smi.sort(key=lambda i: (0 if comp_rows[i]["smiles_origin"] == "source" else 1, comp_rows[i]["compound_key"]))
            track_m_gids.append((gi, ck_to_fidx[comp_rows[ms_smi[0]]["compound_key"]]))
        else:
            track_o_gids.append(gi)

    print(
        f"  Exact identity groups: {n_idg:,} -> Track S (linear seq >=70%): {len(track_s_gids):,}, "
        f"Track M (Butina Morgan T>=0.70): {len(track_m_gids):,}, Track O: {len(track_o_gids):,}",
        flush=True,
    )

    # Track S: Leader-based >= 70% sequence identity clustering
    t_s = time.time()
    seq_cid_map = cluster_sequences_leader(list(set(track_s_gids.values())), thresh=SEQ_IDENTITY_THRESHOLD)
    gid_raw_label: dict[int, str] = {}
    for gi, ps in track_s_gids.items():
        gid_raw_label[gi] = f"S_{seq_cid_map[ps]}"
    n_s_clu = len(set(seq_cid_map.values()))
    print(f"  Track S done in {time.time() - t_s:.1f}s ({len(seq_cid_map):,} unique seqs -> {n_s_clu:,} clusters).", flush=True)

    # Track M: Exact Taylor-Butina leader clustering at Tanimoto >= 0.70
    t_m = time.time()
    m_fidxs = np.array([fidx for _, fidx in track_m_gids], dtype=np.int64)
    m_cids = cluster_morgan_butina(mb[m_fidxs], thresh=TANIMOTO_THRESHOLD)
    for idx_m, (gi, _) in enumerate(track_m_gids):
        gid_raw_label[gi] = f"M_{m_cids[idx_m]}"
    for idx_o, gi in enumerate(track_o_gids):
        gid_raw_label[gi] = f"O_{idx_o}"
    n_m_clu = len(set(m_cids)) if m_cids else 0
    print(f"  Track M done in {time.time() - t_m:.1f}s ({len(track_m_gids):,} id-groups -> {n_m_clu:,} clusters).", flush=True)

    # Finalize deterministic CLU_XXXXXX labels (NEVER touching compounds_v33.tsv!)
    raw_to_cid: dict[str, str] = {}
    comp_cluster: dict[str, str] = {}
    cluster_sizes: Counter[str] = Counter()
    for i, c in enumerate(comp_rows):
        rlab = gid_raw_label[root_to_gid[uf.find(i)]]
        if rlab not in raw_to_cid:
            raw_to_cid[rlab] = f"CLU_{len(raw_to_cid):06d}"
        cid = raw_to_cid[rlab]
        comp_cluster[c["compound_key"]] = cid
        cluster_sizes[cid] += 1

    print(
        f"  Unified non-chaining clusters: {len(cluster_sizes):,} clusters for {n_comp:,} compounds | "
        f"max={max(cluster_sizes.values())}, median={float(np.median(list(cluster_sizes.values()))):.1f}, "
        f"top5={cluster_sizes.most_common(5)}",
        flush=True,
    )

    # Step 3: Per-endpoint grouped splits & leakage + fold-balance verification
    print("[3/4] Freezing per-endpoint ~20% outer group holdout + size-balanced 5-fold grouped CV...", flush=True)
    split_summary = {}
    for ep_path in sorted(RAW_EP_DIR.glob("*.tsv")):
        ep = ep_path.stem
        with ep_path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            in_fields = list(reader.fieldnames or [])
            rows = list(reader)
        if not rows:
            continue
        out_rows, info = assign_splits(rows, comp_cluster, comp_map, ep)
        out_fields = in_fields + ["cluster_id", "is_holdout", "fold_id"]
        with (FINAL_EP_DIR / f"{ep}.tsv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=out_fields, delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(out_rows)
        split_summary[ep] = info
        fs = [info["fold_sizes"][f"fold_{i}"] for i in range(5)]
        print(
            f"  {ep:25s}: n={info['n_rows']:5,d} clu={info['n_clusters']:4,d} | "
            f"ho={info['n_holdout_rows']:4,d} ({info['holdout_frac']*100:4.1f}%) | "
            f"cv={info['n_cv_rows']:5,d} folds={fs} [{info['min_fold_frac']*100:4.1f}%..{info['max_fold_frac']*100:4.1f}%]",
            flush=True,
        )

    out_json = {
        "schema": "peptide_omnipanel_v33_split_summary_v2_nonchaining",
        "elapsed_sec": round(time.time() - t0, 2),
        "clustering_method": {
            "stage_a": "exact_canonical_identity_group (inchikey_14, conn_hash, canonical_smiles)",
            "stage_b_track_s": f"non_chaining_leader_sequence_identity >= {SEQ_IDENTITY_THRESHOLD:.2f} (linear canonical peptides)",
            "stage_b_track_m": f"non_chaining_taylor_butina_morgan_bits_1024 Tanimoto >= {TANIMOTO_THRESHOLD:.2f} (cyclic/short/modified peptides)",
            "murcko_generic_collapse": False,
            "compounds_v33_modified": False,
        },
        "total_compounds": n_comp,
        "exact_identity_groups": n_idg,
        "track_s_identity_groups": len(track_s_gids),
        "track_m_identity_groups": len(track_m_gids),
        "track_s_clusters": n_s_clu,
        "track_m_clusters": n_m_clu,
        "total_clusters": len(cluster_sizes),
        "max_cluster_size": max(cluster_sizes.values()),
        "median_cluster_size": float(np.median(list(cluster_sizes.values()))),
        "top10_cluster_sizes": cluster_sizes.most_common(10),
        "endpoints": split_summary,
    }
    (DATA_DIR / "split_summary.json").write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[4/4] DONE Step 03 in {out_json['elapsed_sec']}s! ZERO leakage & strict fold balance verified across all {len(split_summary)} endpoints!",
        flush=True,
    )


if __name__ == "__main__":
    main()
