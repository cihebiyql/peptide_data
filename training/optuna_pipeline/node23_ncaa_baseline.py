#!/usr/bin/env python3
"""S2: ncAA-RepDistill vs Morgan vs ESM2 baseline comparison (run on node23 CPU).

Consumes: /data/qlyu/ncaa_features_v32.npz (from node1 GPU extraction)
          + endpoint tables transferred from local.

Arms per endpoint (all with GroupKFold-5, group=homology∪skeleton union-find):
  ncaa        — ncAA-RepDistill molecular_proj embedding (handles modified peptides!)
  ncaa+morgan — fusion
  morgan+desc — S0 baseline (for reference)
  (esm2 rows are NEW-only; ncaa covers ALL compounds with SMILES including modified)

Run on node23: python3 /data/qlyu/s2_ncaa_baseline.py
"""

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
csv.field_size_limit(10_000_000)

from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

DATA = Path("/data/qlyu/v32_experiment")
NCAA_NPZ = Path("/data/qlyu/ncaa_features_v32.npz")
OUT = DATA / "s2_ncaa_baseline"
(OUT / "preds").mkdir(parents=True, exist_ok=True)
MIN_ROWS = 100
DESC_COLS = ["MolWt", "cLogP", "TPSA", "HBD", "HBA", "RotB", "Rings",
             "HeavyAtoms", "FracCsp3", "FormalCharge", "QED"]


class UF:
    def __init__(self):
        self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    try:
        sp = float(spearmanr(y, p).statistic)
    except Exception:
        sp = float("nan")
    try:
        pe = float(pearsonr(y, p).statistic)
    except Exception:
        pe = float("nan")
    dy, dp = y - y.mean(), p - p.mean()
    both = (np.abs(dy) > 1e-9) & (np.abs(dp) > 1e-9)
    dirr = float(np.mean(np.sign(dy[both]) == np.sign(dp[both]))) if both.sum() > 10 else float("nan")
    return mae, rmse, sp, pe, dirr


def run_arm(tag, X, y, groups, ep, results, log):
    gkf = GroupKFold(n_splits=5)
    try:
        splits = list(gkf.split(X, y, groups))
    except ValueError:
        return
    pred = np.zeros_like(y)
    for tr, te in splits:
        sc = StandardScaler(with_mean=False).fit(X[tr])
        ridge = Ridge(alpha=10.0).fit(sc.transform(X[tr]), y[tr])
        pred[te] = ridge.predict(sc.transform(X[te]))
    mae, rmse, sp, pe, dr = metrics(y, pred)
    results.append([ep, tag, "ridge", len(y), mae, rmse, sp, pe, dr])
    log(f"  {tag:<24} ridge   MAE={mae:.4f} Sp={sp:.3f} Dir={dr:.3f}")
    if len(y) >= 300:
        seed_preds = []
        for seed in (0, 1, 2):
            pred_l = np.zeros_like(y)
            for tr, te in splits:
                m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                                      min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                                      random_state=seed, verbose=-1, n_jobs=16)
                m.fit(X[tr], y[tr])
                pred_l[te] = m.predict(X[te])
            mae, rmse, sp, pe, dr = metrics(y, pred_l)
            results.append([ep, tag, f"lgbm_s{seed}", len(y), mae, rmse, sp, pe, dr])
            log(f"  {tag:<24} lgbm s{seed} MAE={mae:.4f} Sp={sp:.3f} Dir={dr:.3f}")


def main():
    log_lines = []
    def log(s=""):
        log_lines.append(s)
        print(s, flush=True)

    # ---- load ncaa features ----
    zn = np.load(NCAA_NPZ, allow_pickle=True)
    ncaa_ids = list(zn["compound_ids"])
    ncaa_mat = zn["embeddings"].astype(np.float32)
    ncaa_map = {c: ncaa_mat[i] for i, c in enumerate(ncaa_ids)}
    log(f"[feat] ncAA-RepDistill: {ncaa_mat.shape}")

    # ---- load morgan ----
    zm = np.load(DATA / "morgan_r2_2048.npz", allow_pickle=True)
    morgan_ids = list(zm["compound_ids"])
    fps = zm["fps"].astype(np.float32)
    cid2morgan = {c: fps[i] for i, c in enumerate(morgan_ids)}

    # ---- load descriptors ----
    desc = {}
    with open(DATA / "descriptors.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            desc[r["compound_id"]] = [float(r[c]) if r[c] not in ("", None) else 0.0
                                      for c in DESC_COLS]

    results = []
    for fp in sorted((DATA / "per_endpoint").glob("*.tsv")):
        ep = fp.stem
        rows = []
        with open(fp, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                cid = r.get("new_compound_id", "")
                if not cid:
                    continue
                if cid not in ncaa_map or cid not in cid2morgan:
                    continue
                try:
                    y = float(r["value"])
                except (TypeError, ValueError):
                    continue
                rows.append((cid, y, r.get("homology_cluster_90", ""),
                             r.get("ik_skeleton", "")))
        if len(rows) < MIN_ROWS:
            continue

        uf = UF()
        for cid, _, cl, sk in rows:
            uf.union("C:" + cl, "K:" + sk)
        groups = np.array([uf.find("C:" + r[2]) for r in rows])
        y = np.array([r[1] for r in rows])
        ncaa = np.stack([ncaa_map[r[0]] for r in rows])
        morgan = np.stack([cid2morgan[r[0]] for r in rows])
        descm = np.array([desc.get(r[0], [0.0] * len(DESC_COLS)) for r in rows], float)
        n_groups = len(set(groups))
        log(f"\n[{ep}] rows={len(rows):,} groups={n_groups:,}")

        run_arm("ncaa", ncaa, y, groups, ep, results, log)
        run_arm("ncaa+desc", np.hstack([ncaa, descm]), y, groups, ep, results, log)
        run_arm("morgan+desc", np.hstack([morgan, descm]), y, groups, ep, results, log)
        run_arm("ncaa+morgan+desc", np.hstack([ncaa, morgan, descm]), y, groups, ep, results, log)

    with open(OUT / "results_matrix.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["endpoint", "arm", "model", "rows", "MAE", "RMSE",
                    "Spearman", "Pearson", "DirAgree"])
        w.writerows(results)
    with open(OUT / "report.md", "w", encoding="utf-8") as f:
        f.write("# S2 ncAA-RepDistill 基线对比\n\n```\n" + "\n".join(log_lines) + "\n```\n")
    log(f"\n[out] {OUT}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[elapsed] {time.time()-t0:.0f}s", file=sys.stderr)
