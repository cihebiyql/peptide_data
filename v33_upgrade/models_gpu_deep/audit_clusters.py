#!/usr/bin/env python3
"""Audit cross-endpoint cluster overlap, holdout unions, and condition vocabularies."""
import csv
from pathlib import Path
import numpy as np

z = np.load(r"D:\work\中检院\v33_upgrade\models_gpu_deep\_cache\embeddings\chemberta_mtr.npz")
print("chemberta_mtr.npz:", {k: (z[k].shape, str(z[k].dtype)) for k in z.files})
z.close()

ep_dir = Path(r"D:\work\中检院\v33_upgrade\models_gpu_deep\_cache\data\endpoints")
eps = {}
for p in sorted(ep_dir.glob("*.tsv")):
    ep = p.stem
    with p.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    cv_cl = set(r["cluster_id"] for r in rows if r["is_holdout"] == "0")
    ho_cl = set(r["cluster_id"] for r in rows if r["is_holdout"] == "1")
    all_cl = cv_cl | ho_cl
    c17_cl = set(r["cluster_id"] for r in rows if r["case17_excluded"] == "1")
    cond_cols = [c for c in rows[0].keys() if c.startswith("cond_")]
    fold_cl = {k: set(r["cluster_id"] for r in rows if r["is_holdout"] == "0" and int(r["fold_id"]) == k) for k in range(5)}
    eps[ep] = dict(rows=rows, n=len(rows), cv_cl=cv_cl, ho_cl=ho_cl, all_cl=all_cl, c17_cl=c17_cl, cond_cols=cond_cols, fold_cl=fold_cl)

all_ho_union = set().union(*[d["ho_cl"] for d in eps.values()])
all_cl_union = set().union(*[d["all_cl"] for d in eps.values()])
print(f"Loaded {len(eps)} eps | total clusters={len(all_cl_union)} | union of ALL 24 HO clusters={len(all_ho_union)} ({100*len(all_ho_union)/len(all_cl_union):.1f}%)")

tot_cv = sum(sum(1 for r in d["rows"] if r["is_holdout"] == "0") for d in eps.values())
cv_in_any_other_ho = 0
for ep, d in eps.items():
    other_ho = set().union(*[eps[o]["ho_cl"] for o in eps if o != ep])
    nd = sum(1 for r in d["rows"] if r["is_holdout"] == "0" and r["cluster_id"] in other_ho)
    cv_in_any_other_ho += nd
    n_cv_ep = sum(1 for r in d["rows"] if r["is_holdout"] == "0")
    print(f"{ep:26s} n={d['n']:5d} cv_cl={len(d['cv_cl']):4d} ho_cl={len(d['ho_cl']):4d} | cv_in_ANY_other_HO={nd:4d}/{n_cv_ep:5d} ({100*nd/n_cv_ep:4.1f}%) | conds={d['cond_cols']}")

print(f"\nTotal CV rows overlapping ANY other endpoint HO: {cv_in_any_other_ho}/{tot_cv} ({100*cv_in_any_other_ho/tot_cv:.1f}%)")

groups = {
    "PK_SmallPerm": ["CL_absolute", "CL_per_kg", "Cmax", "F", "PPB", "Vd_per_kg", "MDCK", "RRCK", "LogD7.4", "solubility_reg"],
    "Permeability_Sol": ["PAMPA", "Caco2", "BBB", "solubility_cls"],
    "HalfLife": ["half_life_reg", "half_life_cls"],
    "Safety_Tox": ["hemolysis_cls", "hemolysis_HC50", "cytotoxicity_cls", "overall_peptide_toxicity"],
    "Microbiology": ["MIC_E_coli", "MIC_S_aureus", "MIC_P_aeruginosa", "MIC_C_albicans"],
}
print("\n--- Group HO Union Analysis ---")
for gname, members in groups.items():
    g_ho = set().union(*[eps[m]["ho_cl"] for m in members])
    tot_drop = sum(sum(1 for r in d["rows"] if r["is_holdout"] == "0" and r["cluster_id"] in g_ho) for d in eps.values())
    print(f"Group {gname:18s}: |g_ho|={len(g_ho):4d}, drops {tot_drop:5d}/{tot_cv} ({100*tot_drop/tot_cv:4.1f}%) of all-24 CV rows")
    for m in members:
        cv_r = [r for r in eps[m]["rows"] if r["is_holdout"] == "0"]
        nd = sum(1 for r in cv_r if r["cluster_id"] in g_ho)
        if nd > 0:
            print(f"    INTRA {m}: {nd}/{len(cv_r)} ({100*nd/len(cv_r):.1f}%)")

# Check intra-group pairwise cluster overlap (how much data sharing actually exists!)
print("\n--- Pairwise shared clusters (top pairs) ---")
pairs = []
ep_list = sorted(eps.keys())
for i in range(len(ep_list)):
    for j in range(i + 1, len(ep_list)):
        e1, e2 = ep_list[i], ep_list[j]
        sh = len(eps[e1]["all_cl"] & eps[e2]["all_cl"])
        if sh >= 5:
            pairs.append((sh, e1, e2))
for sh, e1, e2 in sorted(pairs, reverse=True)[:35]:
    print(f"  {sh:5d} shared clusters: {e1} <-> {e2}")

# Check case17_excluded cluster union
all_c17_cl = set().union(*[d["c17_cl"] for d in eps.values()])
tot_c17_drop_by_cl = sum(sum(1 for r in d["rows"] if r["cluster_id"] in all_c17_cl or r["case17_excluded"] == "1") for d in eps.values())
tot_c17_flag = sum(sum(1 for r in d["rows"] if r["case17_excluded"] == "1") for d in eps.values())
print(f"\nCase17: flagged rows={tot_c17_flag}, unique clusters={len(all_c17_cl)}, total rows matching ANY Case17 cluster={tot_c17_drop_by_cl}")

# Check half_life_reg vs half_life_cls fold/holdout alignment!
hl_r = eps["half_life_reg"]
hl_c = eps["half_life_cls"]
print(f"\nhalf_life_reg vs half_life_cls:")
print(f"  ho overlap: {len(hl_r['ho_cl'] & hl_c['ho_cl'])} / reg_ho={len(hl_r['ho_cl'])}, cls_ho={len(hl_c['ho_cl'])}")
print(f"  reg_ho in cls_cv: {len(hl_r['ho_cl'] & hl_c['cv_cl'])}, cls_ho in reg_cv: {len(hl_c['ho_cl'] & hl_r['cv_cl'])}")
for k in range(5):
    print(f"  fold {k}: reg_f in cls_other_f = {len(hl_r['fold_cl'][k] & (hl_c['cv_cl'] - hl_c['fold_cl'][k]))}")
