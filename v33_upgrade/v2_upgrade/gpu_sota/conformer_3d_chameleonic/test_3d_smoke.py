#!/usr/bin/env python3
"""Smoke-test chameleonic_3d.py + direct Uni-Mol2-84M from 3D mols on all 17 Case17 clinical peptides."""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

os.environ["UNIMOL_WEIGHT_DIR"] = r"D:\work\中检院\v33_upgrade\models_gpu\_local_cache\unimol2"
sys.path.insert(0, r"D:\work\中检院\v33_upgrade\models_gpu\_local_cache\unimol_pkg")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from rdkit import Chem
from chameleonic_3d import compute_chameleonic_3d_descriptors
from unimol_tools.data.conformer import mol2unimolv2
from unimol_tools.models import UniMolV2Model
from unimol_tools.predictor import MolDataset
from unimol_tools.tasks import Trainer

with open(r"D:\work\中检院\v33_upgrade\models_gpu_deep\_cache\data\case17_queries.tsv", encoding="utf-8") as f:
    c17 = list(csv.DictReader(f, delimiter="\t"))

mols = []
for r in c17:
    smi = r["canonical_smiles"]
    nh = Chem.MolFromSmiles(smi).GetNumHeavyAtoms()
    t0 = time.perf_counter()
    v, m = compute_chameleonic_3d_descriptors(smi, n_confs=3, max_heavy_atoms=250, num_threads=4, return_best_mol=True)
    dt = (time.perf_counter() - t0) * 1000.0
    dn = r["drug_name"]
    print(
        f"{dn:<18} heavy={nh:3d} valid={int(v[-1])} ms={dt:6.1f} | "
        f"rg={v[6]:.2f}+-{v[8]:.2f} imhb={v[16]:.0f}-{v[18]:.0f} psa3d={v[23]:.0f}-{v[23]+v[25]:.0f} ratio={v[26]:.2f}",
        flush=True,
    )
    if m is not None:
        mols.append((dn, m))

t_u = time.perf_counter()
feats = [mol2unimolv2(m, max_atoms=256, remove_hs=True) for _, m in mols]
model = UniMolV2Model(output_dim=1, model_size="84m").to("cuda:0").eval()
ds = MolDataset(feats)
trainer = Trainer(
    task="repr",
    data_type="molecule",
    batch_size=16,
    remove_hs=True,
    model_name="unimolv2",
    model_size="84m",
    use_cuda=True,
    use_gpu="0",
)
out = trainer.inference(model, return_repr=True, dataset=ds)
arr = np.asarray(out["cls_repr"], dtype=np.float32)
print(
    f"UNIMOL2 direct-from-3D-mols OK: {arr.shape}, finite={bool(np.all(np.isfinite(arr)))}, "
    f"std={float(arr.std()):.3f}, elapsed={time.perf_counter() - t_u:.2f}s",
    flush=True,
)
