"""Built-in curated dictionary of 17 classic clinical peptide cases,
the complete FDA Echinocandin cyclic lipopeptide class (4 drugs), and 5 classic AMPs (25 total)."""

from __future__ import annotations

import re
from typing import Any, Optional

EXAMPLE_CATEGORIES: list[dict[str, str]] = [
    {
        "id": "glp1_insulin",
        "title_en": "Long-Acting Lipidated (GLP-1 & Insulin)",
        "title_zh": "长效脂化 · GLP-1 与胰岛素",
        "badge": "Acylation Anchor",
    },
    {
        "id": "echinocandin",
        "title_en": "Echinocandin Cyclic Lipopeptides (4)",
        "title_zh": "棘白菌素类环肽 (4)",
        "badge": "Echinocandin",
    },
    {
        "id": "cyclic_macrocycle",
        "title_en": "Macrocycles, Disulfides & Cyclic Lipopeptides",
        "title_zh": "环肽、大环与二硫键药物",
        "badge": "Cyclic",
    },
    {
        "id": "linear_clinical",
        "title_en": "Linear Clinical Peptide Therapeutics",
        "title_zh": "线性临床多肽药物",
        "badge": "Linear",
    },
    {
        "id": "amp_sequences",
        "title_en": "Canonical Antimicrobial Peptide Sequences",
        "title_zh": "经典抗菌肽序列",
        "badge": "20-AA Sequence",
    },
]

DRUG_LIBRARY: list[dict[str, Any]] = [
    # ---- 1. Long-Acting Lipidated GLP-1 & Insulin (Case17) ----
    {
        "key": "Semaglutide",
        "display_name": "Semaglutide",
        "zh_name": "司美格鲁肽",
        "aliases": ["semaglutide", "ozempic", "wegovy", "rybelsus", "司美格鲁肽"],
        "category": "glp1_insulin",
        "drug_class": "GLP-1 RA · C18 Fatty Diacid",
        "sequence": None,
        "smiles": "CC[C@H](C)[C@@H](C(=O)N[C@@H](C)C(=O)N[C@@H](CC1=CNC2=CC=CC=C21)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)O)NC(=O)[C@H](CC3=CC=CC=C3)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCCCNC(=O)COCCOCCNC(=O)COCCOCCNC(=O)CC[C@H](C(=O)O)NC(=O)CCCCCCCCCCCCCCCCC(=O)O)NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](CCC(=O)N)NC(=O)CNC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC4=CC=C(C=C4)O)NC(=O)[C@H](CO)NC(=O)[C@H](CO)NC(=O)[C@H](C(C)C)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CO)NC(=O)[C@H]([C@@H](C)O)NC(=O)[C@H](CC5=CC=CC=C5)NC(=O)[C@H]([C@@H](C)O)NC(=O)CNC(=O)[C@H](CCC(=O)O)NC(=O)C(C)(C)NC(=O)[C@H](CC6=CN=CN6)N",
        "fda_half_life": "~165–168 h (~1 week, FDA OZEMPIC)",
        "fda_half_life_h": 168.0,
        "fda_ppb": ">99% (serum albumin)",
        "fda_ppb_frac": 0.99,
        "case17": True,
    },
    {
        "key": "Tirzepatide",
        "display_name": "Tirzepatide",
        "zh_name": "替尔泊肽",
        "aliases": ["tirzepatide", "mounjaro", "zepbound", "替尔泊肽"],
        "category": "glp1_insulin",
        "drug_class": "GIP/GLP-1 Dual Agonist · C20 Diacid",
        "sequence": None,
        "smiles": "CC[C@H](C)[C@@H](C(=O)N[C@@H](C)C(=O)N[C@@H](CCC(=O)N)C(=O)N[C@@H](CCCCNC(=O)COCCOCCNC(=O)COCCOCCNC(=O)CC[C@@H](C(=O)O)NC(=O)CCCCCCCCCCCCCCCCCCC(=O)O)C(=O)N[C@@H](C)C(=O)N[C@@H](CC1=CC=CC=C1)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CCC(=O)N)C(=O)N[C@@H](CC2=CNC3=CC=CC=C32)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H]([C@@H](C)CC)C(=O)N[C@@H](C)C(=O)NCC(=O)NCC(=O)N4CCC[C@H]4C(=O)N[C@@H](CO)C(=O)N[C@@H](CO)C(=O)NCC(=O)N[C@@H](C)C(=O)N5CCC[C@H]5C(=O)N6CCC[C@H]6C(=O)N7CCC[C@H]7C(=O)N[C@@H](CO)C(=O)N)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CC(C)C)NC(=O)C(C)(C)NC(=O)[C@H]([C@@H](C)CC)NC(=O)[C@H](CO)NC(=O)[C@H](CC8=CC=C(C=C8)O)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CO)NC(=O)[C@H]([C@@H](C)O)NC(=O)[C@H](CC9=CC=CC=C9)NC(=O)[C@H]([C@@H](C)O)NC(=O)CNC(=O)[C@H](CCC(=O)O)NC(=O)C(C)(C)NC(=O)[C@H](CC1=CC=C(C=C1)O)N",
        "fda_half_life": "~125 h (~5 days, FDA MOUNJARO)",
        "fda_half_life_h": 125.0,
        "fda_ppb": "~96% (serum albumin)",
        "fda_ppb_frac": 0.96,
        "case17": True,
    },
    {
        "key": "Liraglutide",
        "display_name": "Liraglutide",
        "zh_name": "利拉鲁肽",
        "aliases": ["liraglutide", "victoza", "saxenda", "利拉鲁肽"],
        "category": "glp1_insulin",
        "drug_class": "GLP-1 RA · C16 Palmitoyl Monoacyl",
        "sequence": None,
        "smiles": "CCCCCCCCCCCCCCCC(=O)N[C@@H](CCC(=O)NCCCC[C@@H](C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CC1=CC=CC=C1)C(=O)N[C@@H]([C@@H](C)CC)C(=O)N[C@@H](C)C(=O)N[C@@H](CC2=CNC3=CC=CC=C32)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)O)NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](CCC(=O)N)NC(=O)CNC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC4=CC=C(C=C4)O)NC(=O)[C@H](CO)NC(=O)[C@H](CO)NC(=O)[C@H](C(C)C)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CO)NC(=O)[C@H]([C@@H](C)O)NC(=O)[C@H](CC5=CC=CC=C5)NC(=O)[C@H]([C@@H](C)O)NC(=O)CNC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](C)NC(=O)[C@H](CC6=CN=CN6)N)C(=O)O",
        "fda_half_life": "~13 h SC (FDA VICTOZA)",
        "fda_half_life_h": 13.0,
        "fda_ppb": ">98% (serum albumin)",
        "fda_ppb_frac": 0.985,
        "case17": True,
    },
    {
        "key": "Insulin_detemir",
        "display_name": "Insulin detemir",
        "zh_name": "地特胰岛素",
        "aliases": ["insulindetemir", "detemir", "levemir", "地特胰岛素"],
        "category": "glp1_insulin",
        "drug_class": "Basal Insulin · C14 Myristoyl Monoacyl",
        "sequence": None,
        "smiles": "CCCCCCCCCCCCCC(=O)NCCCC[C@@H](C(=O)O)NC(=O)[C@@H]1CCCN1C(=O)[C@H]([C@@H](C)O)NC(=O)[C@H](CC2=CC=C(C=C2)O)NC(=O)[C@H](CC3=CC=CC=C3)NC(=O)[C@H](CC4=CC=CC=C4)NC(=O)CNC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CCC(=O)O)NC(=O)CNC(=O)[C@@H]5CSSC[C@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H]6CSSC[C@@H](C(=O)N[C@@H](CSSC[C@@H](C(=O)NCC(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N5)C(C)C)CC(C)C)CC7=CC=C(C=C7)O)CC(C)C)C)CCC(=O)O)C(C)C)CC(C)C)CC8=CNC=N8)CO)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC9=CNC=N9)NC(=O)[C@H](CCC(=O)N)NC(=O)[C@H](CC(=O)N)NC(=O)[C@H](C(C)C)NC(=O)[C@H](CC1=CC=CC=C1)N)C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N6)[C@@H](C)CC)CO)[C@@H](C)O)NC(=O)[C@H](CCC(=O)N)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](C(C)C)NC(=O)[C@H]([C@@H](C)CC)NC(=O)CN)CO)CC(C)C)CC1=CC=C(C=C1)O)CCC(=O)N)CC(C)C)CCC(=O)O)CC(=O)N)CC1=CC=C(C=C1)O)C(=O)N[C@@H](CC(=O)N)C(=O)O",
        "fda_half_life": "5–7 h dose-dependent (FDA LEVEMIR)",
        "fda_half_life_h": 6.0,
        "fda_ppb": ">98% (~99%, albumin)",
        "fda_ppb_frac": 0.99,
        "case17": True,
    },
    {
        "key": "Insulin_degludec",
        "display_name": "Insulin degludec",
        "zh_name": "德谷胰岛素",
        "aliases": ["insulindegludec", "degludec", "tresiba", "德谷胰岛素"],
        "category": "glp1_insulin",
        "drug_class": "Ultra-Long Insulin · C16 Diacid Depot",
        "sequence": None,
        "smiles": "CC[C@H](C)[C@H]1C(=O)N[C@H]2CSSC[C@@H](C(=O)N[C@@H](CSSC[C@@H](C(=O)NCC(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@@H](CSSC[C@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC2=O)CO)CC(C)C)CC3=CC=C(C=C3)O)CCC(=O)N)CC(C)C)CCC(=O)O)CC(=O)N)CC4=CC=C(C=C4)O)C(=O)N[C@@H](CC(=O)N)C(=O)O)C(=O)NCC(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)N[C@@H](CC5=CC=CC=C5)C(=O)N[C@@H](CC6=CC=CC=C6)C(=O)N[C@@H](CC7=CC=C(C=C7)O)C(=O)N[C@@H]([C@@H](C)O)C(=O)N8CCC[C@H]8C(=O)N[C@@H](CCCCNC(=O)CC[C@@H](C(=O)O)NC(=O)CCCCCCCCCCCCCCC(=O)O)C(=O)O)C(C)C)CC(C)C)CC9=CC=C(C=C9)O)CC(C)C)C)CCC(=O)O)C(C)C)CC(C)C)CC2=CNC=N2)CO)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC2=CNC=N2)NC(=O)[C@H](CCC(=O)N)NC(=O)[C@H](CC(=O)N)NC(=O)[C@H](C(C)C)NC(=O)[C@H](CC2=CC=CC=C2)N)C(=O)N[C@H](C(=O)N[C@H](C(=O)N1)CO)[C@@H](C)O)NC(=O)[C@H](CCC(=O)N)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](C(C)C)NC(=O)[C@H]([C@@H](C)CC)NC(=O)CN",
        "fda_half_life": "~25 h steady-state (FDA TRESIBA)",
        "fda_half_life_h": 25.0,
        "fda_ppb": ">99% (albumin)",
        "fda_ppb_frac": 0.99,
        "case17": True,
    },
    # ---- 2. Complete FDA Echinocandin Cyclic Lipopeptide Class (NIFDC Focus) ----
    {
        "key": "Caspofungin",
        "display_name": "Caspofungin",
        "zh_name": "卡泊芬净",
        "aliases": ["caspofungin", "cancidas", "卡泊芬净"],
        "category": "echinocandin",
        "drug_class": "Echinocandin Cyclic Lipopeptide",
        "sequence": None,
        "smiles": "CC[C@H](C)C[C@H](C)CCCCCCCCC(=O)N[C@H]1C[C@H]([C@H](NC(=O)[C@@H]2[C@H](CCN2C(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H]3C[C@H](CN3C(=O)[C@@H](NC1=O)[C@@H](C)O)O)[C@@H]([C@H](C4=CC=C(C=C4)O)O)O)[C@@H](CCN)O)O)NCCN)O",
        "fda_half_life": "β-phase 9–11 h (FDA CANCIDAS)",
        "fda_half_life_h": 10.0,
        "fda_ppb": "~97% (albumin)",
        "fda_ppb_frac": 0.97,
        "case17": True,
    },
    {
        "key": "Micafungin",
        "display_name": "Micafungin",
        "zh_name": "米卡芬净",
        "aliases": ["micafungin", "mycamine", "米卡芬净"],
        "category": "echinocandin",
        "drug_class": "Echinocandin Cyclic Lipopeptide (Isoxazole)",
        "sequence": None,
        "smiles": "CCCCCOC1=CC=C(C=C1)C2=CC(=NO2)C3=CC=C(C=C3)C(=O)N[C@H]4C[C@H]([C@H](NC(=O)[C@@H]5[C@H]([C@H](CN5C(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H]6C[C@H](CN6C(=O)[C@@H](NC4=O)[C@@H](C)O)O)[C@@H]([C@H](C7=CC(=C(C=C7)O)OS(=O)(=O)O)O)O)[C@@H](CC(=O)N)O)C)O)O)O",
        "fda_half_life": "14–17 h terminal (FDA MYCAMINE)",
        "fda_half_life_h": 15.0,
        "fda_ppb": ">99% (serum albumin)",
        "fda_ppb_frac": 0.995,
        "case17": False,
    },
    {
        "key": "Anidulafungin",
        "display_name": "Anidulafungin",
        "zh_name": "阿尼芬净",
        "aliases": ["anidulafungin", "eraxis", "ecalda", "阿尼芬净"],
        "category": "echinocandin",
        "drug_class": "Echinocandin Cyclic Lipopeptide (Terphenyl)",
        "sequence": None,
        "smiles": "CCCCCOC1=CC=C(C=C1)C2=CC=C(C=C2)C3=CC=C(C=C3)C(=O)N[C@H]4C[C@H]([C@H](NC(=O)[C@@H]5[C@H]([C@H](CN5C(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H]6C[C@H](CN6C(=O)[C@@H](NC4=O)[C@@H](C)O)O)[C@@H]([C@H](C7=CC=C(C=C7)O)O)O)[C@@H](C)O)C)O)O)O",
        "fda_half_life": "40–50 h terminal (FDA ERAXIS)",
        "fda_half_life_h": 45.0,
        "fda_ppb": ">99% (human plasma)",
        "fda_ppb_frac": 0.995,
        "case17": False,
    },
    {
        "key": "Rezafungin",
        "display_name": "Rezafungin",
        "zh_name": "瑞扎芬净",
        "aliases": ["rezafungin", "rezzayo", "cd101", "瑞扎芬净"],
        "category": "echinocandin",
        "drug_class": "Weekly Echinocandin Cyclic Lipopeptide (Choline Ether)",
        "sequence": None,
        "smiles": "CCCCCOC1=CC=C(C=C1)C2=CC=C(C=C2)C3=CC=C(C=C3)C(=O)N[C@H]4C[C@H]([C@H](NC(=O)[C@@H]5[C@H]([C@H](CN5C(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H]6C[C@H](CN6C(=O)[C@@H](NC4=O)[C@@H](C)O)O)[C@@H]([C@H](C7=CC=C(C=C7)O)O)O)[C@@H](C)O)C)O)OCC[N+](C)(C)C)O",
        "fda_half_life": "~152 h (~6.3 days, weekly, FDA REZZAYO)",
        "fda_half_life_h": 152.0,
        "fda_ppb": "95.6–98.6% (healthy adults)",
        "fda_ppb_frac": 0.97,
        "case17": False,
    },
    # ---- 3. Cyclic Macrocycles, Disulfide Peptides & Lipopeptides (Case17) ----
    {
        "key": "Cyclosporine_A",
        "display_name": "Cyclosporine A",
        "zh_name": "环孢素 A",
        "aliases": ["cyclosporinea", "cyclosporine", "cyclosporin", "csa", "sandimmune", "neoral", "环孢素", "环孢素a"],
        "category": "cyclic_macrocycle",
        "drug_class": "N-Methylated Cyclic Undecapeptide",
        "sequence": None,
        "smiles": "CC[C@H]1C(=O)N(CC(=O)N([C@H](C(=O)N[C@H](C(=O)N([C@H](C(=O)N[C@H](C(=O)N[C@@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N1)[C@@H]([C@H](C)C/C=C/C)O)C)C(C)C)C)CC(C)C)C)CC(C)C)C)C)C)CC(C)C)C)C(C)C)CC(C)C)C)C",
        "fda_half_life": "~8.4 h (5–18 h, FDA NEORAL)",
        "fda_half_life_h": 8.4,
        "fda_ppb": "~90% (lipoproteins)",
        "fda_ppb_frac": 0.90,
        "case17": True,
    },
    {
        "key": "Daptomycin",
        "display_name": "Daptomycin",
        "zh_name": "达托霉素",
        "aliases": ["daptomycin", "cubicin", "达托霉素"],
        "category": "cyclic_macrocycle",
        "drug_class": "Cyclic Lipopeptide Antibiotic (C10)",
        "sequence": None,
        "smiles": "CCCCCCCCCC(=O)N[C@@H](CC1=CNC2=CC=CC=C21)C(=O)N[C@H](CC(=O)N)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@H]3[C@H](OC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@H](NC(=O)CNC(=O)[C@@H](NC(=O)[C@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)CNC3=O)CCCN)CC(=O)O)C)CC(=O)O)CO)[C@H](C)CC(=O)O)CC(=O)C4=CC=CC=C4N)C",
        "fda_half_life": "~8–9 h (FDA CUBICIN)",
        "fda_half_life_h": 8.5,
        "fda_ppb": "90–93% (~92%)",
        "fda_ppb_frac": 0.92,
        "case17": True,
    },
    {
        "key": "Octreotide",
        "display_name": "Octreotide",
        "zh_name": "奥曲肽",
        "aliases": ["octreotide", "sandostatin", "奥曲肽"],
        "category": "cyclic_macrocycle",
        "drug_class": "Disulfide-Cyclic Octapeptide",
        "sequence": None,
        "smiles": "C[C@H]([C@H]1C(=O)N[C@@H](CSSC[C@@H](C(=O)N[C@H](C(=O)N[C@@H](C(=O)N[C@H](C(=O)N1)CCCCN)CC2=CNC3=CC=CC=C32)CC4=CC=CC=C4)NC(=O)[C@@H](CC5=CC=CC=C5)N)C(=O)N[C@H](CO)[C@@H](C)O)O",
        "fda_half_life": "1.7–1.9 h (FDA SANDOSTATIN)",
        "fda_half_life_h": 1.8,
        "fda_ppb": "~65% (lipoprotein + albumin)",
        "fda_ppb_frac": 0.65,
        "case17": True,
    },
    {
        "key": "Oxytocin",
        "display_name": "Oxytocin",
        "zh_name": "缩宫素",
        "aliases": ["oxytocin", "pitocin", "缩宫素"],
        "category": "cyclic_macrocycle",
        "drug_class": "Disulfide-Cyclic Nonapeptide",
        "sequence": None,
        "smiles": "CC[C@H](C)[C@H]1C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@@H](CSSC[C@@H](C(=O)N[C@H](C(=O)N1)CC2=CC=C(C=C2)O)N)C(=O)N3CCC[C@H]3C(=O)N[C@@H](CC(C)C)C(=O)NCC(=O)N)CC(=O)N)CCC(=O)N",
        "fda_half_life": "3–5 min (0.07 h, FDA PITOCIN)",
        "fda_half_life_h": 0.07,
        "fda_ppb": "Low / not quantified",
        "fda_ppb_frac": 0.30,
        "case17": True,
    },
    {
        "key": "Linaclotide",
        "display_name": "Linaclotide",
        "zh_name": "利那洛肽",
        "aliases": ["linaclotide", "linzess", "利那洛肽"],
        "category": "cyclic_macrocycle",
        "drug_class": "Triple-Disulfide Knotted Peptide",
        "sequence": None,
        "smiles": "C[C@H]1C(=O)N[C@H]2CSSC[C@H]3C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@@H](CSSC[C@H](NC(=O)CNC(=O)[C@@H](NC2=O)[C@@H](C)O)C(=O)N[C@@H](CC4=CC=C(C=C4)O)C(=O)O)C(=O)N[C@@H](CSSC[C@@H](C(=O)N3)N)C(=O)N[C@H](C(=O)N5CCC[C@H]5C(=O)N1)CC(=O)N)CC6=CC=C(C=C6)O)CCC(=O)O",
        "fda_half_life": "Locally GI-restricted (<LLOQ plasma)",
        "fda_half_life_h": None,
        "fda_ppb": "N/A (minimal systemic absorption)",
        "fda_ppb_frac": None,
        "case17": True,
    },
    {
        "key": "Ziconotide",
        "display_name": "Ziconotide",
        "zh_name": "齐考诺肽",
        "aliases": ["ziconotide", "prialt", "齐考诺肽"],
        "category": "cyclic_macrocycle",
        "drug_class": "Cystine-Knot Conotoxin (3× S–S)",
        "sequence": None,
        "smiles": "C[C@H]1C(=O)N[C@H](C(=O)N[C@H]2CSSC[C@H]3C(=O)N[C@H](C(=O)N[C@H](C(=O)NCC(=O)N[C@H](C(=O)N[C@@H](CSSC[C@@H](C(=O)N[C@@H](CSSC[C@@H](C(=O)N[C@H](C(=O)NCC(=O)N[C@H](C(=O)NCC(=O)N1)CCCCN)CCCCN)N)C(=O)N[C@H](C(=O)NCC(=O)N[C@H](C(=O)N3)CO)[C@@H](C)O)NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC2=O)CO)CCCNC(=N)N)CC(C)C)CCSC)CC4=CC=C(C=C4)O)CC(=O)O)C(=O)N)CCCCN)CO)CCCNC(=N)N)CCCCN",
        "fda_half_life": "CSF ~4.6 h; plasma ~1.3 h (FDA PRIALT)",
        "fda_half_life_h": 4.6,
        "fda_ppb": "~50%",
        "fda_ppb_frac": 0.50,
        "case17": True,
    },
    # ---- 4. Linear Clinical Peptide Therapeutics (Case17) ----
    {
        "key": "Exenatide",
        "display_name": "Exenatide",
        "zh_name": "艾塞那肽",
        "aliases": ["exenatide", "byetta", "bydureon", "艾塞那肽"],
        "category": "linear_clinical",
        "drug_class": "Linear GLP-1 Agonist (39 AA)",
        "sequence": "HGEGTFTSDLSKQMEEEAVRLFIEWLKNGGPSSGAPPPS",
        "smiles": "CC[C@H](C)[C@@H](C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CC1=CNC2=CC=CC=C21)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CC(=O)N)C(=O)NCC(=O)NCC(=O)N3CCC[C@H]3C(=O)N[C@@H](CO)C(=O)N[C@@H](CO)C(=O)NCC(=O)N[C@@H](C)C(=O)N4CCC[C@H]4C(=O)N5CCC[C@H]5C(=O)N6CCC[C@H]6C(=O)N[C@@H](CO)C(=O)N)NC(=O)[C@H](CC7=CC=CC=C7)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](C(C)C)NC(=O)[C@H](C)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCSC)NC(=O)[C@H](CCC(=O)N)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CO)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CO)NC(=O)[C@H]([C@@H](C)O)NC(=O)[C@H](CC8=CC=CC=C8)NC(=O)[C@H]([C@@H](C)O)NC(=O)CNC(=O)[C@H](CCC(=O)O)NC(=O)CNC(=O)[C@H](CC9=CNC=N9)N",
        "fda_half_life": "2.4 h terminal (FDA BYETTA)",
        "fda_half_life_h": 2.4,
        "fda_ppb": "~53%",
        "fda_ppb_frac": 0.53,
        "case17": True,
    },
    {
        "key": "Leuprolide",
        "display_name": "Leuprolide",
        "zh_name": "亮丙瑞林",
        "aliases": ["leuprolide", "leuprorelin", "lupron", "亮丙瑞林"],
        "category": "linear_clinical",
        "drug_class": "GnRH Analog Nonapeptide (N-Ethylamide)",
        "sequence": None,
        "smiles": "CCNC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCN=C(N)N)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](CC(C)C)NC(=O)[C@H](CC2=CC=C(C=C2)O)NC(=O)[C@H](CO)NC(=O)[C@H](CC3=CNC4=CC=CC=C43)NC(=O)[C@H](CC5=CN=CN5)NC(=O)[C@@H]6CCC(=O)N6",
        "fda_half_life": "~3 h (2.9–3 h, FDA LUPRON)",
        "fda_half_life_h": 3.0,
        "fda_ppb": "43–49%",
        "fda_ppb_frac": 0.46,
        "case17": True,
    },
    {
        "key": "Bivalirudin",
        "display_name": "Bivalirudin",
        "zh_name": "比伐卢定",
        "aliases": ["bivalirudin", "angiomax", "比伐卢定"],
        "category": "linear_clinical",
        "drug_class": "Direct Thrombin Inhibitor (20 AA)",
        "sequence": None,
        "smiles": "CC[C@H](C)[C@@H](C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CC2=CC=C(C=C2)O)C(=O)N[C@@H](CC(C)C)C(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CC3=CC=CC=C3)NC(=O)[C@H](CC(=O)O)NC(=O)CNC(=O)[C@H](CC(=O)N)NC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)[C@@H]4CCCN4C(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H]5CCCN5C(=O)[C@@H](CC6=CC=CC=C6)N",
        "fda_half_life": "~25 min (0.42 h, FDA ANGIOMAX)",
        "fda_half_life_h": 0.42,
        "fda_ppb": "~0% (minimal binding)",
        "fda_ppb_frac": 0.02,
        "case17": True,
    },
    {
        "key": "Glucagon",
        "display_name": "Glucagon",
        "zh_name": "胰高血糖素",
        "aliases": ["glucagon", "glucagen", "胰高血糖素"],
        "category": "linear_clinical",
        "drug_class": "Canonical Linear Hormone (29 AA)",
        "sequence": "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT",
        "smiles": "C[C@H]([C@@H](C(=O)N[C@@H](CC1=CC=CC=C1)C(=O)N[C@@H]([C@@H](C)O)C(=O)N[C@@H](CO)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CC2=CC=C(C=C2)O)C(=O)N[C@@H](CO)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CC3=CC=C(C=C3)O)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CO)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](C)C(=O)N[C@@H](CCC(=O)N)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CC4=CC=CC=C4)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CCC(=O)N)C(=O)N[C@@H](CC5=CNC6=CC=CC=C65)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCSC)C(=O)N[C@@H](CC(=O)N)C(=O)N[C@@H]([C@@H](C)O)C(=O)O)NC(=O)CNC(=O)[C@H](CCC(=O)N)NC(=O)[C@H](CO)NC(=O)[C@H](CC7=CN=CN7)N)O",
        "fda_half_life": "8–18 min (0.2 h, FDA GlucaGen)",
        "fda_half_life_h": 0.2,
        "fda_ppb": "Low / non-acylated",
        "fda_ppb_frac": 0.15,
        "case17": True,
    },
    {
        "key": "Teriparatide",
        "display_name": "Teriparatide",
        "zh_name": "特立帕肽",
        "aliases": ["teriparatide", "forteo", "pth134", "特立帕肽"],
        "category": "linear_clinical",
        "drug_class": "PTH (1–34) Linear Peptide",
        "sequence": "SVSEIQLMHNLGKHLNSMERVEWLRKKLQDVHNF",
        "smiles": "CC[C@H](C)[C@@H](C(=O)N[C@@H](CCC(=O)N)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCSC)C(=O)N[C@@H](CC1=CNC=N1)C(=O)N[C@@H](CC(=O)N)C(=O)N[C@@H](CC(C)C)C(=O)NCC(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CC2=CNC=N2)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(=O)N)C(=O)N[C@@H](CO)C(=O)N[C@@H](CCSC)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CC3=CNC4=CC=CC=C43)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCC(=O)N)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CC5=CNC=N5)C(=O)N[C@@H](CC(=O)N)C(=O)N[C@@H](CC6=CC=CC=C6)C(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CO)NC(=O)[C@H](C(C)C)NC(=O)[C@H](CO)N",
        "fda_half_life": "IV ~5 min; SC ~1 h (FDA FORTEO)",
        "fda_half_life_h": 1.0,
        "fda_ppb": "~0% (negligible)",
        "fda_ppb_frac": 0.05,
        "case17": True,
    },
    # ---- 5. Classic Antimicrobial Peptides (Canonical Sequences) ----
    {
        "key": "Classic_AMP",
        "display_name": "Classic AMP (Benchmark)",
        "zh_name": "经典抗菌肽示例",
        "aliases": ["classicamp", "amp", "exampleamp", "经典抗菌肽", "示例抗菌肽"],
        "category": "amp_sequences",
        "drug_class": "Cationic Amphipathic AMP (23 AA)",
        "sequence": "HISEAVKIFGCKWKENWKWNSNC",
        "smiles": None,
        "fda_half_life": None,
        "fda_half_life_h": None,
        "fda_ppb": None,
        "fda_ppb_frac": None,
        "case17": False,
    },
    {
        "key": "Magainin_2",
        "display_name": "Magainin 2",
        "zh_name": "爪蟾抗菌肽 Magainin 2",
        "aliases": ["magainin2", "magainin", "爪蟾抗菌肽"],
        "category": "amp_sequences",
        "drug_class": "Amphipathic α-Helical AMP (23 AA)",
        "sequence": "GIGKFLHSAKKFGKAFVGEIMNS",
        "smiles": None,
        "fda_half_life": None,
        "fda_half_life_h": None,
        "fda_ppb": None,
        "fda_ppb_frac": None,
        "case17": False,
    },
    {
        "key": "Melittin",
        "display_name": "Melittin",
        "zh_name": "蜂毒肽 Melittin",
        "aliases": ["melittin", "蜂毒肽"],
        "category": "amp_sequences",
        "drug_class": "Hemolytic Amphipathic AMP (26 AA)",
        "sequence": "GIGAVLKVLTTGLPALISWIKRKRQQ",
        "smiles": None,
        "fda_half_life": None,
        "fda_half_life_h": None,
        "fda_ppb": None,
        "fda_ppb_frac": None,
        "case17": False,
    },
    {
        "key": "LL_37",
        "display_name": "LL-37 (Human Cathelicidin)",
        "zh_name": "人源抗菌肽 LL-37",
        "aliases": ["ll37", "cathelicidin", "ll-37"],
        "category": "amp_sequences",
        "drug_class": "Human Host-Defense AMP (37 AA)",
        "sequence": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
        "smiles": None,
        "fda_half_life": None,
        "fda_half_life_h": None,
        "fda_ppb": None,
        "fda_ppb_frac": None,
        "case17": False,
    },
    {
        "key": "Indolicidin",
        "display_name": "Indolicidin",
        "zh_name": "吲哚杀菌肽 Indolicidin",
        "aliases": ["indolicidin", "吲哚杀菌肽"],
        "category": "amp_sequences",
        "drug_class": "Trp/Pro-Rich Short AMP (13 AA)",
        "sequence": "ILPWKWPWWPWRR",
        "smiles": None,
        "fda_half_life": None,
        "fda_half_life_h": None,
        "fda_ppb": None,
        "fda_ppb_frac": None,
        "case17": False,
    },
]

_ALIAS_MAP: dict[str, dict[str, Any]] = {}
for _entry in DRUG_LIBRARY:
    norm_key = re.sub(r"[\s_\-]+", "", _entry["key"].lower())
    _ALIAS_MAP[norm_key] = _entry
    _ALIAS_MAP[re.sub(r"[\s_\-]+", "", _entry["display_name"].lower())] = _entry
    for _al in _entry["aliases"]:
        _ALIAS_MAP[re.sub(r"[\s_\-]+", "", _al.lower())] = _entry


def lookup_drug(query: str) -> Optional[dict[str, Any]]:
    """Lookup a curated drug or classic AMP by English/Chinese/brand name."""
    if not query:
        return None
    norm = re.sub(r"[\s_\-]+", "", query.strip().lower())
    return _ALIAS_MAP.get(norm)


QUICK_EXAMPLES: list[dict[str, str]] = [
    {
        "id": "semaglutide",
        "label": "Semaglutide",
        "tag": "C18 Diacid",
        "query": "Semaglutide",
        "subtitle": "司美格鲁肽 · Weekly GLP-1 RA · Albumin-binding",
    },
    {
        "id": "tirzepatide",
        "label": "Tirzepatide",
        "tag": "C20 Diacid",
        "query": "Tirzepatide",
        "subtitle": "替尔泊肽 · Dual GIP/GLP-1 · C20 Fatty Diacid",
    },
    {
        "id": "cyclosporine",
        "label": "Cyclosporine A",
        "tag": "Cyclic",
        "query": "Cyclosporine A",
        "subtitle": "环孢素 A · N-methylated 33-atom macrocycle",
    },
    {
        "id": "caspofungin",
        "label": "Caspofungin",
        "tag": "Echinocandin",
        "query": "Caspofungin",
        "subtitle": "卡泊芬净 · FDA Echinocandin Cyclic Lipopeptide",
    },
    {
        "id": "micafungin",
        "label": "Micafungin",
        "tag": "Echinocandin",
        "query": "Micafungin",
        "subtitle": "米卡芬净 · FDA Echinocandin Cyclic Lipopeptide",
    },
    {
        "id": "classic_amp",
        "label": "Classic AMP",
        "tag": "Sequence",
        "query": "HISEAVKIFGCKWKENWKWNSNC",
        "subtitle": "经典抗菌肽 · 23-AA canonical cationic peptide",
    },
]

BATCH_DEMO_SAMPLE = """name,query
Semaglutide,Semaglutide
Tirzepatide,Tirzepatide
Cyclosporine A,Cyclosporine A
Caspofungin,Caspofungin
Daptomycin,Daptomycin
Classic AMP,HISEAVKIFGCKWKENWKWNSNC
Liraglutide,Liraglutide
"""

BATCH_CASE17_SAMPLE = """name,query
Semaglutide,Semaglutide
Tirzepatide,Tirzepatide
Liraglutide,Liraglutide
Exenatide,Exenatide
Leuprolide,Leuprolide
Bivalirudin,Bivalirudin
Glucagon,Glucagon
Teriparatide,Teriparatide
Cyclosporine A,Cyclosporine A
Oxytocin,Oxytocin
Octreotide,Octreotide
Linaclotide,Linaclotide
Ziconotide,Ziconotide
Daptomycin,Daptomycin
Caspofungin,Caspofungin
Insulin detemir,Insulin detemir
Insulin degludec,Insulin degludec
"""

BATCH_ECHINOCANDIN_SAMPLE = """name,query
Caspofungin,Caspofungin
Micafungin,Micafungin
Anidulafungin,Anidulafungin
Rezafungin,Rezafungin
Cyclosporine A,Cyclosporine A
Daptomycin,Daptomycin
"""

BATCH_PRESETS: dict[str, dict[str, str]] = {
    "demo_7": {
        "label": "Mixed Benchmark (7)",
        "label_zh": "综合示例 (7)",
        "description": "7-peptide clinical, echinocandin & AMP benchmark",
        "description_zh": "7 个临床多肽、棘白菌素与抗菌肽综合示例",
        "csv": BATCH_DEMO_SAMPLE,
    },
    "case17_17": {
        "label": "17 Clinical Cases (17)",
        "label_zh": "17 个案例 (17)",
        "description": "17 classic clinical peptide cases",
        "description_zh": "17 个经典多肽药物案例",
        "csv": BATCH_CASE17_SAMPLE,
    },
    "echinocandin_6": {
        "label": "Echinocandins & Macrocycles (6)",
        "label_zh": "棘白菌素与大环肽 (6)",
        "description": "Echinocandin cyclic lipopeptides (4) + Cyclosporine A + Daptomycin",
        "description_zh": "4 个棘白菌素环肽 + 环孢素 A + 达托霉素",
        "csv": BATCH_ECHINOCANDIN_SAMPLE,
    },
}
