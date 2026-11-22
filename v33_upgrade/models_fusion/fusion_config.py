#!/usr/bin/env python3
"""Configuration, endpoint contracts, embedding specifications, and Case17 metadata
for the Peptide OmniPanel V33 Wave-2D Multimodal Fusion & Representation Ablation suite.

Strictly owns and writes ONLY under /data/qlyu/v33_upgrade/models_fusion/.
Read-only access to data/, features/, embeddings/, and models_cpu/.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(os.environ.get("V33_BASE_DIR", str(Path(__file__).resolve().parent.parent)))
DATA_DIR = BASE_DIR / "data"
ENDPOINTS_DIR = DATA_DIR / "endpoints"
FEATURES_DIR = BASE_DIR / "features"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
CPU_MODELS_DIR = BASE_DIR / "models_cpu"
OUT_DIR = BASE_DIR / "models_fusion"

# The 9 primary censored AFT targets named in Wave-2D (plus we run AFT on ALL endpoints with censored CV rows!)
KEY_CENSORED_TARGETS: List[str] = [
    "hemolysis_HC50",
    "MIC_E_coli",
    "MIC_S_aureus",
    "MIC_P_aeruginosa",
    "MIC_C_albicans",
    "half_life_reg",
    "PPB",
    "solubility_reg",
    "F",
]

# Pretrained individual representations benchmarked fairly across all 24 endpoints
INDIVIDUAL_REPS: List[Dict[str, Any]] = [
    {
        "name": "ncaa_smiles_repr",
        "label": "ncAA-RepDistill v2 (smiles_repr 768D)",
        "track": "smiles_only",
        "blocks": [("ncaa_smiles_repr", 1.0)],
        "phys_mode": "smiles48",
        "pca_group_dims": [("ncaa_smiles_repr", 64)],
    },
    {
        "name": "ncaa_fusion_hf",
        "label": "ncAA-RepDistill v2 (fusion_hf 2048D)",
        "track": "smiles_only",
        "blocks": [("ncaa_fusion_hf", 1.0)],
        "phys_mode": "smiles48",
        "pca_group_dims": [("ncaa_fusion_hf", 96)],
    },
    {
        "name": "ncaa_protein_proj",
        "label": "ncAA-RepDistill v2 (protein_proj 1280D)",
        "track": "smiles_only",
        "blocks": [("ncaa_protein_proj", 1.0)],
        "phys_mode": "smiles48",
        "pca_group_dims": [("ncaa_protein_proj", 64)],
    },
    {
        "name": "peptideclm2",
        "label": "PeptideCLM-2 (512D)",
        "track": "smiles_only",
        "blocks": [("peptideclm2", 1.0)],
        "phys_mode": "smiles48",
        "pca_group_dims": [("peptideclm2", 64)],
    },
    {
        "name": "chemberta_mtr",
        "label": "ChemBERTa-MTR (384D)",
        "track": "smiles_only",
        "blocks": [("chemberta_mtr", 1.0)],
        "phys_mode": "smiles48",
        "pca_group_dims": [("chemberta_mtr", 64)],
    },
    {
        "name": "esmc_mean",
        "label": "ESM-C 600M (mean 1152D)",
        "track": "dual_modality",
        "blocks": [("esmc_mean", 1.0)],
        "phys_mode": "dual83",
        "pca_group_dims": [("esmc_mean", 96)],
    },
    {
        "name": "esmc_multiregion",
        "label": "ESM-C 600M Multi-Region (mean+max+N5+C5 4608D)",
        "track": "dual_modality",
        "blocks": [
            ("esmc_mean", 0.50),
            ("esmc_max", 0.50),
            ("esmc_nterm5", 0.50),
            ("esmc_cterm5", 0.50),
        ],
        "phys_mode": "dual83",
        "pca_group_dims": [
            ("esmc_mean", 32),
            ("esmc_max", 32),
            ("esmc_nterm5", 32),
            ("esmc_cterm5", 32),
        ],
    },
]

# Direct multimodal fusion representations (compact PCA-GBDT + block-balanced Kernel/Linear/MLP)
DIRECT_FUSION_REPS: Dict[str, Dict[str, Any]] = {
    "direct_smiles": {
        "name": "direct_smiles",
        "label": "Direct Multimodal SMILES (ncAA + PeptideCLM-2 + ChemBERTa + physchem48)",
        "track": "smiles_only",
        "blocks": [
            ("ncaa_smiles_repr", 0.65),
            ("ncaa_fusion_hf", 0.65),
            ("ncaa_protein_proj", 0.45),
            ("peptideclm2", 0.90),
            ("chemberta_mtr", 0.80),
        ],
        "phys_mode": "smiles48",
        "pca_group_dims": [
            ("ncaa_smiles_repr", 32),
            ("ncaa_fusion_hf", 32),
            ("peptideclm2", 64),
            ("chemberta_mtr", 64),
        ],
    },
    "direct_dual": {
        "name": "direct_dual",
        "label": "Direct Multimodal Dual (ncAA + PeptideCLM-2 + ChemBERTa + ESM-C MR + physchem83)",
        "track": "dual_modality",
        "blocks": [
            ("ncaa_smiles_repr", 0.60),
            ("ncaa_fusion_hf", 0.60),
            ("ncaa_protein_proj", 0.40),
            ("peptideclm2", 0.85),
            ("chemberta_mtr", 0.70),
            ("esmc_mean", 0.65),
            ("esmc_max", 0.65),
            ("esmc_nterm5", 0.60),
            ("esmc_cterm5", 0.60),
        ],
        "phys_mode": "dual83",
        "pca_group_dims": [
            ("ncaa_smiles_repr", 32),
            ("ncaa_fusion_hf", 32),
            ("peptideclm2", 48),
            ("chemberta_mtr", 32),
            ("esmc_mean", 32),
            ("esmc_max", 32),
            ("esmc_nterm5", 32),
            ("esmc_cterm5", 32),
        ],
    },
}

# Exact endpoint specifications matching Wave-2C (v33_config.py)
ENDPOINT_SPECS: Dict[str, Dict[str, Any]] = {
    "CL_absolute": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(L/h)",
        "bounds": None,
        "canonical_conditions": {"cond_species": "human", "cond_route": "IV", "cond_estimand": "CL_total"},
        "alt_case17_conditions": {
            "SC_CL_over_F": {"cond_species": "human", "cond_route": "SC", "cond_estimand": "CL_over_F"}
        },
    },
    "Cmax": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(uM)",
        "bounds": None,
        "canonical_conditions": {"cond_species": "human", "cond_route": "IV", "cond_dose": "1_to_10_mg_kg"},
    },
    "F": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "fraction [0,1]",
        "bounds": (0.0, 1.0),
        "canonical_conditions": {"cond_species": "human", "cond_route": "PO"},
        "alt_case17_conditions": {"SC": {"cond_species": "human", "cond_route": "SC"}},
    },
    "PPB": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "fraction [0,1]",
        "bounds": (0.0, 1.0),
        "canonical_conditions": {"cond_species": "human", "cond_matrix": "plasma"},
    },
    "Vd_per_kg": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(L/kg)",
        "bounds": None,
        "canonical_conditions": {"cond_species": "human", "cond_route": "IV", "cond_estimand": "Vss"},
    },
    "RRCK": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(cm/s)",
        "bounds": None,
        "canonical_conditions": {"cond_estimand": "unspecified"},
    },
    "MDCK": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(cm/s)",
        "bounds": None,
        "canonical_conditions": {"cond_assay": "mdck_wt", "cond_estimand": "AB"},
    },
    "CL_per_kg": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(mL/min/kg)",
        "bounds": None,
        "canonical_conditions": {"cond_species": "human", "cond_route": "IV", "cond_estimand": "CL_total"},
    },
    "solubility_reg": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(uM)",
        "bounds": None,
        "canonical_conditions": {"cond_solvent": "pbs_ph74", "cond_assay": "thermodynamic"},
    },
    "LogD7.4": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "LogD7.4",
        "bounds": None,
        "canonical_conditions": {"cond_assay": "shake_flask_octanol_ph74"},
    },
    "Caco2": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(cm/s)",
        "bounds": None,
        "canonical_conditions": {"cond_estimand": "AB"},
    },
    "PAMPA": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "large",
        "unit": "log10(cm/s)",
        "bounds": None,
        "canonical_conditions": {"cond_assay": "pampa_generic", "cond_estimand": "unspecified"},
    },
    "half_life_reg": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(h)",
        "bounds": None,
        "canonical_conditions": {
            "cond_species": "human",
            "cond_matrix": "in_vivo_plasma",
            "cond_route": "IV",
            "cond_estimand": "t1_2_in_vivo",
        },
        "alt_case17_conditions": {
            "SC": {
                "cond_species": "human",
                "cond_matrix": "in_vivo_plasma",
                "cond_route": "SC",
                "cond_estimand": "t1_2_in_vivo",
            }
        },
    },
    "hemolysis_HC50": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "large",
        "unit": "log10(uM)",
        "bounds": None,
        "canonical_conditions": {"cond_species": "human"},
    },
    "MIC_C_albicans": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "bounds": None,
        "canonical_conditions": {"cond_strain": "ATCC_ref_candida", "cond_assay": "RPMI"},
    },
    "MIC_P_aeruginosa": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "bounds": None,
        "canonical_conditions": {"cond_strain": "ATCC_27853", "cond_assay": "MHB"},
    },
    "MIC_S_aureus": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "bounds": None,
        "outlier_winsor_cap": 5.0,
        "canonical_conditions": {"cond_strain": "ATCC_25923", "cond_assay": "MHB"},
    },
    "MIC_E_coli": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "bounds": None,
        "canonical_conditions": {"cond_strain": "ATCC_25922", "cond_assay": "MHB"},
    },
    "BBB": {
        "tier": "tier1",
        "task_type": "classification",
        "size_tier": "medium",
        "unit": "prob [0,1]",
        "bounds": (0.0, 1.0),
        "canonical_conditions": {},
    },
    "cytotoxicity_cls": {
        "tier": "tier1",
        "task_type": "classification",
        "size_tier": "medium",
        "unit": "prob [0,1]",
        "bounds": (0.0, 1.0),
        "canonical_conditions": {"cond_species": "human", "cond_matrix": "normal_mammalian"},
    },
    "half_life_cls": {
        "tier": "tier1",
        "task_type": "classification",
        "size_tier": "medium",
        "unit": "prob [0,1]",
        "bounds": (0.0, 1.0),
        "canonical_conditions": {"cond_species": "human", "cond_matrix": "in_vivo_plasma", "cond_route": "IV"},
    },
    "solubility_cls": {
        "tier": "tier1",
        "task_type": "classification",
        "size_tier": "medium",
        "unit": "prob [0,1]",
        "bounds": (0.0, 1.0),
        "canonical_conditions": {"cond_solvent": "pbs_ph74"},
    },
    "overall_peptide_toxicity": {
        "tier": "tier1",
        "task_type": "classification",
        "size_tier": "large",
        "unit": "prob [0,1]",
        "bounds": (0.0, 1.0),
        "canonical_conditions": {},
    },
    "hemolysis_cls": {
        "tier": "tier1",
        "task_type": "classification",
        "size_tier": "large",
        "unit": "prob [0,1]",
        "bounds": (0.0, 1.0),
        "canonical_conditions": {"cond_species": "human"},
    },
}

CASE17_CLINICAL_ROUTE: Dict[str, str] = {
    "Leuprolide": "SC",
    "Bivalirudin": "IV",
    "Glucagon": "IV",
    "Teriparatide": "SC",
    "Exenatide": "SC",
    "Liraglutide": "SC",
    "Semaglutide": "SC",
    "Tirzepatide": "SC",
    "Cyclosporine_A": "PO",
    "Oxytocin": "IV",
    "Octreotide": "SC",
    "Linaclotide": "PO",
    "Ziconotide": "IV",
    "Daptomycin": "IV",
    "Caspofungin": "IV",
    "Insulin_detemir": "SC",
    "Insulin_degludec": "SC",
}

CASE17_FDA_NUMERIC: Dict[str, Dict[str, Optional[float]]] = {
    "Leuprolide": {"half_life_h": 3.0, "ppb": 0.46},
    "Bivalirudin": {"half_life_h": 0.42, "ppb": 0.0},
    "Glucagon": {"half_life_h": 0.22, "ppb": None},
    "Teriparatide": {"half_life_h": 1.0, "ppb": 0.0},
    "Exenatide": {"half_life_h": 2.4, "ppb": 0.53},
    "Liraglutide": {"half_life_h": 13.0, "ppb": 0.985},
    "Semaglutide": {"half_life_h": 168.0, "ppb": 0.995},
    "Tirzepatide": {"half_life_h": 125.0, "ppb": 0.96},
    "Cyclosporine_A": {"half_life_h": 8.4, "ppb": 0.90},
    "Oxytocin": {"half_life_h": 0.067, "ppb": None},
    "Octreotide": {"half_life_h": 1.8, "ppb": 0.65},
    "Linaclotide": {"half_life_h": None, "ppb": None},
    "Ziconotide": {"half_life_h": 4.6, "ppb": 0.50},
    "Daptomycin": {"half_life_h": 8.5, "ppb": 0.92},
    "Caspofungin": {"half_life_h": 10.0, "ppb": 0.97},
    "Insulin_detemir": {"half_life_h": 6.0, "ppb": 0.985},
    "Insulin_degludec": {"half_life_h": 25.0, "ppb": 0.99},
}


def compute_halflife_mechanism_anchor(smiles: str) -> Dict[str, Any]:
    """Audited literature-anchored fatty-acylation half-life mechanism rule (identical to V32/Wave-2C)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "anchor_class": "parse_error",
            "diacid_internal_c": 0,
            "monoacyl_linear_c": 0,
            "anchor_mid_h": None,
            "anchor_range_h": None,
        }
    diacid = 0
    for length in range(22, 10, -1):
        patt = Chem.MolFromSmarts("C(=O)" + "C" * length + "C(=O)")
        if patt is not None and mol.HasSubstructMatch(patt):
            diacid = length
            break
    can = Chem.MolToSmiles(mol)
    runs = re.findall(r"C{10,}", can)
    mono = max((len(r) for r in runs), default=0)

    if diacid >= 16:
        return {
            "anchor_class": "weekly_diacid",
            "diacid_internal_c": diacid,
            "monoacyl_linear_c": mono,
            "anchor_mid_h": 144.0,
            "anchor_range_h": [120.0, 168.0],
        }
    if diacid >= 12:
        return {
            "anchor_class": "ultralong_depot_diacid",
            "diacid_internal_c": diacid,
            "monoacyl_linear_c": mono,
            "anchor_mid_h": 25.0,
            "anchor_range_h": [25.0, 25.0],
        }
    if mono >= 14:
        return {
            "anchor_class": "daily_acyl",
            "diacid_internal_c": diacid,
            "monoacyl_linear_c": mono,
            "anchor_mid_h": 9.0,
            "anchor_range_h": [5.0, 13.0],
        }
    return {
        "anchor_class": "non_acylated",
        "diacid_internal_c": diacid,
        "monoacyl_linear_c": mono,
        "anchor_mid_h": None,
        "anchor_range_h": None,
    }
