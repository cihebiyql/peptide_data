#!/usr/bin/env python3
"""Configuration, endpoint contracts, canonical clinical defaults, and Case17 metadata
for the Peptide OmniPanel V33 CPU modeling & calibration suite.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(os.environ.get("V33_BASE_DIR", str(Path(__file__).resolve().parent.parent)))
DATA_DIR = BASE_DIR / "data"
ENDPOINTS_DIR = DATA_DIR / "endpoints"
FEATURES_DIR = BASE_DIR / "features"
OUT_DIR = BASE_DIR / "models_cpu"

AUX_ENDPOINTS: List[str] = [
    "PAMPA",
    "Caco2",
    "LogD7.4",
    "solubility_cls",
    "hemolysis_cls",
]

PK_META_TARGETS: List[str] = [
    "PPB",
    "CL_per_kg",
    "CL_absolute",
    "Vd_per_kg",
    "F",
    "Cmax",
    "LogD7.4",
]

SEQ_ONLY_ENDPOINTS: List[str] = [
    # All 6 classification endpoints
    "BBB",
    "cytotoxicity_cls",
    "half_life_cls",
    "hemolysis_cls",
    "overall_peptide_toxicity",
    "solubility_cls",
    # 6 key sequence-heavy regression endpoints
    "half_life_reg",
    "hemolysis_HC50",
    "MIC_E_coli",
    "MIC_S_aureus",
    "MIC_P_aeruginosa",
    "MIC_C_albicans",
]

# All 24 endpoints with canonical human-clinical default condition contracts
ENDPOINT_SPECS: Dict[str, Dict[str, Any]] = {
    # --- Small-n (<300) PK & Permeability (9 endpoints) ---
    "CL_absolute": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(L/h)",
        "bounds": None,
        "canonical_conditions": {
            "cond_species": "human",
            "cond_route": "IV",
            "cond_estimand": "CL_total",
        },
        "alt_case17_conditions": {
            "SC_CL_over_F": {
                "cond_species": "human",
                "cond_route": "SC",
                "cond_estimand": "CL_over_F",
            }
        },
    },
    "Cmax": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(uM)",
        "bounds": None,
        "canonical_conditions": {
            "cond_species": "human",
            "cond_route": "IV",
            "cond_dose": "1_to_10_mg_kg",
        },
    },
    "F": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "fraction [0,1]",
        "bounds": (0.0, 1.0),
        "canonical_conditions": {
            "cond_species": "human",
            "cond_route": "PO",
        },
        "alt_case17_conditions": {
            "SC": {
                "cond_species": "human",
                "cond_route": "SC",
            }
        },
    },
    "PPB": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "fraction [0,1]",
        "bounds": (0.0, 1.0),
        "canonical_conditions": {
            "cond_species": "human",
            "cond_matrix": "plasma",
        },
    },
    "Vd_per_kg": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(L/kg)",
        "bounds": None,
        "canonical_conditions": {
            "cond_species": "human",
            "cond_route": "IV",
            "cond_estimand": "Vss",
        },
    },
    "RRCK": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(cm/s)",
        "bounds": None,
        "canonical_conditions": {
            "cond_estimand": "unspecified",
        },
    },
    "MDCK": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(cm/s)",
        "bounds": None,
        "canonical_conditions": {
            "cond_assay": "mdck_wt",
            "cond_estimand": "AB",
        },
    },
    "CL_per_kg": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(mL/min/kg)",
        "bounds": None,
        "canonical_conditions": {
            "cond_species": "human",
            "cond_route": "IV",
            "cond_estimand": "CL_total",
        },
    },
    "solubility_reg": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(uM)",
        "bounds": None,
        "canonical_conditions": {
            "cond_solvent": "pbs_ph74",
            "cond_assay": "thermodynamic",
        },
    },
    # --- Medium/Large-n Regression (9 endpoints) ---
    "LogD7.4": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "LogD7.4",
        "bounds": None,
        "canonical_conditions": {
            "cond_assay": "shake_flask_octanol_ph74",
        },
    },
    "Caco2": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(cm/s)",
        "bounds": None,
        "canonical_conditions": {
            "cond_estimand": "AB",
        },
    },
    "PAMPA": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "large",
        "unit": "log10(cm/s)",
        "bounds": None,
        "canonical_conditions": {
            "cond_assay": "pampa_generic",
            "cond_estimand": "unspecified",
        },
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
        "canonical_conditions": {
            "cond_species": "human",
        },
    },
    "MIC_C_albicans": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "bounds": None,
        "canonical_conditions": {
            "cond_strain": "ATCC_ref_candida",
            "cond_assay": "RPMI",
        },
    },
    "MIC_P_aeruginosa": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "bounds": None,
        "canonical_conditions": {
            "cond_strain": "ATCC_27853",
            "cond_assay": "MHB",
        },
    },
    "MIC_S_aureus": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "bounds": None,
        "outlier_winsor_cap": 5.0,
        "canonical_conditions": {
            "cond_strain": "ATCC_25923",
            "cond_assay": "MHB",
        },
    },
    "MIC_E_coli": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "bounds": None,
        "canonical_conditions": {
            "cond_strain": "ATCC_25922",
            "cond_assay": "MHB",
        },
    },
    # --- Classification (6 endpoints) ---
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
        "canonical_conditions": {
            "cond_species": "human",
            "cond_matrix": "normal_mammalian",
        },
    },
    "half_life_cls": {
        "tier": "tier1",
        "task_type": "classification",
        "size_tier": "medium",
        "unit": "prob [0,1]",
        "bounds": (0.0, 1.0),
        "canonical_conditions": {
            "cond_species": "human",
            "cond_matrix": "in_vivo_plasma",
            "cond_route": "IV",
        },
    },
    "solubility_cls": {
        "tier": "tier1",
        "task_type": "classification",
        "size_tier": "medium",
        "unit": "prob [0,1]",
        "bounds": (0.0, 1.0),
        "canonical_conditions": {
            "cond_solvent": "pbs_ph74",
        },
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
        "canonical_conditions": {
            "cond_species": "human",
        },
    },
}

# Clinical route for Case17 drugs (for route-conditioned half-life & F comparisons)
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

# Audited FDA numeric reference anchors for Case17 evaluation (None where unquantified/local)
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


def compute_halflife_mechanism_anchor(smiles: str, exclude_drug: Optional[str] = None) -> Dict[str, Any]:
    """Audited literature-anchored fatty-acylation half-life mechanism rule (strictly fatty-acylated classes only).

    Delegates to strict unbranched saturated RDKit SMARTS (no SMILES regex, no Fc/albumin
    fusion proteins) with true dynamic Leave-One-Drug-Out (`exclude_drug`):
      - unbranched C18/C20 fatty-diacid amide on therapeutic peptide (MW>=1500, >=8 amides):
        weekly albumin-binding class (Semaglutide 168 h, Tirzepatide 120 h, Cagrilintide 170 h)
      - unbranched C14-C16 fatty-diacid amide: depot class (Insulin degludec 25 h; None if excluded)
      - unbranched C14-C16 monoacyl-Lys/gammaGlu: daily class (Insulin detemir 6 h, Liraglutide 13 h)
      - non-acylated: None (use Pure-ML prediction)
    """
    from rdkit import Chem
    from v33_upgrade.web.chem_utils import detect_acylation_anchor

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "anchor_class": "parse_error",
            "diacid_internal_c": 0,
            "monoacyl_linear_c": 0,
            "anchor_mid_h": None,
            "anchor_range_h": None,
        }
    anc = detect_acylation_anchor(mol, exclude_drug=exclude_drug)
    if not anc.detected:
        return {
            "anchor_class": "none",
            "diacid_internal_c": 0,
            "monoacyl_linear_c": 0,
            "anchor_mid_h": None,
            "anchor_range_h": None,
        }
    return {
        "anchor_class": anc.acylation_class or "none",
        "diacid_internal_c": (anc.chain_carbons or 0) - 2 if "diacid" in (anc.acylation_class or "") else 0,
        "monoacyl_linear_c": (anc.chain_carbons or 0) if "acyl" in (anc.acylation_class or "") else 0,
        "anchor_mid_h": anc.value_h,
        "anchor_range_h": anc.range_h,
    }
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
