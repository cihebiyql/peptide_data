#!/usr/bin/env python3
"""Wave 5A — V33.1 Endpoint Specifications, Display Metadata, and Dynamic Feature/Fusion Banks
(for N = 46,598 + K compounds) compatible 100% with V33 models_cpu & models_fusion classes.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

csv.field_size_limit(10_000_000)

V33_BASE = Path("<WORK_DIR>/v33_upgrade")
V33_DATA = V33_BASE / "data"
EXP_BASE = V33_BASE / "expansion_v331"
EXP_DATA = EXP_BASE / "data"
EXP_ENDPOINTS_DIR = EXP_DATA / "endpoints"
EXP_FEATURES_DIR = EXP_BASE / "features"
EXP_EMBEDDINGS_DIR = EXP_BASE / "embeddings"
EXP_CPU_DIR = EXP_BASE / "models_cpu"
EXP_FUSION_DIR = EXP_BASE / "models_fusion"

V331_ENDPOINT_SPECS: Dict[str, Dict[str, Any]] = {
    # --- Group A: Tier 1 ADMET & Safety (4 admitted; LogP honestly rejected at 39/40 rows) ---
    "cytotoxicity_CC50": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "safety",
        "display_en": "Mammalian Cytotoxicity CC50",
        "display_zh": "哺乳动物细胞毒性 CC50",
        "bounds": (-4.0, 5.5),
        "outlier_winsor_cap": 5.0,
        "canonical_conditions": {
            "cond_matrix": "normal_mammalian",
            "cond_species": "human",
            "cond_estimand": "CC50",
        },
    },
    "hemolysis_MHC": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "safety",
        "display_en": "Minimum Hemolytic Concentration (MHC/HC10)",
        "display_zh": "最小溶血浓度 MHC/HC10",
        "bounds": (-3.0, 5.5),
        "canonical_conditions": {
            "cond_species": "human",
            "cond_estimand": "MHC",
        },
    },
    "efflux_ratio": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(ER)",
        "linear_unit": "ratio",
        "log10_space": True,
        "domain": "permeability",
        "display_en": "Caco-2/MDCK Efflux Ratio (P-gp)",
        "display_zh": "外排比 Efflux Ratio (P-gp)",
        "bounds": (-2.0, 3.0),
        "canonical_conditions": {
            "cond_assay": "caco2",
        },
    },
    "Vd_absolute": {
        "tier": "tier1",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(L)",
        "linear_unit": "L",
        "log10_space": True,
        "domain": "pk",
        "display_en": "Absolute Volume of Distribution (Vd)",
        "display_zh": "绝对分布容积 Vd",
        "bounds": (-3.0, 5.0),
        "canonical_conditions": {
            "cond_species": "human",
            "cond_route": "IV",
            "cond_estimand": "Vss",
        },
        "alt_case17_conditions": {
            "SC_V_F": {
                "cond_species": "human",
                "cond_route": "SC",
                "cond_estimand": "V_F",
            }
        },
    },
    # --- Group B: Tier 2 New Species-Specific MIC (9 admitted) ---
    "MIC_B_subtilis": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "antimicrobial",
        "display_en": "B. subtilis MIC",
        "display_zh": "枯草芽孢杆菌 MIC (革兰氏阳性)",
        "bounds": (-4.0, 5.5),
        "canonical_conditions": {"cond_strain": "ATCC_ref", "cond_assay": "MHB"},
    },
    "MIC_A_baumannii": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "antimicrobial",
        "display_en": "A. baumannii MIC",
        "display_zh": "鲍曼不动杆菌 MIC (ESKAPE 革兰氏阴性)",
        "bounds": (-4.0, 5.5),
        "canonical_conditions": {"cond_strain": "ATCC_ref", "cond_assay": "MHB"},
    },
    "MIC_K_pneumoniae": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "antimicrobial",
        "display_en": "K. pneumoniae MIC",
        "display_zh": "肺炎克雷伯菌 MIC (ESKAPE 革兰氏阴性)",
        "bounds": (-4.0, 5.5),
        "canonical_conditions": {"cond_strain": "ATCC_ref", "cond_assay": "MHB"},
    },
    "MIC_S_epidermidis": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "antimicrobial",
        "display_en": "S. epidermidis MIC",
        "display_zh": "表皮葡萄球菌 MIC (革兰氏阳性)",
        "bounds": (-4.0, 5.5),
        "canonical_conditions": {"cond_strain": "ATCC_ref", "cond_assay": "MHB"},
    },
    "MIC_Salmonella": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "antimicrobial",
        "display_en": "Salmonella spp. MIC",
        "display_zh": "沙门氏菌 MIC (革兰氏阴性)",
        "bounds": (-4.0, 5.5),
        "canonical_conditions": {"cond_strain": "ATCC_ref", "cond_assay": "MHB"},
    },
    "MIC_E_faecalis": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "antimicrobial",
        "display_en": "E. faecalis MIC",
        "display_zh": "粪肠球菌 MIC (革兰氏阳性)",
        "bounds": (-4.0, 5.5),
        "canonical_conditions": {"cond_strain": "ATCC_ref", "cond_assay": "MHB"},
    },
    "MIC_E_faecium": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "antimicrobial",
        "display_en": "E. faecium MIC",
        "display_zh": "屎肠球菌 MIC (ESKAPE 革兰氏阳性)",
        "bounds": (-4.0, 5.5),
        "canonical_conditions": {"cond_strain": "ATCC_ref", "cond_assay": "MHB"},
    },
    "MIC_L_monocytogenes": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "antimicrobial",
        "display_en": "L. monocytogenes MIC",
        "display_zh": "单核细胞增生李斯特菌 MIC (革兰氏阳性)",
        "bounds": (-4.0, 5.5),
        "canonical_conditions": {"cond_strain": "ATCC_ref", "cond_assay": "MHB"},
    },
    "MIC_C_neoformans": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "antimicrobial",
        "display_en": "C. neoformans MIC",
        "display_zh": "新型隐球菌 MIC (真菌)",
        "bounds": (-4.0, 5.5),
        "canonical_conditions": {"cond_strain": "ATCC_ref", "cond_assay": "RPMI"},
    },
    # --- Group B: Tier 2 Species-Specific Bactericidal MBC (5 admitted; 3 rejected) ---
    "MBC_S_aureus": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "antimicrobial",
        "display_en": "S. aureus MBC (Bactericidal)",
        "display_zh": "金黄色葡萄球菌最小杀菌浓度 MBC",
        "bounds": (-4.0, 5.5),
        "canonical_conditions": {"cond_strain": "ATCC_ref", "cond_assay": "MHB"},
    },
    "MBC_E_coli": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "antimicrobial",
        "display_en": "E. coli MBC (Bactericidal)",
        "display_zh": "大肠杆菌最小杀菌浓度 MBC",
        "bounds": (-4.0, 5.5),
        "canonical_conditions": {"cond_strain": "ATCC_ref", "cond_assay": "MHB"},
    },
    "MBC_P_aeruginosa": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "antimicrobial",
        "display_en": "P. aeruginosa MBC (Bactericidal)",
        "display_zh": "铜绿假单胞菌最小杀菌浓度 MBC",
        "bounds": (-4.0, 5.5),
        "canonical_conditions": {"cond_strain": "ATCC_ref", "cond_assay": "MHB"},
    },
    "MBC_A_baumannii": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "antimicrobial",
        "display_en": "A. baumannii MBC (Bactericidal)",
        "display_zh": "鲍曼不动杆菌最小杀菌浓度 MBC",
        "bounds": (-4.0, 5.5),
        "canonical_conditions": {"cond_strain": "ATCC_ref", "cond_assay": "MHB"},
    },
    "MBC_E_faecalis": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "antimicrobial",
        "display_en": "E. faecalis MBC (Bactericidal)",
        "display_zh": "粪肠球菌最小杀菌浓度 MBC",
        "bounds": (-4.0, 5.5),
        "canonical_conditions": {"cond_strain": "ATCC_ref", "cond_assay": "MHB"},
    },
    # --- Group B: Tier 2 Antifungal Fungicidal, Anti-Biofilm & Lethal Concentration (4 admitted) ---
    "MFC_C_albicans": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "antimicrobial",
        "display_en": "C. albicans MFC (Fungicidal)",
        "display_zh": "白色念珠菌最小杀真菌浓度 MFC",
        "bounds": (-4.0, 5.5),
        "canonical_conditions": {"cond_assay": "RPMI"},
    },
    "MBIC_biofilm": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "antimicrobial",
        "display_en": "Biofilm Inhibition (MBIC)",
        "display_zh": "生物膜最小抑制浓度 MBIC",
        "bounds": (-4.0, 5.5),
        "canonical_conditions": {
            "cond_species": "pseudomonas_aeruginosa",
            "cond_estimand": "MBIC",
            "cond_assay": "TSB",
        },
    },
    "MBEC_biofilm": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "small",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "antimicrobial",
        "display_en": "Biofilm Eradication (MBEC)",
        "display_zh": "生物膜最小清除浓度 MBEC",
        "bounds": (-4.0, 5.5),
        "canonical_conditions": {
            "cond_species": "staphylococcus_aureus",
            "cond_estimand": "MBEC",
            "cond_assay": "TSB",
        },
    },
    "LC_antimicrobial": {
        "tier": "tier2",
        "task_type": "regression",
        "size_tier": "medium",
        "unit": "log10(uM)",
        "linear_unit": "uM",
        "log10_space": True,
        "domain": "antimicrobial",
        "display_en": "Antimicrobial Lethal Concentration (LC)",
        "display_zh": "抗菌致死浓度 LC (DBAASP)",
        "bounds": (-4.0, 5.5),
        "canonical_conditions": {
            "cond_species": "staphylococcus_aureus",
            "cond_assay": "LB",
        },
    },
}

V331_SEQ_ONLY_ENDPOINTS = [
    ep for ep in V331_ENDPOINT_SPECS.keys()
    if ep not in ("efflux_ratio", "Vd_absolute")
]


class V331FeatureBank:
    """Drop-in dynamic FeatureBank for N = 46,598 + K compounds (100% compatible with v33_features.FoldPreprocessor)."""

    def __init__(self, features_dir: Path = EXP_FEATURES_DIR, v33_data_dir: Path = V33_DATA):
        self.features_dir = Path(features_dir)
        self.data_dir = Path(v33_data_dir)
        manifest = json.loads((self.features_dir / "feature_manifest.json").read_text(encoding="utf-8"))
        self.manifest = manifest
        self.case17_indices: Dict[str, int] = manifest["case17_indices"]
        self.physchem_names: List[str] = manifest["physchem_core"]["feature_names"]
        self.n_total: int = int(manifest["n_compounds_total"])

        self.key_to_idx: Dict[str, int] = {}
        self.idx_to_key: List[str] = []
        with (self.features_dir / "compound_index.tsv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                idx = int(r["compound_idx"])
                ck = r["compound_key"]
                self.key_to_idx[ck] = idx
                self.idx_to_key.append(ck)

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

        self.case17_queries: List[Dict[str, Any]] = []
        with (self.data_dir / "case17_queries.tsv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                r["compound_idx"] = self.case17_indices[r["drug_name"]]
                self.case17_queries.append(r)

        assert self.physchem.shape == (self.n_total, 83)
        assert self.morgan_bits.shape == (self.n_total, 1024)
        assert self.morgan_packed.shape == (self.n_total, 128)
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


class V331FusionBank:
    """Drop-in dynamic FusionBank for N = 46,598 + K compounds (100% compatible with fusion_models.EmbeddingFoldPreprocessor)."""

    def __init__(
        self,
        features_dir: Path = EXP_FEATURES_DIR,
        embeddings_dir: Path = EXP_EMBEDDINGS_DIR,
        endpoints_dir: Path = EXP_ENDPOINTS_DIR,
        cpu_models_dir: Path = EXP_CPU_DIR,
        v33_data_dir: Path = V33_DATA,
    ):
        self.features_dir = Path(features_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.endpoints_dir = Path(endpoints_dir)
        self.cpu_models_dir = Path(cpu_models_dir)
        self.data_dir = Path(v33_data_dir)

        manifest = json.loads((self.features_dir / "feature_manifest.json").read_text(encoding="utf-8"))
        self.manifest = manifest
        self.case17_indices: Dict[str, int] = manifest["case17_indices"]
        self.physchem_names: List[str] = manifest["physchem_core"]["feature_names"]
        self.n_total: int = int(manifest["n_compounds_total"])

        self.key_to_idx: Dict[str, int] = {}
        self.idx_to_key: List[str] = []
        with (self.features_dir / "compound_index.tsv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                idx = int(r["compound_idx"])
                ck = r["compound_key"]
                self.key_to_idx[ck] = idx
                self.idx_to_key.append(ck)

        pc = np.load(self.features_dir / "physchem_core.npz")
        self.physchem: np.ndarray = np.ascontiguousarray(pc["X"], dtype=np.float32)
        self.smiles_valid: np.ndarray = np.ascontiguousarray(pc["smiles_valid_mask"], dtype=bool)
        self.seq_valid: np.ndarray = np.ascontiguousarray(pc["seq_valid_mask"], dtype=bool)
        pc.close()

        mb = np.load(self.features_dir / "morgan_bits_1024.npz")
        self.morgan_packed: np.ndarray = np.ascontiguousarray(mb["morgan_bits_packed"], dtype=np.uint8)
        mb.close()

        self.emb_blocks: Dict[str, np.ndarray] = {}
        self.emb_valid: Dict[str, np.ndarray] = {}

        d_ncaa = np.load(self.embeddings_dir / "ncaa_repdistill_v2.npz")
        v_ncaa = np.ascontiguousarray(d_ncaa["valid"], dtype=bool)
        assert np.array_equal(v_ncaa, self.smiles_valid), "ncaa_repdistill_v2 valid mask mismatch!"
        for k, bk in [("smiles_repr", "ncaa_smiles_repr"), ("fusion_hf", "ncaa_fusion_hf"), ("protein_proj", "ncaa_protein_proj")]:
            arr = np.ascontiguousarray(d_ncaa[k], dtype=np.float16)
            assert arr.shape[0] == self.n_total and np.isfinite(arr[v_ncaa]).all()
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
        for k, bk in [("mean", "esmc_mean"), ("max", "esmc_max"), ("nterm5", "esmc_nterm5"), ("cterm5", "esmc_cterm5")]:
            arr = np.ascontiguousarray(d_esmc[k], dtype=np.float16)
            assert arr.shape[0] == self.n_total and np.isfinite(arr[v_esmc]).all()
            self.emb_blocks[bk] = arr
            self.emb_valid[bk] = v_esmc
        d_esmc.close()

        self.case17_queries: List[Dict[str, Any]] = []
        with (self.data_dir / "case17_queries.tsv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                r["compound_idx"] = self.case17_indices[r["drug_name"]]
                self.case17_queries.append(r)

        assert self.physchem.shape == (self.n_total, 83)
        assert self.morgan_packed.shape == (self.n_total, 128)
        assert len(self.case17_queries) == 17

    def map_keys(self, compound_keys: List[str]) -> np.ndarray:
        return np.array([self.key_to_idx[k] for k in compound_keys], dtype=np.int32)

    def get_emb_f32(self, block_name: str, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.emb_blocks[block_name][idx].astype(np.float32),
            self.emb_valid[block_name][idx].copy(),
        )

    def load_endpoint_and_cpu(self, endpoint: str) -> Dict[str, Any]:
        path = self.endpoints_dir / f"{endpoint}.tsv"
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
        assert len(cpu_oof_rows) == len(cv_idx)
        for loc_i, orig_i in enumerate(cv_idx):
            assert cpu_oof_rows[loc_i]["record_id"] == record_ids[orig_i]
            assert cpu_oof_rows[loc_i]["compound_key"] == compound_keys[orig_i]

        with (cpu_dir / "holdout_predictions.tsv").open(encoding="utf-8") as fh:
            cpu_ho_rows = list(csv.DictReader(fh, delimiter="\t"))
        assert len(cpu_ho_rows) == len(ho_idx)
        for loc_i, orig_i in enumerate(ho_idx):
            assert cpu_ho_rows[loc_i]["record_id"] == record_ids[orig_i]
            assert cpu_ho_rows[loc_i]["compound_key"] == compound_keys[orig_i]

        with (cpu_dir / "case17_predictions.tsv").open(encoding="utf-8") as fh:
            cpu_c17_rows = list(csv.DictReader(fh, delimiter="\t"))
        assert len(cpu_c17_rows) == 17

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
