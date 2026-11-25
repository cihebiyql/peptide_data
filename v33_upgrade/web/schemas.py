"""Pydantic v2 request and response schemas for Peptide OmniPanel V33 API."""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class AssayConditions(BaseModel):
    """Optional experimental assay context parameters routed to V33 condition-aware models."""

    rbc_species: Literal["human", "horse", "mouse"] = Field(
        default="human",
        description="Red blood cell species context for hemolysis endpoints (hemolysis_cls / hemolysis_HC50).",
    )
    pk_context: Literal["human_plasma", "human_sc", "human_po", "rat_iv", "mouse_sc"] = Field(
        default="human_plasma",
        description="Pharmacokinetic species / administration route preset.",
    )
    pk_route: Optional[str] = Field(
        default=None,
        description="Optional explicit PK route override (e.g., IV, SC, PO).",
    )
    pk_species: Optional[str] = Field(
        default=None,
        description="Optional explicit PK species override (e.g., human, rat, mouse).",
    )
    pk_matrix: Optional[str] = Field(
        default=None,
        description="Optional explicit PK matrix override (e.g., in_vivo_plasma, plasma).",
    )

    def to_v33_conditions(self) -> dict[str, str]:
        """Resolve preset + explicit overrides into exact V33 inference condition keys."""
        preset_map = {
            "human_plasma": ("human", "IV", "in_vivo_plasma"),
            "human_sc": ("human", "SC", "in_vivo_plasma"),
            "human_po": ("human", "PO", "in_vivo_plasma"),
            "rat_iv": ("rat", "IV", "in_vivo_plasma"),
            "mouse_sc": ("mouse", "SC", "in_vivo_plasma"),
        }
        sp, rt, mx = preset_map.get(self.pk_context, ("human", "IV", "in_vivo_plasma"))
        return {
            "rbc_species": self.rbc_species,
            "pk_species": (self.pk_species or sp).strip().lower(),
            "pk_route": (self.pk_route or rt).strip().upper(),
            "pk_matrix": (self.pk_matrix or mx).strip().lower(),
        }


class PredictRequest(BaseModel):
    """Single-molecule ADMET & peptide property prediction request."""

    query: Optional[str] = Field(
        default=None,
        description="Unified smart input: peptide drug name (e.g. Semaglutide), 20AA sequence, or SMILES.",
        examples=["Semaglutide"],
    )
    sequence: Optional[str] = Field(
        default=None,
        description="Explicit canonical 20-AA amino acid sequence (optional; overrides query auto-detection).",
    )
    smiles: Optional[str] = Field(
        default=None,
        description="Explicit covalent SMILES string (recommended for cyclic/lipidated/modified peptides).",
    )
    include_tier2: bool = Field(
        default=True,
        description="Include Tier-2 reference-strain antimicrobial MIC endpoints (4 endpoints).",
    )
    include_svg: bool = Field(
        default=True,
        description="Include server-rendered RDKit 2D molecular structure SVG with motif highlights.",
    )
    svg_width: int = Field(default=560, ge=240, le=1600)
    svg_height: int = Field(default=340, ge=180, le=1200)
    conditions: AssayConditions = Field(default_factory=AssayConditions)


class RenderSvgRequest(BaseModel):
    """Request to render a 2D molecular structure SVG via RDKit."""

    query: Optional[str] = Field(
        default=None,
        description="Drug name, canonical 20AA sequence, or SMILES.",
    )
    sequence: Optional[str] = Field(default=None)
    smiles: Optional[str] = Field(default=None)
    width: int = Field(default=560, ge=240, le=1600)
    height: int = Field(default=340, ge=180, le=1200)


class HighlightLegendItem(BaseModel):
    """Legend entry for a highlighted substructure motif in the 2D SVG."""

    key: str
    label_en: str
    label_zh: str
    color: str
    atom_count: int


class ModificationBadge(BaseModel):
    """Detected chemical modification or structural topology badge."""

    key: str
    label_en: str
    label_zh: str
    tone: Literal["amber", "teal", "cobalt", "violet", "slate", "emerald"] = "slate"
    detail: Optional[str] = None


class RenderSvgResponse(BaseModel):
    """Server-rendered RDKit 2D SVG response with detected structural motifs."""

    svg: str
    resolved_type: str
    formula: str
    atom_count: int
    bond_count: int
    width: int
    height: int
    highlights: list[HighlightLegendItem] = Field(default_factory=list)
    modifications: list[ModificationBadge] = Field(default_factory=list)


GNN_ATTRIBUTION_ENDPOINT = Literal[
    "PAMPA",
    "Caco2",
    "hemolysis_cls",
    "hemolysis_HC50",
    "overall_peptide_toxicity",
    "cytotoxicity_cls",
    "cytotoxicity_CC50",
    "half_life_reg",
]


class AtomAttributionRequest(BaseModel):
    """Request live Pure-PyTorch E-GAT qualitative 2D-graph surrogate heatmap SVG for one endpoint."""

    query: Optional[str] = Field(
        default=None,
        description="Drug name, canonical 20-AA sequence, or SMILES.",
    )
    sequence: Optional[str] = Field(default=None)
    smiles: Optional[str] = Field(default=None)
    endpoint: GNN_ATTRIBUTION_ENDPOINT = Field(
        default="PAMPA",
        description="Target GNN attribution endpoint (PAMPA, Caco2, hemolysis_cls, hemolysis_HC50, overall_peptide_toxicity, cytotoxicity_cls, cytotoxicity_CC50, half_life_reg).",
    )
    width: int = Field(default=760, ge=320, le=1600)
    height: int = Field(default=460, ge=240, le=1200)


class TopAtomContribution(BaseModel):
    """Top contributing heavy atom entry for attribution transparency."""

    atom_index: int
    symbol: str
    weight: float


class AtomAttributionResponse(BaseModel):
    """Live per-atom GNN attribution heatmap response (Pure-PyTorch E-GAT WeightAndSum)."""

    svg: str
    endpoint: str
    endpoint_label: str
    n_atoms: int
    latency_ms: float
    cache_hit: bool = False
    gnn_prediction: Optional[dict[str, Any]] = None
    top_atoms: list[TopAtomContribution] = Field(default_factory=list)
    available_endpoints: list[str] = Field(default_factory=list)


class PhysChemSummary(BaseModel):
    """Computed physicochemical descriptors strip."""

    mw: float
    logp: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    ring_count: int
    macrocycle_rings: list[int] = Field(default_factory=list)
    heavy_atoms: int
    formal_charge: int
    net_charge_ph74: float
    formula: str
    length_aa: Optional[int] = None
    frac_csp3: float = 0.0


class AnalogEntry(BaseModel):
    """Nearest training analog for applicability domain transparency."""

    name: str
    compound_id: str
    tanimoto: float
    observed_value: Optional[float] = None
    observed_display: str
    unit: str
    censor_type: Optional[str] = None
    smiles: Optional[str] = None
    sequence: Optional[str] = None
    source: str = "V33 Training Corpus"


class EndpointPrediction(BaseModel):
    """Individual endpoint prediction with calibrated uncertainty, censored diagnostics, and AD."""

    endpoint: str
    display_en: str
    display_zh: str
    tier: Literal[1, 2]
    domain: Literal["pk", "permeability", "safety", "physicochemical", "antimicrobial"]
    kind: Literal["regression", "classification", "mechanism"]
    value: Optional[float] = None
    raw_value: Optional[float] = None
    value_display: str
    unit: str
    risk_level: Literal["favorable", "moderate", "caution", "info"]
    risk_label: str
    calibrated_prob: Optional[float] = None
    ci80_low: Optional[float] = None
    ci80_high: Optional[float] = None
    ci80_display: Optional[str] = None
    ci90_low: Optional[float] = None
    ci90_high: Optional[float] = None
    ci90_display: Optional[str] = None
    ci_scale: Literal["linear", "log10", "probability"] = "linear"
    ci_bar_min: Optional[float] = None
    ci_bar_max: Optional[float] = None
    ad_status: Literal["in_domain", "borderline", "ood", "mechanism"]
    max_tanimoto: Optional[float] = None
    top5_tanimoto: Optional[float] = None
    winning_family: Optional[str] = None
    weights: dict[str, float] = Field(default_factory=dict)
    model_family: str
    n_seeds: int = 5
    sd_seeds: Optional[float] = None
    n_train: Optional[int] = None
    n_holdout: Optional[int] = None
    n_censored: int = 0
    cv_metric_name: str = "Spearman"
    cv_score: Optional[float] = None
    holdout_score: Optional[float] = None
    v32_cv_score: Optional[float] = None
    conformal_coverage_80: Optional[float] = None
    conformal_coverage_90: Optional[float] = None
    uq_reliability: Optional[Literal["calibrated", "moderate", "low"]] = None
    uq_note: Optional[str] = None
    brier_holdout: Optional[float] = None
    ece_holdout: Optional[float] = None
    censored_note: Optional[str] = None
    censored_diagnostic_value: Optional[float] = None
    censored_diagnostic_text: Optional[str] = None
    conditions_used: dict[str, str] = Field(default_factory=dict)
    assay_context: Optional[str] = None
    note: Optional[str] = None
    prediction_source: str = "v33_remote"
    nearest_analogs: list[AnalogEntry] = Field(default_factory=list)


class MechanismAnchor(BaseModel):
    """Long-acting half-life fatty-acylation mechanism recognition result."""

    detected: bool = False
    acylation_class: Optional[str] = None
    label_en: Optional[str] = None
    label_zh: Optional[str] = None
    chain_carbons: Optional[int] = None
    value_h: Optional[float] = None
    range_h: Optional[list[float]] = None
    anchor_drugs: list[str] = Field(default_factory=list)
    note: Optional[str] = None
    ml_baseline_h: Optional[float] = None
    ml_baseline_route: Optional[str] = None
    ml_baseline_note: Optional[str] = None


class RadarAxis(BaseModel):
    """One axis of the 6-dimension ADMET overview radar chart."""

    key: str
    label_en: str
    label_zh: str
    score: float = Field(ge=0.0, le=100.0)
    raw_display: str
    status: Literal["favorable", "moderate", "caution", "info"]
    detail: str


class PredictResponse(BaseModel):
    """Complete single-molecule ADMET & physicochemical prediction response."""

    query_name: str
    resolved_type: Literal["smiles", "sequence", "dual", "drug_name"]
    modality_track: Optional[str] = None
    sequence: Optional[str] = None
    smiles: Optional[str] = None
    canonical_smiles: Optional[str] = None
    drug_reference: Optional[dict[str, Any]] = None
    physchem: PhysChemSummary
    modifications: list[ModificationBadge] = Field(default_factory=list)
    svg: Optional[str] = None
    svg_highlights: list[HighlightLegendItem] = Field(default_factory=list)
    atom_attribution_svg: Optional[str] = None
    atom_attribution_endpoint: Optional[str] = None
    atom_attribution_endpoints: list[str] = Field(default_factory=list)
    mechanism_anchor: MechanismAnchor
    radar: list[RadarAxis] = Field(default_factory=list)
    predictions: list[EndpointPrediction] = Field(default_factory=list)
    skipped: list[dict[str, Any]] = Field(default_factory=list)
    chameleonicity: Optional[dict[str, Any]] = None
    clinical_evidence: Optional[dict[str, Any]] = None
    summary: dict[str, Any] = Field(default_factory=dict)
    runtime_ms: float
    cache_hit: bool = False
    engine_timing_ms: Optional[dict[str, Any]] = None
    adapter_mode: str
    scope: str = "internal_research_only"


class BatchItemInput(BaseModel):
    """Single item in a batch prediction request."""

    name: Optional[str] = None
    query: Optional[str] = None
    sequence: Optional[str] = None
    smiles: Optional[str] = None


class BatchPredictRequest(BaseModel):
    """Batch prediction request accepting structured items or raw pasted/file text."""

    items: Optional[list[BatchItemInput]] = Field(default=None, max_length=500)
    raw_text: Optional[str] = Field(default=None, max_length=2_000_000)
    filename: Optional[str] = None
    include_tier2: bool = False
    conditions: AssayConditions = Field(default_factory=AssayConditions)


class BatchParsedRow(BaseModel):
    """One parsed molecule row for batch preview."""

    index: int
    name: str
    detected_type: str
    sequence: Optional[str] = None
    smiles: Optional[str] = None
    valid: bool
    error: Optional[str] = None


class BatchParseResponse(BaseModel):
    """Parsed batch preview response."""

    total: int
    valid_count: int
    invalid_count: int
    truncated: bool = False
    max_batch: int = 500
    rows: list[BatchParsedRow]


class BatchMoleculeResult(BaseModel):
    """Per-molecule result row inside batch prediction and task center responses."""

    index: int
    name: str
    status: Literal["ok", "error"]
    error: Optional[str] = None
    resolved_type: Optional[str] = None
    modality_track: Optional[str] = None
    sequence: Optional[str] = None
    smiles: Optional[str] = None
    canonical_smiles: Optional[str] = None
    drug_reference: Optional[dict[str, Any]] = None
    physchem: Optional[PhysChemSummary] = None
    mw: Optional[float] = None
    logp: Optional[float] = None
    net_charge_ph74: Optional[float] = None
    modifications: list[str] = Field(default_factory=list)
    modification_badges: list[ModificationBadge] = Field(default_factory=list)
    mechanism_anchor: Optional[MechanismAnchor] = None
    radar: list[RadarAxis] = Field(default_factory=list)
    key_metrics: dict[str, Any] = Field(default_factory=dict)
    predictions: list[EndpointPrediction] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    prediction_source: str = "none"


class JobCreateRequest(BaseModel):
    """Request to create an asynchronous background prediction task."""

    title: Optional[str] = Field(default=None, description="Optional user-facing task name.")
    items: Optional[list[BatchItemInput]] = Field(default=None, max_length=500)
    raw_text: Optional[str] = Field(default=None, max_length=2_000_000)
    filename: Optional[str] = None
    include_tier2: bool = True
    conditions: AssayConditions = Field(default_factory=AssayConditions)


class BatchPredictResponse(BaseModel):
    """Batch prediction matrix response."""

    total: int
    succeeded: int
    failed: int
    runtime_ms: float
    adapter_mode: str
    scope: str = "internal_research_only"
    results: list[BatchMoleculeResult]


class EndpointCatalogItem(BaseModel):
    """Audited metadata and validation benchmark entry for one V33 endpoint."""

    endpoint: str
    display_en: str
    display_zh: str
    tier: Literal[1, 2]
    domain: str
    kind: str
    unit: str
    linear_unit: str = ""
    log10_space: bool = False
    n_train: int
    n_holdout: int = 0
    n_total: int = 0
    n_censored: int = 0
    winning_family: str = ""
    weights: dict[str, float] = Field(default_factory=dict)
    model_family: str
    cv_metric_name: str
    cv_score_v32: Optional[float] = None
    holdout_score_v32: Optional[float] = None
    cv_score_v33: Optional[float] = None
    holdout_score_v33: Optional[float] = None
    conformal_coverage_80: Optional[float] = None
    conformal_coverage_90: Optional[float] = None
    uq_reliability: Optional[Literal["calibrated", "moderate", "low"]] = None
    brier_holdout: Optional[float] = None
    ece_holdout: Optional[float] = None
    has_censored_head: bool = False
    input_requirements: str
    scope_note: str
    status: Literal["active", "mechanism_layer", "conditional"] = "active"


class EndpointsResponse(BaseModel):
    """Full audited V33 endpoint catalog & validation transparency response."""

    version: str
    adapter_mode: str
    scope: str = "internal_research_only"
    manifest_generated_at: Optional[str] = None
    dataset_summary: dict[str, Any]
    overall_summary: dict[str, Any] = Field(default_factory=dict)
    representation_ablation_summary: dict[str, Any] = Field(default_factory=dict)
    case17_scorecard: dict[str, Any] = Field(default_factory=dict)
    selection_protocol: dict[str, Any] = Field(default_factory=dict)
    endpoints: list[EndpointCatalogItem]


class HealthResponse(BaseModel):
    """Service liveness and adapter metadata response."""

    ok: bool
    version: str
    adapter_mode: str
    bundle_loaded: bool
    bundle_path: Optional[str] = None
    v33_remote_url: Optional[str] = None
    v33_remote_healthy: bool = False
    manifest_loaded: bool = False
    manifest_path: Optional[str] = None
    rdkit_version: str
    endpoints_total: int
    tier1_endpoints: int
    tier2_endpoints: int
    case17_drugs_loaded: int
    dataset_summary: dict[str, Any] = Field(default_factory=dict)
    uptime_s: float
    scope: str = "internal_research_only"
