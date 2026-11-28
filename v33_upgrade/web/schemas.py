"""Pydantic v2 request and response schemas for Peptide OmniPanel V33 API."""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    interpretability_tier: Optional[Literal["faithful_surrogate", "qualitative_saliency_heuristic", "below_random_unreliable"]] = None
    fidelity_vs_4fm_spearman: Optional[float] = None
    deletion_top_vs_rand_ratio: Optional[float] = None
    deletion_top_vs_rand_ci95: Optional[list[float]] = None


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
    holdout_ci95: Optional[list[float]] = None
    evidence_grade: Optional[Literal["quantitative", "rank_only", "exploratory_not_recommended"]] = None
    not_recommended_warning: Optional[str] = None
    v32_cv_score: Optional[float] = None
    v32_cv_note: str = Field(
        default="deprecated_legacy_unclustered_cv (historical reference only; do not use for model selection)",
        description="Explicit methodological provenance tag for legacy v3.2 CV values.",
    )
    clinical_percentile: Optional[dict[str, Any]] = Field(
        default=None,
        description="Approved-peptide / curated-panel clinical percentile context for this predicted value.",
    )
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
    v32_cv_note: str = Field(
        default="deprecated_legacy_unclustered_cv (historical reference only; do not use for model selection)",
        description="Explicit methodological provenance tag for legacy v3.2 CV values.",
    )
    holdout_score_v32: Optional[float] = None
    cv_score_v33: Optional[float] = None
    holdout_score_v33: Optional[float] = None
    holdout_ci95: Optional[list[float]] = None
    evidence_grade: Optional[Literal["quantitative", "rank_only", "exploratory_not_recommended"]] = None
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
    v335_protocol_audit: dict[str, Any] = Field(default_factory=dict)
    egat_fidelity_audit: dict[str, Any] = Field(default_factory=dict)
    external_baselines_same_split: dict[str, Any] = Field(default_factory=dict)
    temporal_split_benchmarks: dict[str, Any] = Field(default_factory=dict)
    recent_fda_2019_2023_blind_set: dict[str, Any] = Field(default_factory=dict)
    v34_upgrades: dict[str, Any] = Field(default_factory=dict)
    endpoints: list[EndpointCatalogItem]


# =============================================================================
# V3.4 Interactive Design, Batch Screening + Peptide-MPO & Direct Attribution
# =============================================================================


PEPTIDE_OPT_STRATEGY = Literal[
    "ala_scan",
    "point_mutation",
    "n_methylation",
    "cyclization",
    "lipidation_c16_c18",
]


class MpoWeights(BaseModel):
    """Customizable Peptide Multi-Parameter Optimization (Peptide-MPO) weights and flags."""

    permeability: float = Field(default=0.25, ge=0.0, le=10.0)
    half_life: float = Field(default=0.25, ge=0.0, le=10.0)
    low_hemolysis: float = Field(default=0.25, ge=0.0, le=10.0)
    low_toxicity: float = Field(default=0.25, ge=0.0, le=10.0)
    penalize_ood: bool = Field(default=True, description="Apply applicability-domain OOD penalty multiplier.")
    weight_by_evidence_grade: bool = Field(
        default=True,
        description="Weight pillars by endpoint evidence grade (quantitative=1.0, rank_only=0.75, exploratory=0.25).",
    )


class PeptideOptObjective(BaseModel):
    """Single optimization axis specification (endpoint + maximization/minimization direction)."""

    endpoint: str = Field(default="PAMPA", description="Target V33 endpoint identifier.")
    direction: Literal["max", "min"] = Field(default="max", description="Optimization direction: 'max' or 'min'.")


class PeptideOptRequest(BaseModel):
    """PeptideOpt structure-activity modification & Pareto optimization request."""

    query: Optional[str] = Field(
        default=None,
        description="Optional unified peptide drug name, canonical 20-AA sequence, or SMILES.",
    )
    sequence: Optional[str] = Field(default=None, description="Explicit canonical 20-AA sequence.")
    smiles: Optional[str] = Field(default=None, description="Explicit covalent SMILES string.")
    strategies: list[PEPTIDE_OPT_STRATEGY] = Field(
        default_factory=lambda: [
            "ala_scan",
            "point_mutation",
            "n_methylation",
            "cyclization",
            "lipidation_c16_c18",
        ],
        min_length=1,
        description="Modification strategies to enumerate chemically valid RDKit-sanitized variants.",
    )
    max_variants: int = Field(default=20, ge=1, le=32, description="Maximum variants to evaluate (capped at 32).")
    objective_x: PeptideOptObjective = Field(
        default_factory=lambda: PeptideOptObjective(endpoint="PAMPA", direction="max")
    )
    objective_y: PeptideOptObjective = Field(
        default_factory=lambda: PeptideOptObjective(endpoint="half_life_reg", direction="max")
    )
    conditions: AssayConditions = Field(default_factory=AssayConditions)
    mpo_weights: MpoWeights = Field(default_factory=MpoWeights)


class PeptideOptObjectiveMeta(BaseModel):
    """Resolved axis metadata for PeptideOpt scatter & Pareto visualization."""

    endpoint: str
    direction: Literal["max", "min"]
    display_en: str
    display_zh: str
    unit: str
    log10_space: bool = False


class PeptideOptVariant(BaseModel):
    """Evaluated parent or modified peptide variant with Pareto & safety annotations."""

    id: str
    name: str
    strategy: str
    mutation_label: str
    description: str
    sequence: Optional[str] = None
    smiles: str
    is_parent: bool = False
    is_pareto_optimal: bool = False
    x_value: Optional[float] = None
    x_raw: Optional[float] = None
    x_display: str = "N/A"
    y_value: Optional[float] = None
    y_raw: Optional[float] = None
    y_display: str = "N/A"
    delta_x: Optional[float] = None
    delta_y: Optional[float] = None
    delta_x_raw: Optional[float] = None
    delta_y_raw: Optional[float] = None
    delta_x_pct: Optional[float] = None
    delta_y_pct: Optional[float] = None
    ad_status: Literal["in_domain", "borderline", "ood"] = "borderline"
    ad_status_x: Literal["in_domain", "borderline", "ood"] = "borderline"
    ad_status_y: Literal["in_domain", "borderline", "ood"] = "borderline"
    mpo_score: float = Field(default=0.0, ge=0.0, le=100.0)
    safety_guardrail_pass: bool = True
    safety_flags: list[str] = Field(default_factory=list)
    mw: Optional[float] = None
    logp: Optional[float] = None
    tpsa: Optional[float] = None
    hbd: Optional[int] = None
    net_charge_ph74: Optional[float] = None
    key_metrics: dict[str, Any] = Field(default_factory=dict)


class PeptideOptResponse(BaseModel):
    """Complete PeptideOpt multi-objective optimization & Pareto frontier response."""

    parent: PeptideOptVariant
    variants: list[PeptideOptVariant] = Field(default_factory=list)
    pareto_frontier: list[PeptideOptVariant] = Field(default_factory=list)
    objective_x: PeptideOptObjectiveMeta
    objective_y: PeptideOptObjectiveMeta
    strategies_used: list[str] = Field(default_factory=list)
    strategies_skipped: list[dict[str, str]] = Field(default_factory=list)
    total_generated: int = 0
    total_evaluated: int = 0
    pareto_count: int = 0
    runtime_ms: float
    adapter_mode: str
    scope: str = "internal_research_only"


class BatchScreenItemInput(BaseModel):
    """Single item for Peptide-MPO batch screening (up to 100 compounds)."""

    id: Optional[str] = None
    name: Optional[str] = None
    query: Optional[str] = None
    sequence: Optional[str] = None
    smiles: Optional[str] = None


class BatchScreenRequest(BaseModel):
    """Batch screening & Peptide-MPO ranking request (max 100 compounds)."""

    items: Optional[list[BatchScreenItemInput]] = Field(default=None, max_length=100)
    raw_text: Optional[str] = Field(default=None, max_length=500_000)
    filename: Optional[str] = None
    weights: MpoWeights = Field(default_factory=MpoWeights)
    include_tier2: bool = False
    conditions: AssayConditions = Field(default_factory=AssayConditions)


class BatchScreenResult(BaseModel):
    """One ranked compound row in Peptide-MPO batch screening."""

    rank: Optional[int] = None
    id: str
    name: str
    status: Literal["ok", "error"]
    error: Optional[str] = None
    resolved_type: Optional[str] = None
    sequence: Optional[str] = None
    smiles: Optional[str] = None
    canonical_smiles: Optional[str] = None
    peptide_mpo_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    mpo_tier: Literal["top", "moderate", "caution", "error"] = "moderate"
    pillar_scores: dict[str, float] = Field(default_factory=dict)
    pillar_details: dict[str, dict[str, Any]] = Field(default_factory=dict)
    ood_penalty: float = 1.0
    ad_overall: Literal["in_domain", "borderline", "ood"] = "borderline"
    ad_counts: dict[str, int] = Field(default_factory=dict)
    mw: Optional[float] = None
    logp: Optional[float] = None
    net_charge_ph74: Optional[float] = None
    key_metrics: dict[str, Any] = Field(default_factory=dict)
    prediction_source: str = "none"


class BatchScreenResponse(BaseModel):
    """Sorted Peptide-MPO screening leaderboard + ready-to-download CSV string."""

    total: int
    succeeded: int
    failed: int
    truncated: bool = False
    max_items: int = 100
    weights_used: MpoWeights
    results: list[BatchScreenResult] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    csv_export: str = ""
    runtime_ms: float
    adapter_mode: str
    scope: str = "internal_research_only"


class DirectAttributionRequest(BaseModel):
    """Model-faithful direct ensemble perturbation attribution request (all 47 endpoints)."""

    query: Optional[str] = Field(default=None, description="Drug name, canonical 20-AA sequence, or SMILES.")
    sequence: Optional[str] = Field(default=None)
    smiles: Optional[str] = Field(default=None)
    endpoint: str = Field(
        default="PAMPA",
        description="Any supported V33 endpoint identifier (all 47 Tier-1 and Tier-2 endpoints).",
    )
    width: int = Field(default=760, ge=320, le=1600)
    height: int = Field(default=460, ge=240, le=1200)
    max_perturbations: int = Field(default=24, ge=4, le=40)
    conditions: AssayConditions = Field(default_factory=AssayConditions)


class DirectAtomWeight(BaseModel):
    """Per-atom direct ensemble perturbation contribution."""

    atom_index: int
    symbol: str
    weight: float = Field(ge=-1.0, le=1.0, description="Signed normalized contribution in [-1, 1].")
    weight_01: float = Field(ge=0.0, le=1.0, description="Neutral-centered [0, 1] weight for diverging heatmap.")
    raw_delta: float = Field(description="Raw model-space delta (pred_full - pred_masked).")
    group_id: Optional[str] = None


class ResidueOrGroupImpact(BaseModel):
    """One residue masking or chemical substituent pruning impact entry."""

    id: str
    label: str
    kind: Literal["residue", "sidechain", "n_methyl", "lipid_tail", "functional_group"]
    position: Optional[int] = None
    original_residue: Optional[str] = None
    masked_to: Optional[str] = None
    perturbed_smiles: str
    perturbed_value: Optional[float] = None
    perturbed_raw: Optional[float] = None
    delta_raw: float
    delta_display: Optional[float] = None
    normalized_weight: float = Field(ge=-1.0, le=1.0)
    atom_indices: list[int] = Field(default_factory=list)
    direction: Literal["increases", "decreases", "neutral"]


class DirectAttributionResponse(BaseModel):
    """100% production-ensemble-faithful direct perturbation attribution response."""

    endpoint: str
    endpoint_label: str
    endpoint_label_zh: str
    interpretability_tier: Literal["direct_ensemble_faithful"] = "direct_ensemble_faithful"
    method: Literal["residue_alanine_glycine_masking", "chemical_substituent_pruning"]
    method_description: str
    base_prediction: dict[str, Any]
    n_atoms: int
    n_perturbations: int
    atom_weights: list[DirectAtomWeight] = Field(default_factory=list)
    top_atoms: list[TopAtomContribution] = Field(default_factory=list)
    residue_or_group_impacts: list[ResidueOrGroupImpact] = Field(default_factory=list)
    svg_2d: str
    svg: str
    available_endpoints: list[str] = Field(default_factory=list)
    latency_ms: float
    cache_hit: bool = False
    adapter_mode: str


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
    harness_llm_enabled: bool = False
    scope: str = "internal_research_only"


# =============================================================================
# V3.5 Peptide-Harness Controlled Multi-Round Optimization Schemas
# =============================================================================


class PeptideHarnessRequest(BaseModel):
    """Multi-round controlled evolutionary optimization request (strictly forbids unknown fields / SSRF params)."""

    model_config = ConfigDict(extra="forbid")

    query: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Unified peptide drug name, canonical 20-AA sequence, or SMILES.",
    )
    sequence: Optional[str] = Field(default=None, max_length=120, description="Explicit canonical 20-AA sequence.")
    smiles: Optional[str] = Field(default=None, max_length=4000, description="Explicit covalent SMILES string.")
    objective_x: PeptideOptObjective = Field(
        default_factory=lambda: PeptideOptObjective(endpoint="PAMPA", direction="max")
    )
    objective_y: PeptideOptObjective = Field(
        default_factory=lambda: PeptideOptObjective(endpoint="half_life_reg", direction="max")
    )
    functional_endpoint: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional Tier-A functional endpoint constrained to non-worsening relative to initial seed.",
    )
    functional_direction: Optional[Literal["max", "min"]] = Field(
        default=None,
        description="Optional override for functional endpoint direction ('max' or 'min').",
    )
    strategies: list[PEPTIDE_OPT_STRATEGY] = Field(
        default_factory=lambda: [
            "ala_scan",
            "point_mutation",
            "n_methylation",
            "cyclization",
            "lipidation_c16_c18",
        ],
        min_length=1,
        max_length=5,
    )
    locked_positions: list[int] = Field(
        default_factory=list,
        max_length=65,
        description="1-based canonical residue indices locked against mutation or chemical modification.",
    )
    lock_disulfides: bool = Field(default=True, description="Lock native disulfide bridges (S-S) intact.")
    locked_smarts: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Optional SMARTS core motifs preserved across all proposals (<=8 patterns, <=256 chars each).",
    )
    rounds: int = Field(default=3, ge=1, le=4)
    population_size: int = Field(default=6, ge=2, le=8)
    top_k: int = Field(default=5, ge=1, le=5)
    critic_max_perturbations: int = Field(default=8, ge=2, le=10)
    max_total_evals: int = Field(default=36, ge=4, le=60)
    wall_clock_budget_s: float = Field(default=120.0, ge=10.0, le=180.0)
    requested_max_aa_edits: Optional[int] = Field(default=None, ge=0, le=3)
    requested_max_chem_edits: Optional[int] = Field(default=None, ge=0, le=2)
    compute_elite_cross_track_audit: bool = Field(default=True)
    mpo_weights: MpoWeights = Field(default_factory=MpoWeights)
    conditions: AssayConditions = Field(default_factory=AssayConditions)

    @field_validator("locked_smarts")
    @classmethod
    def _validate_locked_smarts(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for s in v:
            sc = (s or "").strip()
            if not sc:
                continue
            if len(sc) > 256:
                raise ValueError("Each locked_smarts pattern must be <= 256 characters.")
            out.append(sc)
        return out


class PeptideHarnessSyncRequest(PeptideHarnessRequest):
    """Strictly capped single-round fast synchronous Peptide-Harness request (never exceeds gateway timeout)."""

    rounds: int = Field(default=1, ge=1, le=1)
    population_size: int = Field(default=4, ge=2, le=4)
    critic_max_perturbations: int = Field(default=4, ge=2, le=4)
    max_total_evals: int = Field(default=6, ge=4, le=8)
    wall_clock_budget_s: float = Field(default=15.0, ge=5.0, le=20.0)


class PeptideHarnessJobMeta(BaseModel):
    """Metadata and live round progress snapshot for an async Peptide-Harness job."""

    job_id: str
    status: Literal["queued", "running", "completed", "partial", "cancelled", "failed"]
    phase: str
    current_round: int = 0
    total_rounds: int = 1
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    runtime_ms: float = 0.0
    cancel_requested: bool = False
    error: Optional[str] = None
    preflight: dict[str, Any] = Field(default_factory=dict)
    request_summary: dict[str, Any] = Field(default_factory=dict)


class PeptideHarnessJobStatusResponse(PeptideHarnessJobMeta):
    """Full status & result response for `GET /api/v1/peptide-harness/jobs/{job_id}`."""

    result: Optional[dict[str, Any]] = None


class PeptideHarnessCancelResponse(PeptideHarnessJobMeta):
    """Cooperative cancellation response for `DELETE /api/v1/peptide-harness/jobs/{job_id}`."""

    already_finished: bool = False


class PeptideHarnessJobListResponse(BaseModel):
    """List of Peptide-Harness jobs scoped to the calling X-Client-Id."""

    jobs: list[PeptideHarnessJobMeta] = Field(default_factory=list)
    active_count: int = 0
    total: int = 0
