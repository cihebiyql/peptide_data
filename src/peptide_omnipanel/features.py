"""Deterministic baseline features with an optional RDKit structure branch."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import CANONICAL_AMINO_ACIDS
from .representations import is_sentinel


FEATURE_SCHEMA_VERSION = "peptide_omnipanel_features_v1"
CHAR_HASH_VERSION = "blake2b_signed_l2_v1"
AMINO_ACID_ORDER = tuple("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC_RESIDUES = frozenset("AVILMFWY")
AROMATIC_RESIDUES = frozenset("FWY")
POLAR_RESIDUES = frozenset("STNQCY")
CHARGED_RESIDUES = frozenset("DEKRH")
KYTE_DOOLITTLE = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}
RDKIT_DESCRIPTOR_NAMES = (
    "mol_weight",
    "mol_logp",
    "tpsa",
    "hbond_donors",
    "hbond_acceptors",
    "rotatable_bonds",
    "ring_count",
    "heavy_atom_count",
    "fraction_csp3",
    "formal_charge",
)


@dataclass(frozen=True)
class FeatureBlock:
    """One fixed-width feature block and its availability contract."""

    name: str
    values: tuple[float, ...]
    feature_names: tuple[str, ...]
    available: bool
    fallback_reason: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if len(self.values) != len(self.feature_names):
            raise ValueError("Feature values and names must have the same length")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError(f"Feature names are not unique in block {self.name}")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError(f"Feature block {self.name} contains non-finite values")
        if self.available and self.fallback_reason:
            raise ValueError("An available feature block cannot have a fallback reason")


@dataclass(frozen=True)
class PeptideFeatureVector:
    """Concatenated blocks with explicit block-level availability features."""

    values: tuple[float, ...]
    feature_names: tuple[str, ...]
    block_names: tuple[str, ...]
    availability_mask: tuple[bool, ...]
    fallback_reasons: tuple[tuple[str, str | None], ...]
    metadata: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.feature_names):
            raise ValueError("Feature values and names must have the same length")
        if len(self.block_names) != len(self.availability_mask):
            raise ValueError("Block names and availability mask must have the same length")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Combined feature names must be unique")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("Combined features contain non-finite values")

    @property
    def availability(self) -> dict[str, bool]:
        return dict(zip(self.block_names, self.availability_mask))

    @property
    def fallbacks(self) -> dict[str, str | None]:
        return dict(self.fallback_reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": list(self.feature_names),
            "values": list(self.values),
            "block_names": list(self.block_names),
            "availability_mask": [bool(value) for value in self.availability_mask],
            "fallback_reasons": dict(self.fallback_reasons),
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )


def _normalize_character_input(value: object, modality: str) -> str | None:
    if is_sentinel(value):
        return None
    text = "".join(str(value).split())
    if not text:
        return None
    return text.upper() if modality == "sequence" else text


def _ngram_bounds(ngram_range: Sequence[int]) -> tuple[int, int]:
    if len(ngram_range) != 2:
        raise ValueError("ngram_range must contain exactly two integers")
    lower, upper = int(ngram_range[0]), int(ngram_range[1])
    if lower < 1 or upper < lower:
        raise ValueError("ngram_range must satisfy 1 <= lower <= upper")
    return lower, upper


def hashed_character_ngrams(
    value: object,
    *,
    modality: str,
    n_features: int = 256,
    ngram_range: Sequence[int] = (1, 3),
) -> FeatureBlock:
    """Return signed, L2-normalized character hashing features.

    Python's randomized ``hash`` is deliberately avoided. The modality, n-gram
    size and token are salted into BLAKE2b, so sequence, HELM and SMILES blocks
    cannot collide merely because their source text is identical.
    """

    if n_features < 1:
        raise ValueError("n_features must be at least 1")
    if not modality or not modality.replace("_", "").isalnum():
        raise ValueError("modality must be a non-empty alphanumeric identifier")
    lower, upper = _ngram_bounds(ngram_range)
    feature_names = tuple(
        f"{modality}_char_hash_{index:04d}" for index in range(n_features)
    )
    text = _normalize_character_input(value, modality)
    if text is None:
        return FeatureBlock(
            name=f"{modality}_char",
            values=(0.0,) * n_features,
            feature_names=feature_names,
            available=False,
            fallback_reason="input_missing_or_sentinel",
            metadata=(("hash_version", CHAR_HASH_VERSION),),
        )

    bounded = f"^{text}$"
    values = [0.0] * n_features
    for size in range(lower, upper + 1):
        if len(bounded) < size:
            continue
        for start in range(len(bounded) - size + 1):
            ngram = bounded[start : start + size]
            payload = f"{CHAR_HASH_VERSION}\0{modality}\0{size}\0{ngram}".encode(
                "utf-8"
            )
            digest = hashlib.blake2b(
                payload, digest_size=16, person=b"POMNI-ngram-v1"
            ).digest()
            bucket = int.from_bytes(digest[:8], "big") % n_features
            sign = 1.0 if digest[8] & 1 else -1.0
            values[bucket] += sign
    norm = math.sqrt(sum(value * value for value in values))
    if norm:
        values = [value / norm for value in values]
    return FeatureBlock(
        name=f"{modality}_char",
        values=tuple(values),
        feature_names=feature_names,
        available=True,
        metadata=(
            ("hash_version", CHAR_HASH_VERSION),
            ("ngram_range", f"{lower}-{upper}"),
        ),
    )


def _basic_side_chain_charge(sequence: str, ph: float) -> float:
    base_pka = {"H": 6.0, "K": 10.5, "R": 12.5}
    acid_pka = {"D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1}
    positive = 1.0 / (1.0 + 10.0 ** (ph - 8.0))
    negative = 1.0 / (1.0 + 10.0 ** (3.1 - ph))
    for residue, pka in base_pka.items():
        positive += sequence.count(residue) / (1.0 + 10.0 ** (ph - pka))
    for residue, pka in acid_pka.items():
        negative += sequence.count(residue) / (1.0 + 10.0 ** (pka - ph))
    return positive - negative


def sequence_physicochemical_features(
    value: object, *, ph: float = 7.0
) -> FeatureBlock:
    """Compute composition and simple free-termini sequence descriptors."""

    if not math.isfinite(ph):
        raise ValueError("ph must be finite")
    feature_names = tuple(
        [f"sequence_aa_fraction_{residue}" for residue in AMINO_ACID_ORDER]
        + [
            "sequence_length",
            "sequence_log1p_length",
            f"sequence_estimated_net_charge_pH{ph:g}_free_termini",
            "sequence_hydrophobic_fraction",
            "sequence_aromatic_fraction",
            "sequence_polar_fraction",
            "sequence_charged_fraction",
            "sequence_mean_kyte_doolittle",
            "sequence_unknown_fraction",
        ]
    )
    sequence = _normalize_character_input(value, "sequence")
    if sequence is None:
        return FeatureBlock(
            name="sequence_physchem",
            values=(0.0,) * len(feature_names),
            feature_names=feature_names,
            available=False,
            fallback_reason="input_missing_or_sentinel",
            metadata=(("net_charge_assumption", "free_termini"),),
        )

    length = len(sequence)
    canonical_count = sum(residue in CANONICAL_AMINO_ACIDS for residue in sequence)
    fractions = [sequence.count(residue) / length for residue in AMINO_ACID_ORDER]
    hydrophobic = sum(residue in HYDROPHOBIC_RESIDUES for residue in sequence) / length
    aromatic = sum(residue in AROMATIC_RESIDUES for residue in sequence) / length
    polar = sum(residue in POLAR_RESIDUES for residue in sequence) / length
    charged = sum(residue in CHARGED_RESIDUES for residue in sequence) / length
    hydropathy = sum(KYTE_DOOLITTLE.get(residue, 0.0) for residue in sequence) / length
    values = fractions + [
        float(length),
        math.log1p(length),
        _basic_side_chain_charge(sequence, ph),
        hydrophobic,
        aromatic,
        polar,
        charged,
        hydropathy,
        1.0 - canonical_count / length,
    ]
    return FeatureBlock(
        name="sequence_physchem",
        values=tuple(values),
        feature_names=feature_names,
        available=True,
        metadata=(
            ("net_charge_assumption", "free_termini"),
            ("hydropathy_scale", "Kyte-Doolittle"),
        ),
    )


def _morgan_feature_names(radius: int, n_bits: int) -> tuple[str, ...]:
    return tuple(f"rdkit_morgan_r{radius}_{index:04d}" for index in range(n_bits))


def _descriptor_feature_names() -> tuple[str, ...]:
    return tuple(f"rdkit_descriptor_{name}" for name in RDKIT_DESCRIPTOR_NAMES)


def _unavailable_rdkit_blocks(
    radius: int, n_bits: int, reason: str, version: str | None = None
) -> tuple[FeatureBlock, FeatureBlock]:
    metadata = (("rdkit_version", version or "unavailable"),)
    return (
        FeatureBlock(
            name="rdkit_morgan",
            values=(0.0,) * n_bits,
            feature_names=_morgan_feature_names(radius, n_bits),
            available=False,
            fallback_reason=reason,
            metadata=metadata,
        ),
        FeatureBlock(
            name="rdkit_descriptors",
            values=(0.0,) * len(RDKIT_DESCRIPTOR_NAMES),
            feature_names=_descriptor_feature_names(),
            available=False,
            fallback_reason=reason,
            metadata=metadata,
        ),
    )


def _load_rdkit() -> tuple[Mapping[str, Any] | None, str | None, str | None]:
    try:
        from rdkit import Chem, rdBase  # type: ignore[import-not-found]
        from rdkit.Chem import (  # type: ignore[import-not-found]
            Crippen,
            Descriptors,
            rdFingerprintGenerator,
            rdMolDescriptors,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        return None, None, f"rdkit_unavailable:{type(exc).__name__}"
    backend = {
        "Chem": Chem,
        "Crippen": Crippen,
        "Descriptors": Descriptors,
        "rdFingerprintGenerator": rdFingerprintGenerator,
        "rdMolDescriptors": rdMolDescriptors,
    }
    return backend, str(getattr(rdBase, "rdkitVersion", "unknown")), None


def rdkit_morgan_descriptor_features(
    smiles: object,
    *,
    radius: int = 2,
    n_bits: int = 2048,
    use_rdkit: bool = True,
    rdkit_backend: Mapping[str, Any] | None = None,
    rdkit_version: str | None = None,
) -> tuple[FeatureBlock, FeatureBlock]:
    """Return Morgan bits and basic descriptors, or explicit zero fallbacks."""

    if radius < 0:
        raise ValueError("radius must be non-negative")
    if n_bits < 1:
        raise ValueError("n_bits must be at least 1")
    if is_sentinel(smiles):
        return _unavailable_rdkit_blocks(radius, n_bits, "smiles_missing_or_sentinel")
    if not use_rdkit:
        return _unavailable_rdkit_blocks(radius, n_bits, "rdkit_not_requested")

    import_error = None
    if rdkit_backend is None:
        rdkit_backend, detected_version, import_error = _load_rdkit()
        rdkit_version = rdkit_version or detected_version
    if rdkit_backend is None:
        return _unavailable_rdkit_blocks(
            radius, n_bits, import_error or "rdkit_unavailable", rdkit_version
        )

    version = rdkit_version or "injected_backend"
    metadata = (("rdkit_version", version),)
    chem = rdkit_backend["Chem"]
    try:
        molecule = chem.MolFromSmiles(str(smiles).strip())
    except Exception as exc:
        return _unavailable_rdkit_blocks(
            radius, n_bits, f"smiles_parse_failed:{type(exc).__name__}", version
        )
    if molecule is None:
        return _unavailable_rdkit_blocks(
            radius, n_bits, "smiles_parse_failed:MolFromSmiles_returned_None", version
        )

    try:
        generator = rdkit_backend["rdFingerprintGenerator"].GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
        fingerprint = generator.GetFingerprint(molecule)
        morgan_values = [0.0] * n_bits
        for bit in fingerprint.GetOnBits():
            morgan_values[int(bit)] = 1.0
        morgan = FeatureBlock(
            name="rdkit_morgan",
            values=tuple(morgan_values),
            feature_names=_morgan_feature_names(radius, n_bits),
            available=True,
            metadata=metadata,
        )
    except Exception as exc:
        morgan, _ = _unavailable_rdkit_blocks(
            radius,
            n_bits,
            f"morgan_failed:{type(exc).__name__}",
            version,
        )

    try:
        descriptors = rdkit_backend["Descriptors"]
        crippen = rdkit_backend["Crippen"]
        mol_descriptors = rdkit_backend["rdMolDescriptors"]
        descriptor_values = (
            float(descriptors.MolWt(molecule)),
            float(crippen.MolLogP(molecule)),
            float(mol_descriptors.CalcTPSA(molecule)),
            float(mol_descriptors.CalcNumHBD(molecule)),
            float(mol_descriptors.CalcNumHBA(molecule)),
            float(mol_descriptors.CalcNumRotatableBonds(molecule)),
            float(mol_descriptors.CalcNumRings(molecule)),
            float(molecule.GetNumHeavyAtoms()),
            float(mol_descriptors.CalcFractionCSP3(molecule)),
            float(chem.GetFormalCharge(molecule)),
        )
        descriptor_block = FeatureBlock(
            name="rdkit_descriptors",
            values=descriptor_values,
            feature_names=_descriptor_feature_names(),
            available=True,
            metadata=metadata,
        )
    except Exception as exc:
        _, descriptor_block = _unavailable_rdkit_blocks(
            radius,
            n_bits,
            f"descriptor_failed:{type(exc).__name__}",
            version,
        )
    return morgan, descriptor_block


def combine_feature_blocks(blocks: Sequence[FeatureBlock]) -> PeptideFeatureVector:
    """Concatenate fixed-width blocks and append one availability feature each."""

    if not blocks:
        raise ValueError("At least one feature block is required")
    block_names = tuple(block.name for block in blocks)
    if len(set(block_names)) != len(block_names):
        raise ValueError("Feature block names must be unique")
    base_values = tuple(value for block in blocks for value in block.values)
    base_names = tuple(name for block in blocks for name in block.feature_names)
    mask_names = tuple(f"availability_{name}" for name in block_names)
    mask_values = tuple(1.0 if block.available else 0.0 for block in blocks)
    metadata: dict[str, str] = {"feature_schema_version": FEATURE_SCHEMA_VERSION}
    for block in blocks:
        for key, value in block.metadata:
            metadata[f"{block.name}.{key}"] = value
    return PeptideFeatureVector(
        values=base_values + mask_values,
        feature_names=base_names + mask_names,
        block_names=block_names,
        availability_mask=tuple(block.available for block in blocks),
        fallback_reasons=tuple(
            (block.name, block.fallback_reason) for block in blocks
        ),
        metadata=tuple(sorted(metadata.items())),
    )


def build_peptide_features(
    *,
    sequence: object = None,
    helm: object = None,
    smiles: object = None,
    char_hash_features: int = 256,
    char_ngram_range: Sequence[int] = (1, 3),
    ph: float = 7.0,
    morgan_radius: int = 2,
    morgan_bits: int = 2048,
    use_rdkit: bool = True,
    rdkit_backend: Mapping[str, Any] | None = None,
    rdkit_version: str | None = None,
) -> PeptideFeatureVector:
    """Build all baseline blocks without requiring any optional dependency."""

    blocks = [
        hashed_character_ngrams(
            sequence,
            modality="sequence",
            n_features=char_hash_features,
            ngram_range=char_ngram_range,
        ),
        hashed_character_ngrams(
            helm,
            modality="helm",
            n_features=char_hash_features,
            ngram_range=char_ngram_range,
        ),
        hashed_character_ngrams(
            smiles,
            modality="smiles",
            n_features=char_hash_features,
            ngram_range=char_ngram_range,
        ),
        sequence_physicochemical_features(sequence, ph=ph),
    ]
    blocks.extend(
        rdkit_morgan_descriptor_features(
            smiles,
            radius=morgan_radius,
            n_bits=morgan_bits,
            use_rdkit=use_rdkit,
            rdkit_backend=rdkit_backend,
            rdkit_version=rdkit_version,
        )
    )
    return combine_feature_blocks(blocks)


# Concise aliases for training scripts.
char_ngram_features = hashed_character_ngrams
physicochemical_features = sequence_physicochemical_features
