"""Validated sequence input contracts for Peptide-OmniPanel."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping


CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_CHEMOTYPE = "natural_L_linear_free_termini"
SUPPORTED_SEQUENCE_FORMATS = frozenset({"auto", "plain", "fasta"})


class InputValidationError(ValueError):
    """Raised when a user sequence cannot be interpreted without guessing."""


@dataclass(frozen=True)
class ValidatedSequence:
    """Normalized standard sequence plus the assumptions used to interpret it."""

    sequence: str
    input_format: str
    source_notation: str
    assumed_chemotype: str = DEFAULT_CHEMOTYPE
    record_id: str | None = None
    ambiguity_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "input_format": self.input_format,
            "source_notation": self.source_notation,
            "assumed_chemotype": self.assumed_chemotype,
            "record_id": self.record_id,
            "ambiguity_flags": list(self.ambiguity_flags),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )


@dataclass(frozen=True)
class PeptideInput:
    """Minimal public request contract for plain or FASTA sequence input."""

    sequence: str
    format: str = "auto"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PeptideInput":
        if "sequence" not in value:
            raise InputValidationError("Input mapping must contain a 'sequence' field")
        return cls(sequence=value["sequence"], format=value.get("format", "auto"))

    def validate(self) -> ValidatedSequence:
        return validate_sequence_input(self.sequence, self.format)


def _normalized_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _parse_fasta(value: str) -> tuple[str, str]:
    lines = _normalized_lines(value)
    if not lines or not lines[0].startswith(">"):
        raise InputValidationError("FASTA input must start with a '>' header")
    if len(lines[0]) == 1:
        raise InputValidationError("FASTA header must contain a record identifier")
    if any(line.startswith(">") for line in lines[1:]):
        raise InputValidationError("Exactly one FASTA record is accepted per request")
    sequence = "".join(lines[1:])
    if not sequence:
        raise InputValidationError("FASTA record contains no sequence")
    record_id = lines[0][1:].strip().split(maxsplit=1)[0]
    return sequence, record_id


def _parse_plain(value: str) -> str:
    if ">" in value:
        raise InputValidationError("Plain sequence input cannot contain a FASTA header")
    sequence = "".join(value.split())
    if not sequence:
        raise InputValidationError("Sequence input is empty")
    return sequence


def _validate_canonical_sequence(sequence: str) -> tuple[str, tuple[str, ...]]:
    flags: list[str] = []
    if any(character.islower() for character in sequence):
        flags.append("input_case_normalized_to_upper")
    normalized = sequence.upper()
    invalid = sorted(set(normalized) - CANONICAL_AMINO_ACIDS)
    if invalid:
        rendered = ",".join(invalid)
        raise InputValidationError(
            f"Plain/FASTA sequence contains unsupported monomer codes: {rendered}. "
            "Use an explicit HELM/BILN or modification-aware input for noncanonical monomers."
        )
    return normalized, tuple(flags)


def validate_sequence_input(
    value: str, input_format: str = "auto"
) -> ValidatedSequence:
    """Validate one plain or FASTA sequence and expose the default chemotype.

    Lowercase input is accepted as a convenience but is explicitly flagged before
    interpretation as the default natural-L chemotype. Noncanonical residue codes
    fail closed rather than being silently projected onto a canonical residue.
    """

    if not isinstance(value, str):
        raise InputValidationError("Sequence input must be a string")
    if not isinstance(input_format, str):
        raise InputValidationError("Input format must be a string")
    normalized_format = input_format.strip().lower()
    if normalized_format not in SUPPORTED_SEQUENCE_FORMATS:
        raise InputValidationError(
            f"Unsupported sequence format: {input_format!r}; expected auto, plain, or fasta"
        )

    lines = _normalized_lines(value)
    detected_format = "fasta" if lines and lines[0].startswith(">") else "plain"
    if normalized_format != "auto" and normalized_format != detected_format:
        raise InputValidationError(
            f"Input content looks like {detected_format}, not requested {normalized_format}"
        )

    record_id: str | None = None
    if detected_format == "fasta":
        raw_sequence, record_id = _parse_fasta(value)
    else:
        raw_sequence = _parse_plain(value)
    sequence, flags = _validate_canonical_sequence(raw_sequence)
    return ValidatedSequence(
        sequence=sequence,
        input_format=detected_format,
        source_notation=value.strip(),
        record_id=record_id,
        ambiguity_flags=flags,
    )
