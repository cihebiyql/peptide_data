"""Peptide Intermediate Representation and minimal HELM parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .contracts import (
    CANONICAL_AMINO_ACIDS,
    DEFAULT_CHEMOTYPE,
    PeptideInput,
    ValidatedSequence,
)


_HELM_POLYMER = re.compile(r"^(?P<polymer>PEPTIDE\d+)\{(?P<body>.*)\}$")
_HELM_CONNECTION = re.compile(
    r"^(?P<left_polymer>[^,]+),(?P<right_polymer>[^,]+),"
    r"(?P<left_position>\d+):(?P<left_attachment>[^-]+)-"
    r"(?P<right_position>\d+):(?P<right_attachment>.+)$"
)


@dataclass(frozen=True)
class Monomer:
    position: int
    monomer_id: str
    symbol: str
    stereochemistry: str
    modifications: tuple[str, ...] = ()
    is_canonical: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "monomer_id": self.monomer_id,
            "symbol": self.symbol,
            "stereochemistry": self.stereochemistry,
            "modifications": list(self.modifications),
            "is_canonical": self.is_canonical,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Monomer":
        return cls(
            position=int(value["position"]),
            monomer_id=str(value["monomer_id"]),
            symbol=str(value["symbol"]),
            stereochemistry=str(value["stereochemistry"]),
            modifications=tuple(str(item) for item in value.get("modifications", [])),
            is_canonical=bool(value.get("is_canonical", False)),
        )


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    edge_type: str
    source_attachment: str | None = None
    target_attachment: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "source_attachment": self.source_attachment,
            "target_attachment": self.target_attachment,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Edge":
        return cls(
            source=int(value["source"]),
            target=int(value["target"]),
            edge_type=str(value["edge_type"]),
            source_attachment=(
                None
                if value.get("source_attachment") is None
                else str(value["source_attachment"])
            ),
            target_attachment=(
                None
                if value.get("target_attachment") is None
                else str(value["target_attachment"])
            ),
        )


@dataclass(frozen=True)
class TerminalGroups:
    n_term: str
    c_term: str

    def to_dict(self) -> dict[str, str]:
        return {"n_term": self.n_term, "c_term": self.c_term}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TerminalGroups":
        return cls(n_term=str(value["n_term"]), c_term=str(value["c_term"]))


@dataclass(frozen=True)
class PeptideIntermediateRepresentation:
    monomers: tuple[Monomer, ...]
    backbone_edges: tuple[Edge, ...]
    cyclization_edges: tuple[Edge, ...]
    linker_edges: tuple[Edge, ...]
    terminal_groups: TerminalGroups
    source_notation: str
    source_format: str
    assumed_chemotype: str | None
    ambiguity_flags: tuple[str, ...] = ()
    unknown_monomers: tuple[str, ...] = ()

    @property
    def sequence_projection(self) -> str:
        return "".join(monomer.symbol for monomer in self.monomers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "monomers": [monomer.to_dict() for monomer in self.monomers],
            "backbone_edges": [edge.to_dict() for edge in self.backbone_edges],
            "cyclization_edges": [edge.to_dict() for edge in self.cyclization_edges],
            "linker_edges": [edge.to_dict() for edge in self.linker_edges],
            "terminal_groups": self.terminal_groups.to_dict(),
            "source_notation": self.source_notation,
            "source_format": self.source_format,
            "assumed_chemotype": self.assumed_chemotype,
            "ambiguity_flags": list(self.ambiguity_flags),
            "unknown_monomers": list(self.unknown_monomers),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "PeptideIntermediateRepresentation":
        return cls(
            monomers=tuple(Monomer.from_dict(item) for item in value["monomers"]),
            backbone_edges=tuple(
                Edge.from_dict(item) for item in value.get("backbone_edges", [])
            ),
            cyclization_edges=tuple(
                Edge.from_dict(item) for item in value.get("cyclization_edges", [])
            ),
            linker_edges=tuple(
                Edge.from_dict(item) for item in value.get("linker_edges", [])
            ),
            terminal_groups=TerminalGroups.from_dict(value["terminal_groups"]),
            source_notation=str(value["source_notation"]),
            source_format=str(value["source_format"]),
            assumed_chemotype=(
                None
                if value.get("assumed_chemotype") is None
                else str(value["assumed_chemotype"])
            ),
            ambiguity_flags=tuple(
                str(item) for item in value.get("ambiguity_flags", [])
            ),
            unknown_monomers=tuple(
                str(item) for item in value.get("unknown_monomers", [])
            ),
        )

    @classmethod
    def from_json(cls, value: str) -> "PeptideIntermediateRepresentation":
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("PIR JSON must contain an object")
        return cls.from_dict(parsed)


def _backbone_edges(length: int) -> tuple[Edge, ...]:
    return tuple(
        Edge(source=position, target=position + 1, edge_type="backbone")
        for position in range(1, length)
    )


def pir_from_sequence(
    value: str | PeptideInput | ValidatedSequence, input_format: str = "auto"
) -> PeptideIntermediateRepresentation:
    """Construct the explicit default natural-L, linear PIR."""

    if isinstance(value, ValidatedSequence):
        validated = value
    elif isinstance(value, PeptideInput):
        validated = value.validate()
    else:
        validated = PeptideInput(sequence=value, format=input_format).validate()
    monomers = tuple(
        Monomer(
            position=position,
            monomer_id=symbol,
            symbol=symbol,
            stereochemistry="L",
        )
        for position, symbol in enumerate(validated.sequence, start=1)
    )
    return PeptideIntermediateRepresentation(
        monomers=monomers,
        backbone_edges=_backbone_edges(len(monomers)),
        cyclization_edges=(),
        linker_edges=(),
        terminal_groups=TerminalGroups(
            n_term="free_amine", c_term="free_carboxyl"
        ),
        source_notation=validated.source_notation,
        source_format=validated.input_format,
        assumed_chemotype=DEFAULT_CHEMOTYPE,
        ambiguity_flags=validated.ambiguity_flags,
    )


def parse_helm_tokens(body: str) -> tuple[str, ...]:
    """Split a HELM peptide body on dots while respecting bracketed monomers."""

    tokens: list[str] = []
    current: list[str] = []
    bracket_depth = 0
    for character in body.strip():
        if character == "[":
            bracket_depth += 1
        elif character == "]":
            bracket_depth -= 1
            if bracket_depth < 0:
                raise ValueError("HELM monomer body has an unmatched closing bracket")
        if character == "." and bracket_depth == 0:
            token = "".join(current).strip()
            if not token:
                raise ValueError("HELM monomer body contains an empty token")
            tokens.append(token)
            current = []
        else:
            current.append(character)
    if bracket_depth != 0:
        raise ValueError("HELM monomer body has an unmatched opening bracket")
    token = "".join(current).strip()
    if not token:
        raise ValueError("HELM monomer body is empty or ends with an empty token")
    tokens.append(token)
    return tuple(tokens)


def _classify_helm_monomer(token: str, position: int) -> tuple[Monomer, str | None]:
    if token in CANONICAL_AMINO_ACIDS:
        return Monomer(position, token, token, "L"), None

    bracketed = token.startswith("[") and token.endswith("]")
    monomer_id = token[1:-1].strip() if bracketed else token
    if not monomer_id:
        raise ValueError("HELM contains an empty bracketed monomer")
    normalized = monomer_id.replace("_", "-")
    modifications: list[str] = []
    if normalized.lower().startswith("nme-"):
        modifications.append("N_methyl")
        normalized = normalized[4:]
    elif normalized.lower().startswith("nme") and len(normalized) == 4:
        modifications.append("N_methyl")
        normalized = normalized[3:]

    stereochemistry = "L"
    if len(normalized) == 2 and normalized[0].lower() == "d":
        stereochemistry = "D"
        normalized = normalized[1:]
    elif len(normalized) == 3 and normalized[:2].lower() == "d-":
        stereochemistry = "D"
        normalized = normalized[2:]

    if normalized in CANONICAL_AMINO_ACIDS:
        return (
            Monomer(
                position=position,
                monomer_id=monomer_id,
                symbol=normalized,
                stereochemistry=stereochemistry,
                modifications=tuple(modifications),
                is_canonical=(
                    stereochemistry == "L" and not modifications and not bracketed
                ),
            ),
            None,
        )

    unknown = monomer_id
    return (
        Monomer(
            position=position,
            monomer_id=unknown,
            symbol="X",
            stereochemistry="unknown",
            modifications=tuple(modifications),
            is_canonical=False,
        ),
        unknown,
    )


def _parse_connections(
    value: str, polymer_id: str, monomer_count: int
) -> tuple[tuple[Edge, ...], tuple[str, ...]]:
    edges: list[Edge] = []
    flags: list[str] = []
    if not value:
        return (), ()
    for connection in value.split("|"):
        connection = connection.strip()
        match = _HELM_CONNECTION.fullmatch(connection)
        if not match:
            flags.append(f"unparsed_helm_connection:{connection}")
            continue
        if {
            match.group("left_polymer"),
            match.group("right_polymer"),
        } != {polymer_id}:
            flags.append(f"cross_polymer_connection_not_supported:{connection}")
            continue
        source = int(match.group("left_position"))
        target = int(match.group("right_position"))
        if not 1 <= source <= monomer_count or not 1 <= target <= monomer_count:
            raise ValueError(f"HELM connection position is out of range: {connection}")
        edges.append(
            Edge(
                source=source,
                target=target,
                edge_type="cyclization",
                source_attachment=match.group("left_attachment"),
                target_attachment=match.group("right_attachment"),
            )
        )
    return tuple(edges), tuple(flags)


def _unique_in_order(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def pir_from_helm(value: str) -> PeptideIntermediateRepresentation:
    """Parse one basic HELM PEPTIDE polymer into a loss-aware PIR.

    The parser intentionally handles only a single peptide polymer and simple
    intra-polymer connections. Unsupported monomers and connections are preserved
    through explicit flags instead of being guessed or discarded.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError("HELM input must be a non-empty string")
    helm = value.strip()
    parts = helm.split("$")
    polymer_text = parts[0].strip()
    if "|" in polymer_text:
        raise ValueError("Basic PIR parser accepts exactly one HELM peptide polymer")
    match = _HELM_POLYMER.fullmatch(polymer_text)
    if not match:
        raise ValueError("HELM input must start with one PEPTIDE<n>{...} polymer")
    polymer_id = match.group("polymer")
    tokens = parse_helm_tokens(match.group("body"))

    monomers: list[Monomer] = []
    unknown: list[str] = []
    for position, token in enumerate(tokens, start=1):
        monomer, unknown_id = _classify_helm_monomer(token, position)
        monomers.append(monomer)
        if unknown_id:
            unknown.append(unknown_id)

    connection_text = parts[1].strip() if len(parts) > 1 else ""
    cycle_edges, connection_flags = _parse_connections(
        connection_text, polymer_id, len(monomers)
    )
    flags = ["terminal_groups_not_explicit"]
    flags.extend(f"unknown_monomer:{monomer}" for monomer in unknown)
    flags.extend(connection_flags)
    return PeptideIntermediateRepresentation(
        monomers=tuple(monomers),
        backbone_edges=_backbone_edges(len(monomers)),
        cyclization_edges=cycle_edges,
        linker_edges=(),
        terminal_groups=TerminalGroups(n_term="not_explicit", c_term="not_explicit"),
        source_notation=helm,
        source_format="helm",
        assumed_chemotype=None,
        ambiguity_flags=_unique_in_order(flags),
        unknown_monomers=_unique_in_order(unknown),
    )
