"""Deterministic identity/sequence grouping and release split helpers."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

from .representations import is_sentinel


DEFAULT_SPLIT_FRACTIONS = {
    "development": 0.70,
    "calibration": 0.15,
    "sealed": 0.15,
}


def _stable_text(value: Hashable) -> str:
    return json.dumps(
        {"type": f"{type(value).__module__}.{type(value).__qualname__}", "value": repr(value)},
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_sort_key(value: Hashable) -> tuple[str, str]:
    return type(value).__qualname__, _stable_text(value)


class DeterministicDSU:
    """Disjoint-set union whose representative does not depend on union order."""

    def __init__(self, values: Iterable[Hashable] = ()) -> None:
        self._parent: dict[Hashable, Hashable] = {}
        for value in values:
            self.add(value)

    def add(self, value: Hashable) -> None:
        if value not in self._parent:
            self._parent[value] = value

    def find(self, value: Hashable) -> Hashable:
        self.add(value)
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: Hashable, right: Hashable) -> Hashable:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        root, child = sorted((left_root, right_root), key=_stable_sort_key)
        self._parent[child] = root
        return root

    def groups(self) -> dict[Hashable, tuple[Hashable, ...]]:
        grouped: dict[Hashable, list[Hashable]] = defaultdict(list)
        for value in sorted(self._parent, key=_stable_sort_key):
            grouped[self.find(value)].append(value)
        return {
            root: tuple(sorted(members, key=_stable_sort_key))
            for root, members in sorted(grouped.items(), key=lambda item: _stable_sort_key(item[0]))
        }


def normalize_sequence(value: object) -> str | None:
    """Normalize case and whitespace without erasing modification punctuation."""

    if is_sentinel(value):
        return None
    sequence = "".join(str(value).split()).upper()
    return sequence or None


def sequence_kmers(sequence: object, *, k: int = 3) -> frozenset[str]:
    if k < 1:
        raise ValueError("k must be at least 1")
    normalized = normalize_sequence(sequence)
    if normalized is None:
        return frozenset()
    if len(normalized) < k:
        return frozenset({f"__SHORT__:{normalized}"})
    return frozenset(normalized[index : index + k] for index in range(len(normalized) - k + 1))


def kmer_jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = frozenset(left)
    right_set = frozenset(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _component_id(members: Sequence[Hashable]) -> str:
    payload = "\n".join(_stable_text(member) for member in sorted(members, key=_stable_sort_key))
    return f"component_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def build_identity_sequence_components(
    rows: Iterable[Mapping[str, Any]],
    *,
    record_field: str = "record_id",
    identity_field: str = "identity_group_id",
    sequence_field: str = "sequence",
    k: int = 3,
    min_kmer_jaccard: float = 0.80,
) -> dict[Hashable, str]:
    """Group records connected by exact identity or k-mer similarity.

    Similarity edges are deliberately explicit and transitive through the DSU.
    Callers should audit component sizes because any threshold graph can form
    chained components.
    """

    if not 0.0 < min_kmer_jaccard <= 1.0:
        raise ValueError("min_kmer_jaccard must be in (0, 1]")
    ordered_rows = sorted(list(rows), key=lambda row: _stable_sort_key(row[record_field]))
    record_ids = [row[record_field] for row in ordered_rows]
    if any(is_sentinel(record_id) for record_id in record_ids):
        raise ValueError(f"{record_field} values must not be missing or sentinel values")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError(f"{record_field} values must be unique")

    dsu = DeterministicDSU(record_ids)
    by_identity: dict[str, list[Hashable]] = defaultdict(list)
    kmer_sets: dict[Hashable, frozenset[str]] = {}
    inverted_kmers: dict[str, list[Hashable]] = defaultdict(list)

    for row in ordered_rows:
        record_id = row[record_field]
        identity = row.get(identity_field)
        if not is_sentinel(identity):
            by_identity[str(identity).strip()].append(record_id)
        kmers = sequence_kmers(row.get(sequence_field), k=k)
        kmer_sets[record_id] = kmers
        for kmer in sorted(kmers):
            inverted_kmers[kmer].append(record_id)

    for members in by_identity.values():
        anchor = members[0]
        for member in members[1:]:
            dsu.union(anchor, member)

    candidate_pairs: set[tuple[Hashable, Hashable]] = set()
    for members in inverted_kmers.values():
        ordered_members = sorted(set(members), key=_stable_sort_key)
        for left_index, left in enumerate(ordered_members):
            for right in ordered_members[left_index + 1 :]:
                candidate_pairs.add((left, right))
    for left, right in sorted(
        candidate_pairs,
        key=lambda pair: (_stable_sort_key(pair[0]), _stable_sort_key(pair[1])),
    ):
        if kmer_jaccard(kmer_sets[left], kmer_sets[right]) >= min_kmer_jaccard:
            dsu.union(left, right)

    components = dsu.groups()
    component_by_record: dict[Hashable, str] = {}
    for members in components.values():
        component_id = _component_id(members)
        for record_id in members:
            component_by_record[record_id] = component_id
    return component_by_record


def _validate_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    if not fractions:
        raise ValueError("at least one split fraction is required")
    normalized = {str(name): float(value) for name, value in fractions.items()}
    if any(not math.isfinite(value) or value < 0 for value in normalized.values()):
        raise ValueError("split fractions must be finite and non-negative")
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("split fractions must have a positive sum")
    return {name: value / total for name, value in normalized.items() if value > 0}


def assign_component_splits(
    component_by_record: Mapping[Hashable, Hashable],
    *,
    fractions: Mapping[str, float] = DEFAULT_SPLIT_FRACTIONS,
    seed: int | str = 20260715,
    record_weights: Mapping[Hashable, float] | None = None,
) -> dict[Hashable, str]:
    """Assign whole components to development/calibration/sealed deterministically."""

    split_fractions = _validate_fractions(fractions)
    component_weights: dict[Hashable, float] = defaultdict(float)
    for record_id, component_id in component_by_record.items():
        weight = 1.0 if record_weights is None else float(record_weights.get(record_id, 1.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("record weights must be finite and positive")
        component_weights[component_id] += weight
    if not component_weights:
        return {}

    def tie_breaker(component_id: Hashable) -> str:
        payload = f"{seed}\0{_stable_text(component_id)}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    ordered_components = sorted(
        component_weights,
        key=lambda component_id: (-component_weights[component_id], tie_breaker(component_id)),
    )
    split_names = list(split_fractions)
    assigned_weight = {name: 0.0 for name in split_names}
    component_split: dict[Hashable, str] = {}

    # Seed each requested split when enough independent components exist.
    seed_splits = sorted(split_names, key=lambda name: (-split_fractions[name], split_names.index(name)))
    for component_id, split_name in zip(ordered_components, seed_splits):
        component_split[component_id] = split_name
        assigned_weight[split_name] += component_weights[component_id]

    for component_id in ordered_components[len(seed_splits) :]:
        split_name = min(
            split_names,
            key=lambda name: (
                assigned_weight[name] / split_fractions[name],
                split_names.index(name),
            ),
        )
        component_split[component_id] = split_name
        assigned_weight[split_name] += component_weights[component_id]

    return {
        record_id: component_split[component_id]
        for record_id, component_id in sorted(
            component_by_record.items(), key=lambda item: _stable_sort_key(item[0])
        )
    }


def crossing_audit(
    group_by_record: Mapping[Hashable, Hashable],
    split_by_record: Mapping[Hashable, str],
    *,
    group_kind: str = "group",
) -> dict[str, Any]:
    """Report whether any identity/cluster group crosses release splits."""

    grouped_records: dict[Hashable, list[Hashable]] = defaultdict(list)
    missing_records: list[Hashable] = []
    missing_group_records: list[Hashable] = []
    for record_id, group_id in group_by_record.items():
        if record_id not in split_by_record:
            missing_records.append(record_id)
            continue
        if is_sentinel(group_id):
            missing_group_records.append(record_id)
            continue
        grouped_records[group_id].append(record_id)

    rows: list[dict[str, Any]] = []
    for group_id in sorted(grouped_records, key=_stable_sort_key):
        records = sorted(grouped_records[group_id], key=_stable_sort_key)
        splits = sorted({split_by_record[record_id] for record_id in records})
        rows.append(
            {
                "group_kind": group_kind,
                "group_id": group_id,
                "record_count": len(records),
                "splits": splits,
                "crosses_splits": len(splits) > 1,
            }
        )
    crossing_rows = [row for row in rows if row["crosses_splits"]]
    return {
        "group_kind": group_kind,
        "passed": not crossing_rows and not missing_records and not missing_group_records,
        "group_count": len(rows),
        "crossing_group_count": len(crossing_rows),
        "missing_record_count": len(missing_records),
        "missing_records": sorted(missing_records, key=_stable_sort_key),
        "missing_group_record_count": len(missing_group_records),
        "missing_group_records": sorted(missing_group_records, key=_stable_sort_key),
        "groups": rows,
    }


def audit_multiple_crossings(
    split_by_record: Mapping[Hashable, str],
    groupings: Mapping[str, Mapping[Hashable, Hashable]],
) -> dict[str, Any]:
    audits = {
        name: crossing_audit(grouping, split_by_record, group_kind=name)
        for name, grouping in sorted(groupings.items())
    }
    return {"passed": all(audit["passed"] for audit in audits.values()), "audits": audits}


# Compatibility aliases with shorter names for script callers.
build_components = build_identity_sequence_components
assign_splits = assign_component_splits
audit_crossings = crossing_audit
