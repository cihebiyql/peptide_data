"""Representation validation helpers with an optional RDKit dependency."""

from __future__ import annotations

import math
from typing import Any, Iterable


DEFAULT_SENTINELS = frozenset(
    {
        "",
        "-",
        ".",
        "<na>",
        "na",
        "n/a",
        "n.a",
        "n.a.",
        "nan",
        "none",
        "null",
        "not available",
        "not applicable",
        "not reported",
        "unknown",
    }
)


def is_sentinel(value: object, *, sentinels: Iterable[str] = DEFAULT_SENTINELS) -> bool:
    """Return whether a representation value is missing or a known sentinel.

    Numeric zero and ``False`` are deliberately not treated as missing. String
    matching is case-insensitive and ignores surrounding whitespace.
    """

    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return False
    if not isinstance(value, str):
        return False
    normalized_sentinels = {str(item).strip().casefold() for item in sentinels}
    return value.strip().casefold() in normalized_sentinels


def _load_rdkit_chem() -> tuple[Any | None, str | None, str | None]:
    try:
        from rdkit import Chem, rdBase  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    version = getattr(rdBase, "rdkitVersion", None)
    return Chem, str(version) if version is not None else "unknown", None


def _empty_audit(value: object) -> dict[str, Any]:
    return {
        "input": value,
        "normalized_input": None,
        "is_sentinel": False,
        "status": "not_checked",
        "rdkit_status": "not_checked",
        "rdkit_version": None,
        "parse_ok": None,
        "sanitize_ok": None,
        "roundtrip_ok": None,
        "canonical_smiles": None,
        "roundtrip_smiles": None,
        "error": None,
    }


def audit_smiles_representation(
    value: object,
    *,
    check_roundtrip: bool = True,
    use_rdkit: bool = True,
    chem_module: Any | None = None,
    rdkit_version: str | None = None,
) -> dict[str, Any]:
    """Audit a SMILES value without making RDKit a mandatory dependency.

    ``chem_module`` is an explicit dependency-injection hook for callers and
    tests. If it is omitted, RDKit is imported lazily. A missing RDKit install
    returns ``status='rdkit_unavailable'`` rather than treating the structure
    as valid or invalid.
    """

    result = _empty_audit(value)
    if is_sentinel(value):
        result.update(
            {
                "is_sentinel": True,
                "status": "sentinel",
                "rdkit_status": "not_checked",
            }
        )
        return result

    text = str(value).strip()
    result["normalized_input"] = text
    if not use_rdkit:
        result.update({"status": "rdkit_not_requested", "rdkit_status": "not_requested"})
        return result

    import_error = None
    if chem_module is None:
        chem_module, detected_version, import_error = _load_rdkit_chem()
        rdkit_version = rdkit_version or detected_version
    if chem_module is None:
        result.update(
            {
                "status": "rdkit_unavailable",
                "rdkit_status": "unavailable",
                "error": import_error or "RDKit Chem module was not provided",
            }
        )
        return result

    result.update({"rdkit_status": "available", "rdkit_version": rdkit_version})
    try:
        molecule = chem_module.MolFromSmiles(text, sanitize=False)
    except Exception as exc:  # RDKit raises several exception types for bad input.
        result.update({"status": "parse_failed", "parse_ok": False, "error": str(exc)})
        return result
    if molecule is None:
        result.update(
            {"status": "parse_failed", "parse_ok": False, "error": "MolFromSmiles returned None"}
        )
        return result
    result["parse_ok"] = True

    try:
        chem_module.SanitizeMol(molecule)
    except Exception as exc:
        result.update(
            {"status": "sanitize_failed", "sanitize_ok": False, "error": str(exc)}
        )
        return result
    result["sanitize_ok"] = True

    try:
        canonical = chem_module.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception as exc:
        result.update({"status": "canonicalization_failed", "error": str(exc)})
        return result
    if not canonical:
        result.update(
            {"status": "canonicalization_failed", "error": "MolToSmiles returned an empty value"}
        )
        return result
    result["canonical_smiles"] = canonical

    if not check_roundtrip:
        result["status"] = "passed"
        return result

    try:
        roundtrip_molecule = chem_module.MolFromSmiles(canonical, sanitize=True)
        if roundtrip_molecule is None:
            raise ValueError("round-trip MolFromSmiles returned None")
        roundtrip_smiles = chem_module.MolToSmiles(
            roundtrip_molecule,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception as exc:
        result.update({"status": "roundtrip_failed", "roundtrip_ok": False, "error": str(exc)})
        return result

    result["roundtrip_smiles"] = roundtrip_smiles
    result["roundtrip_ok"] = roundtrip_smiles == canonical
    result["status"] = "passed" if result["roundtrip_ok"] else "roundtrip_mismatch"
    return result


def audit_smiles_many(
    values: Iterable[object],
    *,
    check_roundtrip: bool = True,
    use_rdkit: bool = True,
) -> list[dict[str, Any]]:
    """Audit multiple values while importing RDKit at most once."""

    chem_module = None
    version = None
    import_error = None
    if use_rdkit:
        chem_module, version, import_error = _load_rdkit_chem()
    if use_rdkit and chem_module is None:
        audits: list[dict[str, Any]] = []
        for value in values:
            audit = _empty_audit(value)
            if is_sentinel(value):
                audit.update({"is_sentinel": True, "status": "sentinel"})
            else:
                audit.update(
                    {
                        "normalized_input": str(value).strip(),
                        "status": "rdkit_unavailable",
                        "rdkit_status": "unavailable",
                        "error": import_error or "RDKit Chem module was not provided",
                    }
                )
            audits.append(audit)
        return audits
    return [
        audit_smiles_representation(
            value,
            check_roundtrip=check_roundtrip,
            use_rdkit=use_rdkit,
            chem_module=chem_module,
            rdkit_version=version,
        )
        for value in values
    ]


# Short alias for callers that do not need to distinguish future representation types.
audit_smiles = audit_smiles_representation
