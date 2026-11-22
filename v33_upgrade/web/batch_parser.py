"""Robust batch file / text parser supporting CSV, TSV, TXT, and FASTA up to 500 molecules."""

from __future__ import annotations

import csv
import io
import re
from typing import Optional

from .chem_utils import AA_ONLY_RE, clean_sequence_text, is_canonical_sequence, parse_smiles_safe
from .schemas import BatchItemInput, BatchParsedRow, BatchParseResponse

MAX_BATCH_SIZE = 500


def parse_batch_text(
    raw_text: str,
    filename: Optional[str] = None,
    drug_lookup_fn=None,
) -> BatchParseResponse:
    """Parse pasted text or uploaded CSV/TSV/TXT/FASTA content into validated molecule rows."""
    text = (raw_text or "").strip()
    if not text:
        return BatchParseResponse(total=0, valid_count=0, invalid_count=0, rows=[])

    ext = (filename or "").lower().rsplit(".", 1)[-1] if filename and "." in filename else ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return BatchParseResponse(total=0, valid_count=0, invalid_count=0, rows=[])

    parsed_items: list[tuple[str, Optional[str], Optional[str], Optional[str]]] = []

    # 1. FASTA detection
    if ext in ("fasta", "fa", "faa") or any(ln.startswith(">") for ln in lines[:3]):
        cur_name: Optional[str] = None
        cur_seq_parts: list[str] = []

        def flush_fasta():
            nonlocal cur_name, cur_seq_parts
            if cur_seq_parts:
                body = "".join(cur_seq_parts).strip()
                name = cur_name or f"Peptide_{len(parsed_items) + 1}"
                parsed_items.append((name, body, None, None))
            cur_name = None
            cur_seq_parts = []

        for ln in lines:
            if ln.startswith(">"):
                flush_fasta()
                cur_name = ln[1:].strip().split()[0] if len(ln) > 1 else f"Peptide_{len(parsed_items) + 1}"
            else:
                cur_seq_parts.append(ln)
        flush_fasta()
    else:
        # 2. Delimited table (CSV / TSV) vs plain 1-per-line
        first = lines[0]
        delim: Optional[str] = None
        if ext == "csv":
            delim = ","
        elif ext == "tsv":
            delim = "\t"
        elif "\t" in first:
            delim = "\t"
        elif "," in first and not parse_smiles_safe(first):
            # Only treat comma as delimiter when first line isn't itself a SMILES
            delim = ","

        has_header = False
        header_cols: list[str] = []
        if delim:
            reader = list(csv.reader(io.StringIO(text), delimiter=delim))
            if reader:
                first_row = [c.strip().lower() for c in reader[0]]
                header_tokens = {"name", "id", "drug", "molecule", "compound", "sequence", "seq", "smiles", "query", "名称", "序列", "结构"}
                if any(any(tok in col for tok in header_tokens) for col in first_row):
                    has_header = True
                    header_cols = first_row
                    data_rows = reader[1:]
                else:
                    data_rows = reader

                def col_idx(keys: tuple[str, ...]) -> Optional[int]:
                    for i, c in enumerate(header_cols):
                        if any(k == c or k in c for k in keys):
                            return i
                    return None

                i_name = col_idx(("name", "id", "drug", "compound", "molecule", "名称", "编号")) if has_header else None
                i_seq = col_idx(("sequence", "seq", "序列")) if has_header else None
                i_smi = col_idx(("smiles", "smi", "结构")) if has_header else None
                i_query = col_idx(("query", "input", "输入")) if has_header else None

                for row in data_rows:
                    cells = [c.strip() for c in row]
                    if not any(cells):
                        continue
                    if has_header:
                        name = cells[i_name] if i_name is not None and i_name < len(cells) and cells[i_name] else f"M{len(parsed_items) + 1}"
                        seq = cells[i_seq] if i_seq is not None and i_seq < len(cells) and cells[i_seq] else None
                        smi = cells[i_smi] if i_smi is not None and i_smi < len(cells) and cells[i_smi] else None
                        qry = cells[i_query] if i_query is not None and i_query < len(cells) and cells[i_query] else None
                        parsed_items.append((name, qry, seq, smi))
                    else:
                        if len(cells) >= 2:
                            # Heuristic: [name, query] or [name, seq, smi]
                            if len(cells) >= 3 and is_canonical_sequence(cells[1]) and parse_smiles_safe(cells[2]):
                                parsed_items.append((cells[0] or f"M{len(parsed_items) + 1}", None, cells[1], cells[2]))
                            else:
                                parsed_items.append((cells[0] or f"M{len(parsed_items) + 1}", cells[1], None, None))
                        else:
                            parsed_items.append((f"M{len(parsed_items) + 1}", cells[0], None, None))
        else:
            for ln in lines:
                parsed_items.append((f"M{len(parsed_items) + 1}", ln, None, None))

    truncated = len(parsed_items) > MAX_BATCH_SIZE
    parsed_items = parsed_items[:MAX_BATCH_SIZE]

    rows: list[BatchParsedRow] = []
    valid_count = 0
    for idx, (name, qry, seq, smi) in enumerate(parsed_items, start=1):
        row = validate_parsed_entry(idx, name, qry, seq, smi, drug_lookup_fn=drug_lookup_fn)
        if row.valid:
            valid_count += 1
        rows.append(row)

    return BatchParseResponse(
        total=len(rows),
        valid_count=valid_count,
        invalid_count=len(rows) - valid_count,
        truncated=truncated,
        max_batch=MAX_BATCH_SIZE,
        rows=rows,
    )


def validate_parsed_entry(
    index: int,
    name: str,
    query: Optional[str],
    sequence: Optional[str],
    smiles: Optional[str],
    drug_lookup_fn=None,
) -> BatchParsedRow:
    """Validate and classify a single parsed batch item."""
    seq = clean_sequence_text(sequence) if sequence else None
    smi = (smiles or "").strip() or None
    q = (query or "").strip() or None

    if not seq and not smi and q:
        if drug_lookup_fn is not None:
            drug = drug_lookup_fn(q)
            if drug is not None:
                return BatchParsedRow(
                    index=index,
                    name=drug["display_name"] if name.startswith("M") else name,
                    detected_type="drug_name",
                    sequence=drug.get("sequence"),
                    smiles=drug.get("smiles"),
                    valid=True,
                )
        if is_canonical_sequence(q):
            seq = clean_sequence_text(q)
            return BatchParsedRow(
                index=index,
                name=name,
                detected_type="sequence",
                sequence=seq,
                smiles=None,
                valid=True,
            )
        if parse_smiles_safe(q) is not None:
            return BatchParsedRow(
                index=index,
                name=name,
                detected_type="smiles",
                sequence=None,
                smiles=q,
                valid=True,
            )
        return BatchParsedRow(
            index=index,
            name=name,
            detected_type="invalid",
            valid=False,
            error="Unrecognized drug name, 20-AA sequence, or SMILES",
        )

    if seq and not AA_ONLY_RE.fullmatch(seq):
        return BatchParsedRow(
            index=index,
            name=name,
            detected_type="invalid",
            sequence=seq,
            smiles=smi,
            valid=False,
            error="Sequence contains non-canonical amino acid symbols",
        )
    if smi and parse_smiles_safe(smi) is None:
        return BatchParsedRow(
            index=index,
            name=name,
            detected_type="invalid",
            sequence=seq,
            smiles=smi,
            valid=False,
            error="Invalid SMILES syntax",
        )
    if not seq and not smi:
        return BatchParsedRow(
            index=index,
            name=name,
            detected_type="invalid",
            valid=False,
            error="Empty input row",
        )

    det = "dual" if (seq and smi) else ("sequence" if seq else "smiles")
    return BatchParsedRow(
        index=index,
        name=name,
        detected_type=det,
        sequence=seq,
        smiles=smi,
        valid=True,
    )


def items_from_batch_request(items: list[BatchItemInput], drug_lookup_fn=None) -> list[BatchParsedRow]:
    """Convert explicit API BatchItemInput list to validated BatchParsedRow list."""
    out: list[BatchParsedRow] = []
    for idx, it in enumerate(items[:MAX_BATCH_SIZE], start=1):
        name = (it.name or "").strip() or f"M{idx}"
        out.append(
            validate_parsed_entry(
                index=idx,
                name=name,
                query=it.query,
                sequence=it.sequence,
                smiles=it.smiles,
                drug_lookup_fn=drug_lookup_fn,
            )
        )
    return out
