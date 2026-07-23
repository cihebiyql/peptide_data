"""Batch input parsing and export contracts for the V31 research website."""

from __future__ import annotations

import csv
import io
import json
import re
import uuid
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from peptide_omnipanel.v31_peptide18 import V31Peptide18Error, canonical_sequence

MAX_BATCH_SIZE = 32
MAX_UPLOAD_BYTES = 1_000_000
SEQUENCE_COLUMN_NAMES = {
    "sequence",
    "seq",
    "peptide",
    "peptide_sequence",
    "peptidesequence",
    "amino_acid_sequence",
    "序列",
    "肽序列",
    "氨基酸序列",
}
ID_COLUMN_NAMES = {
    "id",
    "name",
    "identifier",
    "sequence_id",
    "peptide_id",
    "名称",
    "编号",
    "序列编号",
}


class BatchInputError(ValueError):
    """Raised when no safe, bounded batch can be constructed."""


@dataclass(frozen=True)
class BatchSequence:
    job_id: str
    display_name: str
    sequence: str
    source: str
    input_index: int


@dataclass(frozen=True)
class RejectedSequence:
    display_name: str
    raw_sequence: str
    source: str
    reason: str
    input_index: int


def _normalized_column(value: Any) -> str:
    return re.sub(r"[\s\-]+", "_", str(value).strip().lower())


def _safe_name(value: Any, fallback: str) -> str:
    compact = " ".join(str(value or "").replace("\x00", "").split())
    return (compact or fallback)[:80]


def _fasta_records(text: str, source: str) -> list[tuple[str, str, str]]:
    records: list[tuple[str, str, str]] = []
    name: str | None = None
    fragments: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append(
                    (
                        _safe_name(name, f"序列-{len(records) + 1}"),
                        "".join(fragments),
                        source,
                    )
                )
            name = line[1:].strip() or f"序列-{len(records) + 1}"
            fragments = []
        elif name is None:
            raise BatchInputError("FASTA 内容在第一个 > 标题之前出现了序列")
        else:
            fragments.append(line)
    if name is not None:
        records.append((_safe_name(name, f"序列-{len(records) + 1}"), "".join(fragments), source))
    return records


def _delimited_records(
    text: str,
    *,
    source: str,
    delimiter: str,
) -> list[tuple[str, str, str]]:
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [[cell.strip() for cell in row] for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return []
    normalized_header = [_normalized_column(value) for value in rows[0]]
    sequence_index = next(
        (index for index, value in enumerate(normalized_header) if value in SEQUENCE_COLUMN_NAMES),
        None,
    )
    id_index = next(
        (index for index, value in enumerate(normalized_header) if value in ID_COLUMN_NAMES),
        None,
    )
    if sequence_index is not None:
        data_rows = rows[1:]
    elif all(len(row) >= 2 for row in rows):
        sequence_index = 1
        id_index = 0
        data_rows = rows
    elif all(len(row) == 1 for row in rows):
        sequence_index = 0
        id_index = None
        data_rows = rows
    else:
        raise BatchInputError(
            "CSV/TSV 需要 sequence/序列列，或使用“名称 + 序列”两列格式"
        )
    records: list[tuple[str, str, str]] = []
    for row_index, row in enumerate(data_rows, start=1):
        if sequence_index >= len(row):
            records.append((f"第 {row_index} 行", "", source))
            continue
        name = (
            row[id_index]
            if id_index is not None and id_index < len(row) and row[id_index]
            else f"序列-{row_index}"
        )
        records.append((_safe_name(name, f"序列-{row_index}"), row[sequence_index], source))
    return records


def _plain_records(text: str, source: str) -> list[tuple[str, str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    records: list[tuple[str, str, str]] = []
    for index, line in enumerate(lines, start=1):
        if "\t" in line:
            name, sequence = line.split("\t", 1)
        elif line.count(",") == 1:
            possible_name, possible_sequence = line.split(",", 1)
            if possible_name and possible_sequence:
                name, sequence = possible_name, possible_sequence
            else:
                name, sequence = f"序列-{index}", line
        else:
            name, sequence = f"序列-{index}", line
        records.append((_safe_name(name, f"序列-{index}"), sequence, source))
    return records


def _records_from_text(text: str, *, source: str, suffix: str = "") -> list[tuple[str, str, str]]:
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith(">"):
        return _fasta_records(stripped, source)
    suffix = suffix.lower()
    if suffix == ".csv":
        return _delimited_records(stripped, source=source, delimiter=",")
    if suffix == ".tsv":
        return _delimited_records(stripped, source=source, delimiter="\t")
    first_line = next((line for line in stripped.splitlines() if line.strip()), "")
    normalized_cells = [_normalized_column(value) for value in re.split(r"[,\t]", first_line)]
    if any(value in SEQUENCE_COLUMN_NAMES for value in normalized_cells):
        delimiter = "\t" if "\t" in first_line else ","
        return _delimited_records(stripped, source=source, delimiter=delimiter)
    return _plain_records(stripped, source)


def parse_batch_inputs(
    text_input: str | None,
    upload_path: str | Path | None,
    *,
    max_batch_size: int = MAX_BATCH_SIZE,
) -> tuple[list[BatchSequence], list[RejectedSequence]]:
    """Parse text plus one optional file into a bounded, auditable task list."""

    raw_records = _records_from_text(str(text_input or ""), source="文本输入")
    if upload_path:
        path = Path(upload_path)
        if not path.is_file():
            raise BatchInputError(f"上传文件不存在：{path.name}")
        if path.stat().st_size > MAX_UPLOAD_BYTES:
            raise BatchInputError("上传文件超过 1 MB 安全限制")
        try:
            file_text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as error:
            raise BatchInputError("上传文件必须是 UTF-8 文本") from error
        suffix = path.suffix.lower()
        if suffix not in {".fa", ".faa", ".fasta", ".txt", ".csv", ".tsv"}:
            raise BatchInputError("仅支持 FASTA/FA/FAA/TXT/CSV/TSV 文件")
        raw_records.extend(
            _records_from_text(file_text, source=f"文件：{path.name}", suffix=suffix)
        )
    if not raw_records:
        raise BatchInputError("请粘贴至少一条序列，或上传序列文件")

    accepted: list[BatchSequence] = []
    rejected: list[RejectedSequence] = []
    for input_index, (name, raw_sequence, source) in enumerate(raw_records, start=1):
        try:
            sequence = canonical_sequence(raw_sequence)
        except V31Peptide18Error as error:
            rejected.append(
                RejectedSequence(
                    display_name=name,
                    raw_sequence=str(raw_sequence)[:120],
                    source=source,
                    reason=str(error),
                    input_index=input_index,
                )
            )
            continue
        accepted.append(
            BatchSequence(
                job_id=f"任务-{len(accepted) + 1:03d}-{uuid.uuid4().hex[:6]}",
                display_name=name,
                sequence=sequence,
                source=source,
                input_index=input_index,
            )
        )
    if len(accepted) > max_batch_size:
        raise BatchInputError(
            f"一个批次最多 {max_batch_size} 条有效序列；当前为 {len(accepted)} 条"
        )
    if not accepted:
        details = rejected[0].reason if rejected else "未识别到有效序列"
        raise BatchInputError(f"没有可执行的序列：{details}")
    return accepted, rejected


def batch_export_payload(batch_state: dict[str, Any]) -> dict[str, Any]:
    """Remove UI-only objects while retaining all prediction provenance."""

    return {
        "schema_version": "peptide_omnipanel_v31_web_batch_v1",
        "scope": "internal_research_only",
        "batch_id": batch_state["batch_id"],
        "created_at": batch_state["created_at"],
        "completed_at": batch_state.get("completed_at"),
        "jobs": batch_state["jobs"],
        "rejected_inputs": batch_state["rejected_inputs"],
        "results": batch_state["results"],
        "warnings": [
            "仅供科研",
            "18 个端点当前均未通过独立外部验证",
            "裸序列被解释为未修饰线性天然 L-氨基酸肽",
        ],
    }


def write_batch_exports(
    batch_state: dict[str, Any],
    aggregate_table: pd.DataFrame,
    export_dir: Path,
) -> tuple[str, str, str]:
    """Write one JSON, one flat TSV and one per-sequence ZIP package."""

    export_dir.mkdir(parents=True, exist_ok=True)
    stem = f"peptide18_batch_{batch_state['batch_id']}"
    json_path = export_dir / f"{stem}.json"
    tsv_path = export_dir / f"{stem}.tsv"
    zip_path = export_dir / f"{stem}.zip"
    payload = batch_export_payload(batch_state)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate_table.to_csv(tsv_path, sep="\t", index=False)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname=json_path.name)
        archive.write(tsv_path, arcname=tsv_path.name)
        for job in batch_state["jobs"]:
            result = batch_state["results"].get(job["job_id"])
            if result is None:
                continue
            safe_job = re.sub(r"[^A-Za-z0-9_.-]+", "_", job["job_id"])
            archive.writestr(
                f"individual/{safe_job}.json",
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
    return str(json_path), str(tsv_path), str(zip_path)


def dataclass_rows(values: list[BatchSequence] | list[RejectedSequence]) -> list[dict[str, Any]]:
    return [asdict(value) for value in values]
