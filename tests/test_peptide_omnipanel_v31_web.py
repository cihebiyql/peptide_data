from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path

import pytest

from peptide_omnipanel.v31_peptide18 import ENDPOINT_ORDER
from peptide_omnipanel.v31_web import (
    ENDPOINT_LABELS,
    EVIDENCE_COLUMNS,
    TABLE_COLUMNS,
    batch_prediction_table,
    batch_summary_html,
    build_demo,
    endpoint_cards,
    endpoint_evidence_table,
    history_choices,
    history_table,
    history_task_table,
    prediction_table,
    score_figure,
    selected_detail,
    summary_html,
    task_progress_html,
    task_table,
)
from peptide_omnipanel.v31_web_batch import (
    BatchInputError,
    parse_batch_inputs,
    write_batch_exports,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/peptide_omnipanel_v31_peptide18_example_prediction.json"
LOCAL_BUNDLE_MANIFEST = (
    ROOT / "models/peptide_omnipanel_v31/release_peptide18_a3_20260722/bundle_manifest.json"
)
PUBLISHED_BUNDLE_MANIFEST = ROOT / "docs/peptide_omnipanel_v31_peptide18_bundle_manifest.json"


def _bundle_manifest_path() -> Path:
    return LOCAL_BUNDLE_MANIFEST if LOCAL_BUNDLE_MANIFEST.is_file() else PUBLISHED_BUNDLE_MANIFEST


def _example() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _batch_state() -> dict:
    return {
        "batch_id": "test-batch",
        "created_at": "2026-07-23T00:00:00+00:00",
        "completed_at": "2026-07-23T00:00:03+00:00",
        "jobs": [
            {
                "job_id": "任务-001-test",
                "display_name": "候选肽-1",
                "sequence": "GIGKFLHSAKKFGKAFVGEIMNS",
                "source": "文本输入",
                "input_index": 1,
                "status": "已完成",
                "progress": "18 / 18",
                "elapsed_seconds": 2.5,
                "message": "18 个本地预测头全部完成",
            }
        ],
        "rejected_inputs": [],
        "results": {"任务-001-test": _example()},
    }


def test_plain_single_and_named_multi_sequence_input() -> None:
    single, rejected = parse_batch_inputs("GIGKFLHSAKKFGKAFVGEIMNS", None)
    assert rejected == []
    assert len(single) == 1
    assert single[0].display_name == "序列-1"
    assert single[0].sequence == "GIGKFLHSAKKFGKAFVGEIMNS"

    multiple, rejected = parse_batch_inputs("候选肽A\tRRWWRF\n候选肽B\tACDEFGHIK", None)
    assert rejected == []
    assert [row.display_name for row in multiple] == ["候选肽A", "候选肽B"]
    assert [row.sequence for row in multiple] == ["RRWWRF", "ACDEFGHIK"]


def test_multirecord_fasta_text_and_file_parsing(tmp_path: Path) -> None:
    text = ">peptide_A description\nACDEF\nGHIK\n>peptide_B\nRRWWRF\n"
    records, rejected = parse_batch_inputs(text, None)
    assert rejected == []
    assert [(row.display_name, row.sequence) for row in records] == [
        ("peptide_A description", "ACDEFGHIK"),
        ("peptide_B", "RRWWRF"),
    ]

    fasta = tmp_path / "batch.fasta"
    fasta.write_text(">file_A\nMKWVTFISLL\n>file_B\nKWKLFKKIEK\n", encoding="utf-8")
    records, rejected = parse_batch_inputs("ACDEFG", fasta)
    assert rejected == []
    assert len(records) == 3
    assert records[1].source == "文件：batch.fasta"


def test_csv_tsv_columns_and_invalid_rows_are_auditable(tmp_path: Path) -> None:
    csv_path = tmp_path / "peptides.csv"
    csv_path.write_text(
        "名称,序列\n有效肽,ACDEFGHIK\n无效肽,ACDX\n", encoding="utf-8"
    )
    records, rejected = parse_batch_inputs(None, csv_path)
    assert [row.display_name for row in records] == ["有效肽"]
    assert len(rejected) == 1
    assert rejected[0].display_name == "无效肽"
    assert "canonical" in rejected[0].reason

    tsv_path = tmp_path / "peptides.tsv"
    tsv_path.write_text("id\tpeptide_sequence\nt1\tRRWWRF\nt2\tACDEFG\n", encoding="utf-8")
    records, rejected = parse_batch_inputs(None, tsv_path)
    assert rejected == []
    assert [row.display_name for row in records] == ["t1", "t2"]


def test_batch_input_is_bounded_and_requires_at_least_one_valid_sequence() -> None:
    with pytest.raises(BatchInputError, match="没有可执行"):
        parse_batch_inputs("ACDX", None)
    too_many = "\n".join("ACDEFG" for _ in range(33))
    with pytest.raises(BatchInputError, match="最多 32"):
        parse_batch_inputs(too_many, None)


def test_main_web_table_is_exact_ordered_chinese_18_without_repeated_warning_columns() -> None:
    frame = prediction_table(_example())
    assert frame.shape == (18, len(TABLE_COLUMNS))
    assert frame["端点"].tolist() == [ENDPOINT_LABELS[value] for value in ENDPOINT_ORDER]
    assert "验证状态" not in frame
    assert "信心标记" not in frame
    assert set(frame["输出类型"]) <= {"0–1 概率", "0–1 排序分数", "连续值回归"}
    assert not any(
        value in {"traditional", "esm_sequence", "structure", "multimodal"}
        for value in frame["特征路由"]
    )


def test_main_cards_are_deemphasized_but_raw_and_explanation_keep_evidence() -> None:
    result = _example()
    cards = endpoint_cards(result)
    summary_result = summary_html(result, "候选肽-1")
    state = _batch_state()
    summary = batch_summary_html(state)
    progress = task_progress_html(state)
    assert cards.count("endpoint-card") == 18
    assert "低置信度" not in cards
    assert "仅供科研" not in cards
    assert "未独立验证" not in cards
    assert "网站与结果说明" in summary_result
    assert all(row["research_only"] for row in result["endpoints"])
    assert all(row["validation_status"] == "not_validated" for row in result["endpoints"])
    assert "有效任务" in summary
    assert "本批次已执行完毕" in progress
    assert "100%" in progress


def test_explanation_evidence_table_keeps_all_endpoint_boundaries() -> None:
    manifest = json.loads(_bundle_manifest_path().read_text(encoding="utf-8"))
    frame = endpoint_evidence_table(manifest)
    assert frame.shape == (18, len(EVIDENCE_COLUMNS))
    assert set(frame["使用范围"]) == {"仅供科研"}
    assert set(frame["验证阶段"]) == {"未独立验证"}
    assert "低置信度" in set(frame["信心标记"])


def test_batch_tables_selection_and_exports(tmp_path: Path) -> None:
    state = _batch_state()
    aggregate = batch_prediction_table(state)
    tasks = task_table(state)
    assert aggregate.shape[0] == 18
    assert set(aggregate["任务"]) == {"任务-001-test"}
    assert tasks.loc[0, "状态"] == "已完成"
    detail = selected_detail(state, "任务-001-test")
    assert "当前序列" in detail[0]
    assert detail[2].shape[0] == 18

    json_path, tsv_path, zip_path = write_batch_exports(state, aggregate, tmp_path)
    assert Path(json_path).is_file() and Path(tsv_path).is_file() and Path(zip_path).is_file()
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert payload["scope"] == "internal_research_only"
    assert len(payload["results"]) == 1
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    assert any(name.startswith("individual/") and name.endswith(".json") for name in names)


def test_history_views_show_batches_and_submitted_sequences() -> None:
    metadata = [
        {
            "batch_id": "test-batch",
            "created_at": "2026-07-23T00:00:00+00:00",
            "completed_at": "2026-07-23T00:00:03+00:00",
            "sequence_count": 1,
            "completed_count": 1,
            "failed_count": 0,
            "rejected_count": 0,
            "prediction_count": 18,
            "sequence_names": ["候选肽-1"],
        }
    ]
    batches = history_table(metadata)
    choices = history_choices(metadata)
    submitted = history_task_table(_batch_state())
    assert batches.loc[0, "批次编号"] == "test-batch"
    assert batches.loc[0, "预测数值数"] == 18
    assert choices[0][1] == "test-batch"
    assert submitted.loc[0, "标准序列"] == "GIGKFLHSAKKFGKAFVGEIMNS"
    assert submitted.loc[0, "端点数"] == 18


def test_score_plot_contains_only_probability_and_ranking_outputs() -> None:
    result = _example()
    expected = sum(
        row["value_kind"] in {"probability", "ranking_score"} for row in result["endpoints"]
    )
    figure = score_figure(result)
    assert len(figure.data) == 1
    assert len(figure.data[0].x) == expected
    assert list(figure.layout.xaxis.range) == [0, 1.08]
    assert figure.layout.xaxis.title.text == "模型输出"


def test_four_page_demo_persists_and_reopens_a_completed_history_batch(
    tmp_path: Path,
) -> None:
    manifest_path = _bundle_manifest_path()

    class FakePredictor:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        @staticmethod
        def runtime_info() -> dict:
            return {"device_name": "测试 GPU", "model_load_seconds": 0.1}

        @staticmethod
        def predict(sequence: str) -> dict:
            result = copy.deepcopy(_example())
            result["input"]["canonical_sequence"] = sequence
            return result

    demo = build_demo(
        predictor=FakePredictor(),
        export_dir=tmp_path / "exports",
        history_dir=tmp_path / "history",
    )
    event_names = [dependency.name for dependency in demo.fns.values()]
    assert "run_batch" in event_names
    run_event = next(
        dependency.fn for dependency in demo.fns.values() if dependency.name == "run_batch"
    )
    outputs = list(run_event("GIGKFLHSAKKFGKAFVGEIMNS", None, "browser-a"))
    final = outputs[-1]
    assert len(final) == 20
    assert final[0]["completed_at"] is not None
    assert final[15] == "browser-a"
    assert final[16].shape[0] == 1
    batch_id = final[0]["batch_id"]

    open_event = next(
        dependency.fn
        for dependency in demo.fns.values()
        if dependency.name == "open_history_batch"
    )
    restored = open_event("browser-a", batch_id)
    assert len(restored) == 15
    assert restored[0]["batch_id"] == batch_id
    assert restored[1]["selected"] == "prediction"
