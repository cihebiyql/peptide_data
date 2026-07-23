# ruff: noqa: E501
"""Peptide-only V31 Chinese batch research website."""

from __future__ import annotations

import copy
import html
import os
import time
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from peptide_omnipanel.v31_peptide18 import ENDPOINT_ORDER, V31Peptide18Error
from peptide_omnipanel.v31_web_batch import (
    MAX_BATCH_SIZE,
    BatchInputError,
    dataclass_rows,
    parse_batch_inputs,
    write_batch_exports,
)
from peptide_omnipanel.v31_web_history import HistoryStore, HistoryStoreError

ENDPOINT_LABELS = {
    "LogD7.4": "LogD 7.4（pH 7.4 分配系数）",
    "solubility": "溶解性",
    "F": "F（绝对生物利用度）",
    "T1/2": "T₁/₂（稳定至少 1 小时的概率）",
    "PPB": "PPB（血浆蛋白结合）",
    "CL": "CL（系统清除率）",
    "Vd": "Vd（表观分布容积）",
    "BBB": "BBB（血脑屏障穿透）",
    "permeability": "渗透性",
    "overall_peptide_toxicity": "肽类总体毒性",
    "cytotoxicity": "细胞毒性",
    "hemolysis": "溶血性",
    "HC50": "HC50（50% 溶血浓度）",
    "cell_penetration": "细胞穿透能力",
    "membrane_retention": "膜滞留",
    "human_plasma_stability": "人血浆稳定性",
    "mouse_plasma_stability": "小鼠血浆稳定性",
    "intestinal_stability": "肠道/蛋白酶稳定性",
}

ENDPOINT_GROUPS = {
    "ADME / PK（吸收、分布、代谢、排泄与药代）": ENDPOINT_ORDER[:9],
    "安全性": ENDPOINT_ORDER[9:13],
    "细胞与膜行为": ENDPOINT_ORDER[13:15],
    "生物稳定性": ENDPOINT_ORDER[15:],
}

VALUE_KIND_LABELS = {
    "probability": "0–1 概率",
    "ranking_score": "0–1 排序分数",
    "regression": "连续值回归",
}

FEATURE_ROUTE_LABELS = {
    "traditional": "传统肽序列特征",
    "esm_sequence": "传统特征 + ESM2 序列表征",
    "structure": "HELM/SMILES + Morgan/RDKit 二维结构特征",
    "multimodal": "序列 + ESM2 + 二维结构多模态融合",
}

EVIDENCE_LABELS = {
    "direct_measured_structure_conditioned": "直接测量；结构/条件敏感",
    "source_defined_binary": "数据源定义的二分类",
    "direct_measured": "直接测量",
    "weak_binary": "弱标签二分类",
    "direct_source_resolved_binary": "来源已解析的直接二分类",
    "direct_measured_single_source": "单来源直接测量",
    "positive_unlabeled": "正样本-未标注排序",
    "direct_measured_structure_only": "直接测量；仅结构特征",
}

CONDITION_LABELS = {
    "pH=7.4": "pH = 7.4",
    "source-defined solvent-conditioned binary benchmark": "数据源定义、溶剂条件相关的二分类基准",
    "mixed species and administration routes; route returned per training contract": "混合物种与给药途径；按训练合同返回结果",
    "source-defined stability with threshold t1/2 >= 1 hour": "数据源定义的稳定性；T₁/₂ 阈值 ≥ 1 小时",
    "species and plasma/serum heterogeneous": "物种及血浆/血清条件存在异质性",
    "absolute systemic clearance only; CL/F and CLint excluded": "仅绝对系统清除率；已排除 CL/F 和 CLint",
    "absolute volume_of_distribution only; apparent V/F and compartment volumes excluded": "仅绝对分布容积；已排除表观 V/F 和房室容积",
    "compiled-positive/rule-negative peptide benchmark": "汇编阳性 + 规则阴性的肽类基准",
    "frozen dominant PAMPA task mlv1.permeability.regression.apparent_permeability.e74490767e77cabe": "冻结的主要 PAMPA 表观渗透率任务",
    "ToxinPred3-derived weak peptide benchmark": "来自 ToxinPred3 的肽类弱标签基准",
    "mixed cell-line/concentration/time source-defined benchmark": "混合细胞系、浓度和时间的数据源定义基准",
    "human erythrocytes; >=20% hemolysis at 50 uM": "人红细胞；50 μM 时溶血率 ≥ 20%",
    "mammalian erythrocytes; source species not row-resolved": "哺乳动物红细胞；来源物种未解析到单行",
    "positive membership versus peptide-domain unlabeled background": "阳性成员与肽类领域未标注背景的排序任务",
    "single modified-peptide PAMPA series": "单一修饰肽 PAMPA 系列",
    "plasma_serum_stability semantics with explicit human plasma context; systemic terminal PK excluded": "明确人血浆语境的血浆/血清稳定性；已排除系统终末药代",
    "plasma_serum_stability semantics with explicit mouse plasma context; systemic PK, PPB and protease-condition rows excluded": "明确小鼠血浆语境；已排除系统药代、PPB 和蛋白酶条件行",
    "protease_intestinal_stability semantic family; not duplicated as protease_stability": "蛋白酶/肠道稳定性语义家族；未重复计入通用蛋白酶稳定性",
}

UNIT_LABELS = {
    "dimensionless_log10": "无量纲 log10",
    "score_0_1": "0–1 分数",
    "fraction": "0–1 比例",
    "probability": "0–1 概率",
    "fraction_bound": "0–1 结合比例",
    "mL/min/kg": "mL/min/kg",
    "L/kg": "L/kg",
    "source_log10_permeability": "数据源定义的 log10 渗透率",
    "log10(mol/L)": "log10(mol/L)",
    "ranking_score_0_1": "0–1 排序分数",
    "fraction_retained": "0–1 滞留比例",
    "log10(h)": "log10(小时)",
}

TABLE_COLUMNS = [
    "端点",
    "预测值",
    "单位",
    "输出类型",
    "训练样本数",
    "开发集 OOF 指标",
    "证据通道",
    "特征路由",
    "修饰敏感",
    "条件合同",
]

EVIDENCE_COLUMNS = [
    "端点",
    "训练样本数",
    "开发集 OOF 指标",
    "证据通道",
    "特征路由",
    "信心标记",
    "使用范围",
    "验证阶段",
    "条件合同",
]

TASK_COLUMNS = ["任务", "序列名称", "来源", "长度", "状态", "进度", "用时（秒）", "信息"]

BATCH_COLUMNS = [
    "任务",
    "序列名称",
    "标准序列",
    "序列长度",
    *TABLE_COLUMNS,
]

HISTORY_COLUMNS = [
    "批次编号",
    "提交时间",
    "完成时间",
    "有效序列",
    "已完成",
    "失败",
    "拒绝",
    "预测数值数",
    "序列名称摘要",
]

HISTORY_TASK_COLUMNS = ["任务", "序列名称", "标准序列", "来源", "状态", "端点数"]


def _formatted_value(record: dict[str, Any]) -> str:
    value = float(record["value"])
    if record["value_kind"] in {"probability", "ranking_score"}:
        return f"{value:.4f}"
    magnitude = abs(value)
    if magnitude >= 1000 or (magnitude and magnitude < 0.001):
        return f"{value:.4e}"
    return f"{value:.4f}"


def _metric_summary(metrics: dict[str, Any]) -> str:
    preferred = ["roc_auc", "average_precision", "balanced_accuracy", "mae", "rmse", "r2"]
    labels = {
        "roc_auc": "ROC-AUC",
        "average_precision": "平均精确率",
        "balanced_accuracy": "平衡准确率",
        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R²",
    }
    values = []
    for key in preferred:
        if key in metrics:
            values.append(f"{labels[key]}={float(metrics[key]):.3f}")
    return "；".join(values)


def _localized_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "端点": ENDPOINT_LABELS[record["endpoint_id"]],
        "预测值": _formatted_value(record),
        "单位": UNIT_LABELS.get(record["unit"], record["unit"] or "—"),
        "输出类型": VALUE_KIND_LABELS.get(record["value_kind"], record["value_kind"]),
        "训练样本数": int(record["training_row_count"]),
        "开发集 OOF 指标": _metric_summary(record["oof_metrics"]),
        "证据通道": EVIDENCE_LABELS.get(record["evidence_lane"], record["evidence_lane"]),
        "特征路由": FEATURE_ROUTE_LABELS.get(record["feature_route"], record["feature_route"]),
        "验证状态": "未通过独立外部验证",
        "信心标记": "仅供科研 · 低置信度" if record["low_confidence"] else "仅供科研",
        "修饰敏感": "是" if record["modification_sensitive"] else "否",
        "条件合同": CONDITION_LABELS.get(
            record["condition_contract"], record["condition_contract"]
        ),
    }


def prediction_table(result: dict[str, Any]) -> pd.DataFrame:
    """Flatten one strict 18-endpoint result with Chinese evidence labels."""

    frame = pd.DataFrame([_localized_record(record) for record in result["endpoints"]], columns=TABLE_COLUMNS)
    if frame.shape != (18, len(TABLE_COLUMNS)):
        raise RuntimeError("网页结果表必须精确包含 18 个端点")
    return frame


def endpoint_evidence_table(manifest: dict[str, Any]) -> pd.DataFrame:
    """Build the detailed evidence matrix shown only on the explanation page."""

    rows = []
    for endpoint in ENDPOINT_ORDER:
        descriptor = manifest["endpoints"][endpoint]
        rows.append(
            {
                "端点": ENDPOINT_LABELS[endpoint],
                "训练样本数": int(descriptor["training_row_count"]),
                "开发集 OOF 指标": _metric_summary(descriptor["oof_metrics"]),
                "证据通道": EVIDENCE_LABELS.get(
                    descriptor["evidence_lane"], descriptor["evidence_lane"]
                ),
                "特征路由": FEATURE_ROUTE_LABELS.get(
                    descriptor["feature_route"], descriptor["feature_route"]
                ),
                "信心标记": "低置信度" if descriptor["low_confidence"] else "常规",
                "使用范围": "仅供科研" if descriptor["research_only"] else "其他",
                "验证阶段": (
                    "未独立验证"
                    if descriptor["validation_status"] == "not_validated"
                    else descriptor["validation_status"]
                ),
                "条件合同": CONDITION_LABELS.get(
                    descriptor["condition"], descriptor["condition"]
                ),
            }
        )
    return pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)


def batch_prediction_table(batch_state: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for job in batch_state.get("jobs", []):
        result = batch_state.get("results", {}).get(job["job_id"])
        if result is None:
            continue
        for record in result["endpoints"]:
            rows.append(
                {
                    "任务": job["job_id"],
                    "序列名称": job["display_name"],
                    "标准序列": job["sequence"],
                    "序列长度": len(job["sequence"]),
                    **_localized_record(record),
                }
            )
    return pd.DataFrame(rows, columns=BATCH_COLUMNS)


def endpoint_cards(result: dict[str, Any]) -> str:
    indexed = {row["endpoint_id"]: row for row in result["endpoints"]}
    sections: list[str] = []
    for group, endpoints in ENDPOINT_GROUPS.items():
        cards = []
        for endpoint in endpoints:
            row = indexed[endpoint]
            cards.append(
                "<article class='endpoint-card'>"
                f"<div class='endpoint-top'><span>{html.escape(ENDPOINT_LABELS[endpoint])}</span></div>"
                f"<div class='endpoint-value'>{html.escape(_formatted_value(row))}</div>"
                f"<div class='endpoint-unit'>{html.escape(UNIT_LABELS.get(row['unit'], str(row['unit'] or '—')))}</div>"
                f"<div class='endpoint-meta'>{html.escape(VALUE_KIND_LABELS.get(row['value_kind'], row['value_kind']))} · "
                f"训练 n={int(row['training_row_count'])}</div>"
                "</article>"
            )
        sections.append(
            f"<section class='endpoint-section'><h3>{html.escape(group)}</h3>"
            f"<div class='endpoint-grid'>{''.join(cards)}</div></section>"
        )
    return "".join(sections)


def summary_html(result: dict[str, Any], sequence_name: str | None = None) -> str:
    sequence = result["input"]["canonical_sequence"]
    runtime_seconds = result.get("runtime_seconds", result.get("web_elapsed_seconds"))
    runtime_text = "—" if runtime_seconds is None else f"{float(runtime_seconds):.2f} 秒"
    name_line = (
        f"<div class='selected-sequence'>当前序列：<b>{html.escape(sequence_name)}</b> · "
        f"<code>{html.escape(sequence)}</code></div>"
        if sequence_name
        else ""
    )
    return (
        name_line
        + "<div class='summary-grid'>"
        "<div class='summary-card'><span>动态预测端点</span><strong>18 / 18</strong></div>"
        "<div class='summary-card'><span>本地预测头</span><strong>18 个</strong></div>"
        f"<div class='summary-card'><span>序列长度</span><strong>{len(sequence)} aa</strong></div>"
        f"<div class='summary-card'><span>GPU 推理用时</span><strong>{runtime_text}</strong></div>"
        "</div>"
        "<div class='result-note'>数值的证据来源、适用边界与验证阶段请见“网站与结果说明”页。</div>"
    )


def score_figure(result: dict[str, Any]) -> go.Figure:
    records = [
        row for row in result["endpoints"] if row["value_kind"] in {"probability", "ranking_score"}
    ]
    figure = go.Figure(
        go.Bar(
            x=[float(row["value"]) for row in records],
            y=[ENDPOINT_LABELS[row["endpoint_id"]] for row in records],
            orientation="h",
            marker_color="#14b8a6",
            text=[f"{float(row['value']):.3f}" for row in records],
            textposition="outside",
            hovertemplate="%{y}<br>模型输出=%{x:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="分类/排序端点（0–1；不同任务分数不可直接横向比较）",
        xaxis={"range": [0, 1.08], "title": "模型输出"},
        yaxis={"autorange": "reversed"},
        height=max(360, 55 * len(records)),
        margin={"l": 35, "r": 35, "t": 70, "b": 45},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,18,36,0.03)",
        font={"family": "Inter, ui-sans-serif, system-ui", "color": "#17324d"},
        showlegend=False,
    )
    return figure


def task_table(batch_state: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for job in batch_state.get("jobs", []):
        rows.append(
            {
                "任务": job["job_id"],
                "序列名称": job["display_name"],
                "来源": job["source"],
                "长度": len(job["sequence"]),
                "状态": job["status"],
                "进度": job["progress"],
                "用时（秒）": "" if job.get("elapsed_seconds") is None else f"{job['elapsed_seconds']:.2f}",
                "信息": job.get("message", ""),
            }
        )
    for rejected in batch_state.get("rejected_inputs", []):
        rows.append(
            {
                "任务": "未创建",
                "序列名称": rejected["display_name"],
                "来源": rejected["source"],
                "长度": len(rejected["raw_sequence"]),
                "状态": "输入无效",
                "进度": "0 / 18",
                "用时（秒）": "",
                "信息": rejected["reason"],
            }
        )
    return pd.DataFrame(rows, columns=TASK_COLUMNS)


def batch_summary_html(batch_state: dict[str, Any]) -> str:
    jobs = batch_state.get("jobs", [])
    completed = sum(job["status"] == "已完成" for job in jobs)
    failed = sum(job["status"] == "失败" for job in jobs)
    rejected = len(batch_state.get("rejected_inputs", []))
    return (
        "<div class='batch-summary'>"
        f"<div><span>批次编号</span><strong>{html.escape(batch_state.get('batch_id', '尚未建立'))}</strong></div>"
        f"<div><span>有效任务</span><strong>{len(jobs)}</strong></div>"
        f"<div><span>已完成</span><strong>{completed}</strong></div>"
        f"<div><span>失败 / 拒绝</span><strong>{failed} / {rejected}</strong></div>"
        "</div>"
    )


def task_progress_html(batch_state: dict[str, Any]) -> str:
    jobs = batch_state.get("jobs", [])
    terminal = sum(job["status"] in {"已完成", "失败"} for job in jobs)
    percent = int(round(100 * terminal / len(jobs))) if jobs else 0
    current = next((job for job in jobs if job["status"] == "预测中"), None)
    if current:
        current_text = f"正在处理：{current['display_name']} · {current['progress']}"
        state_class = "running"
    elif jobs and terminal == len(jobs):
        current_text = "本批次已执行完毕"
        state_class = "done"
    else:
        current_text = "任务已创建，等待 GPU 队列"
        state_class = "queued"
    return (
        f"<aside class='task-progress {state_class}'>"
        "<div class='task-progress-top'><span>任务进展</span>"
        f"<strong>{terminal} / {len(jobs)}</strong></div>"
        f"<div class='progress-track'><i style='width:{percent}%'></i></div>"
        f"<div class='task-current'>{html.escape(current_text)}</div>"
        f"<div class='task-percent'>{percent}%</div></aside>"
    )


def ensure_browser_owner(owner_token: str | None) -> str:
    """Create a stable opaque token that Gradio persists in this browser."""

    return owner_token if owner_token else uuid.uuid4().hex


def _display_time(value: Any) -> str:
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return str(value)


def history_table(metadata: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in metadata:
        names = [str(value) for value in row.get("sequence_names", [])]
        summary = "、".join(names[:4]) + (f" 等 {len(names)} 条" if len(names) > 4 else "")
        rows.append(
            {
                "批次编号": row["batch_id"],
                "提交时间": _display_time(row.get("created_at")),
                "完成时间": _display_time(row.get("completed_at")),
                "有效序列": int(row.get("sequence_count", 0)),
                "已完成": int(row.get("completed_count", 0)),
                "失败": int(row.get("failed_count", 0)),
                "拒绝": int(row.get("rejected_count", 0)),
                "预测数值数": int(row.get("prediction_count", 0)),
                "序列名称摘要": summary or "—",
            }
        )
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def history_choices(metadata: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [
        (
            f"{_display_time(row.get('created_at'))} · {row['batch_id']} · "
            f"{int(row.get('sequence_count', 0))} 条序列",
            str(row["batch_id"]),
        )
        for row in metadata
    ]


def history_task_table(batch_state: dict[str, Any]) -> pd.DataFrame:
    results = batch_state.get("results", {})
    rows = [
        {
            "任务": job["job_id"],
            "序列名称": job["display_name"],
            "标准序列": job["sequence"],
            "来源": job["source"],
            "状态": job["status"],
            "端点数": len(results.get(job["job_id"], {}).get("endpoints", [])),
        }
        for job in batch_state.get("jobs", [])
    ]
    return pd.DataFrame(rows, columns=HISTORY_TASK_COLUMNS)


def history_summary_html(batch_state: dict[str, Any] | None) -> str:
    if not batch_state:
        return "<div class='empty-result'>选择一个历史批次后，这里会显示其序列和执行状态。</div>"
    return (
        "<div class='history-selected'>"
        f"<b>已选批次：</b>{html.escape(str(batch_state.get('batch_id', '')))}"
        f"<span>提交：{html.escape(_display_time(batch_state.get('created_at')))}</span>"
        f"<span>完成：{html.escape(_display_time(batch_state.get('completed_at')))}</span>"
        "</div>"
    )


def _completed_choices(batch_state: dict[str, Any]) -> list[tuple[str, str]]:
    choices = []
    for job in batch_state.get("jobs", []):
        if job["job_id"] in batch_state.get("results", {}):
            label = f"{job['display_name']} · {job['sequence'][:24]}{'…' if len(job['sequence']) > 24 else ''}"
            choices.append((label, job["job_id"]))
    return choices


def _job_by_id(batch_state: dict[str, Any], job_id: str | None) -> dict[str, Any] | None:
    return next((job for job in batch_state.get("jobs", []) if job["job_id"] == job_id), None)


def _empty_detail() -> tuple[Any, Any, Any, Any, Any]:
    return (
        "<div class='empty-result'>首个序列完成后，这里将显示 18 个端点。</div>",
        "",
        pd.DataFrame(columns=TABLE_COLUMNS),
        None,
        None,
    )


def selected_detail(batch_state: dict[str, Any], job_id: str | None) -> tuple[Any, Any, Any, Any, Any]:
    if not batch_state or not job_id:
        return _empty_detail()
    result = batch_state.get("results", {}).get(job_id)
    job = _job_by_id(batch_state, job_id)
    if result is None or job is None:
        return _empty_detail()
    return (
        summary_html(result, job["display_name"]),
        endpoint_cards(result),
        prediction_table(result),
        score_figure(result),
        result,
    )


def _view_outputs(
    batch_state: dict[str, Any],
    selected_job_id: str | None,
    *,
    export_paths: tuple[str, str, str] | None = None,
) -> tuple[Any, ...]:
    choices = _completed_choices(batch_state)
    available_ids = {value for _, value in choices}
    if selected_job_id not in available_ids:
        selected_job_id = choices[0][1] if choices else None
    detail = selected_detail(batch_state, selected_job_id)
    aggregate = batch_prediction_table(batch_state)
    files: tuple[Any, Any, Any] = export_paths or (None, None, None)
    return (
        batch_state,
        gr.update(selected="prediction"),
        batch_summary_html(batch_state),
        task_progress_html(batch_state),
        task_table(batch_state),
        gr.update(choices=choices, value=selected_job_id, interactive=bool(choices)),
        *detail,
        aggregate,
        *files,
    )


def build_demo(
    *,
    predictor: Any,
    export_dir: Path,
    history_dir: Path | None = None,
) -> gr.Blocks:
    """Build a four-page Chinese-first website around one loaded predictor."""

    runtime = predictor.runtime_info()
    manifest = predictor.manifest
    history_store = HistoryStore(history_dir or export_dir.parent / "v31_history")

    def _history_outputs(
        owner_token: str,
        selected_batch_id: str | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        metadata = history_store.list(owner_token)
        choices = history_choices(metadata)
        identifiers = {value for _, value in choices}
        if selected_batch_id not in identifiers:
            selected_batch_id = choices[0][1] if choices else None
        selected_state = (
            history_store.load(owner_token, selected_batch_id) if selected_batch_id else None
        )
        return (
            history_table(metadata),
            gr.update(
                choices=choices,
                value=selected_batch_id,
                interactive=bool(choices),
            ),
            history_summary_html(selected_state),
            history_task_table(selected_state or {}),
        )

    def _event_outputs(
        batch_state: dict[str, Any],
        selected_job_id: str | None,
        owner_token: str,
        *,
        export_paths: tuple[str, str, str] | None = None,
        selected_history_id: str | None = None,
    ) -> tuple[Any, ...]:
        return (
            *_view_outputs(batch_state, selected_job_id, export_paths=export_paths),
            owner_token,
            *_history_outputs(owner_token, selected_history_id),
        )

    def run_batch(
        text_input: str,
        uploaded_file: str | None,
        owner_token: str | None,
    ) -> Iterator[tuple[Any, ...]]:
        owner_token = ensure_browser_owner(owner_token)
        try:
            accepted, rejected = parse_batch_inputs(text_input, uploaded_file)
        except BatchInputError as error:
            raise gr.Error(f"无法创建批次：{error}") from error
        batch_state: dict[str, Any] = {
            "schema_version": "peptide_omnipanel_v31_web_batch_state_v1",
            "batch_id": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "jobs": [
                {
                    **row,
                    "status": "排队中",
                    "progress": "0 / 18",
                    "elapsed_seconds": None,
                    "message": "已通过序列格式检查",
                }
                for row in dataclass_rows(accepted)
            ],
            "rejected_inputs": dataclass_rows(rejected),
            "results": {},
        }
        selected_job_id: str | None = None
        yield _event_outputs(batch_state, selected_job_id, owner_token)
        sequence_cache: dict[str, dict[str, Any]] = {}
        for job in batch_state["jobs"]:
            job["status"] = "预测中"
            job["progress"] = "正在提取 ESM2 特征"
            job["message"] = "正在计算序列表征与 18 个端点"
            yield _event_outputs(batch_state, selected_job_id, owner_token)
            started = time.perf_counter()
            try:
                if job["sequence"] in sequence_cache:
                    result = copy.deepcopy(sequence_cache[job["sequence"]])
                    result["batch_cache_hit"] = True
                else:
                    result = predictor.predict(job["sequence"])
                    result["batch_cache_hit"] = False
                    sequence_cache[job["sequence"]] = copy.deepcopy(result)
                result["web_elapsed_seconds"] = time.perf_counter() - started
                job["elapsed_seconds"] = result["web_elapsed_seconds"]
                job["status"] = "已完成"
                job["progress"] = "18 / 18"
                job["message"] = "使用批内相同序列缓存" if result["batch_cache_hit"] else "18 个本地预测头全部完成"
                batch_state["results"][job["job_id"]] = result
                if selected_job_id is None:
                    selected_job_id = job["job_id"]
            except (V31Peptide18Error, RuntimeError, ValueError) as error:
                job["elapsed_seconds"] = time.perf_counter() - started
                job["status"] = "失败"
                job["progress"] = "0 / 18"
                job["message"] = str(error)
            yield _event_outputs(batch_state, selected_job_id, owner_token)
        batch_state["completed_at"] = datetime.now(timezone.utc).isoformat()
        aggregate = batch_prediction_table(batch_state)
        exports = write_batch_exports(batch_state, aggregate, export_dir)
        try:
            history_store.save(owner_token, batch_state)
        except (HistoryStoreError, OSError, TypeError, ValueError) as error:
            batch_state["history_save_error"] = str(error)
        yield _event_outputs(
            batch_state,
            selected_job_id,
            owner_token,
            export_paths=exports,
            selected_history_id=batch_state["batch_id"],
        )

    def show_selected(batch_state: dict[str, Any], job_id: str) -> tuple[Any, ...]:
        return selected_detail(batch_state, job_id)

    def initialize_history(owner_token: str | None) -> tuple[Any, ...]:
        owner_token = ensure_browser_owner(owner_token)
        return (owner_token, *_history_outputs(owner_token))

    def show_history_batch(
        owner_token: str | None, batch_id: str | None
    ) -> tuple[Any, Any]:
        owner_token = ensure_browser_owner(owner_token)
        if not batch_id:
            return history_summary_html(None), history_task_table({})
        try:
            state = history_store.load(owner_token, batch_id)
        except HistoryStoreError as error:
            raise gr.Error(str(error)) from error
        return history_summary_html(state), history_task_table(state)

    def open_history_batch(
        owner_token: str | None, batch_id: str | None
    ) -> tuple[Any, ...]:
        owner_token = ensure_browser_owner(owner_token)
        if not batch_id:
            raise gr.Error("请先选择一个历史批次")
        try:
            state = history_store.load(owner_token, batch_id)
        except HistoryStoreError as error:
            raise gr.Error(str(error)) from error
        choices = _completed_choices(state)
        selected_job_id = choices[0][1] if choices else None
        aggregate = batch_prediction_table(state)
        exports = write_batch_exports(state, aggregate, export_dir)
        return _view_outputs(state, selected_job_id, export_paths=exports)

    runtime_banner = (
        f"<span><b>序列编码器</b> ESM2-650M 残基平均池化</span>"
        f"<span><b>计算设备</b> {html.escape(runtime['device_name'])}</span>"
        f"<span><b>预测头</b> 18 个本地模型</span>"
        f"<span><b>启动加载</b> {runtime['model_load_seconds']:.1f} 秒</span>"
    )
    css = """
    :root { --ink:#102a43; --muted:#526d82; --navy:#071a2d; --teal:#0f9f91; --cyan:#0891b2; --amber:#f59e0b; }
    .gradio-container { max-width:1480px !important; background:radial-gradient(circle at 85% 0%,#dff7ff 0,transparent 24%),linear-gradient(180deg,#f6fbff 0,#eef5f9 100%); color:var(--ink); }
    .page-panel,.result-panel,.task-panel { color-scheme:light; color:var(--ink) !important; }
    .page-panel .prose,.result-panel .prose,.task-panel .prose,.page-panel label,.result-panel label,.task-panel label,.result-panel button { color:var(--ink) !important; }
    .page-panel textarea,.page-panel input { background:#fbfdff !important; color:#17324d !important; border-color:#b9ced9 !important; }
    .page-panel textarea::placeholder,.page-panel input::placeholder { color:#78909f !important; }
    .hero { background:linear-gradient(125deg,#06172a,#0a3650 62%,#0f5e68); border-radius:24px; padding:30px 34px; color:white; box-shadow:0 18px 55px rgba(7,26,45,.18); margin-bottom:18px; position:relative; overflow:hidden; }
    .hero:after { content:''; position:absolute; width:330px; height:330px; border:1px solid rgba(94,234,212,.28); border-radius:50%; right:-80px; top:-180px; box-shadow:0 0 0 42px rgba(56,189,248,.04),0 0 0 84px rgba(56,189,248,.035); }
    .hero-kicker { letter-spacing:.14em; font-size:12px; color:#67e8f9; font-weight:800; }
    .hero h1 { margin:8px 0 6px; font-size:clamp(30px,4vw,52px); line-height:1.05; color:white !important; }
    .hero p { max-width:980px; color:#d4eef8; font-size:16px; margin:0; }
    .hero-badges { display:flex; gap:8px; flex-wrap:wrap; margin-top:18px; }
    .hero-badges span,.runtime-strip span { border:1px solid rgba(255,255,255,.18); background:rgba(255,255,255,.08); padding:7px 11px; border-radius:999px; font-size:12px; }
    .hero-badges span { color:white !important; }
    .runtime-strip { display:flex; flex-wrap:wrap; gap:10px; padding:12px 16px; background:#e5f5f5; border:1px solid #bce7e3; border-radius:14px; margin-bottom:14px; }
    .runtime-strip span { color:#164e63; border-color:#acdcd8; background:#f4ffff; }
    #workflow-pages > .tab-nav { background:white; border:1px solid #d4e4ec; padding:6px; border-radius:14px; margin-bottom:14px; box-shadow:0 8px 24px rgba(31,73,102,.05); }
    #workflow-pages > .tab-nav button { font-size:15px; font-weight:750; padding:12px 22px; color:#294e63 !important; }
    #workflow-pages > .tab-nav button.selected { color:#0f766e !important; background:#e6faf6; border-radius:10px; }
    .page-panel,.result-panel,.task-panel { border:1px solid #d4e4ec !important; border-radius:18px !important; background:rgba(255,255,255,.9) !important; box-shadow:0 10px 30px rgba(31,73,102,.06); }
    .input-guide { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:8px 0 18px; }
    .input-guide div { background:#f0f9fb; border:1px solid #cce7eb; border-radius:12px; padding:12px; color:#315c6b; font-size:12px; }
    .input-guide b { display:block; color:#0f766e; font-size:13px; margin-bottom:4px; }
    .assumption { padding:12px 14px; border-left:4px solid var(--amber); background:#fff9e8; border-radius:10px; color:#7a4a05; font-size:13px; margin:8px 0 14px; }
    #predict-button { background:linear-gradient(100deg,#0f766e,#0891b2); border:none; box-shadow:0 8px 20px rgba(8,145,178,.23); min-height:48px; }
    #predict-button,#predict-button * { color:white !important; }
    .batch-summary { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
    .batch-summary div { background:white; border:1px solid #d6e8ee; border-top:3px solid var(--teal); border-radius:13px; padding:12px 14px; }
    .batch-summary span { display:block; color:var(--muted); font-size:11px; }
    .batch-summary strong { display:block; color:#082f49; font-size:18px; margin-top:4px; overflow-wrap:anywhere; }
    .task-progress { border:1px solid #b9dfe3; background:linear-gradient(135deg,#f2ffff,#e8f7fb); border-radius:15px; padding:15px; min-height:118px; position:relative; }
    .task-progress-top { display:flex; justify-content:space-between; color:#164e63; font-size:13px; font-weight:750; }
    .progress-track { height:8px; background:#d8ebee; border-radius:99px; overflow:hidden; margin:13px 0 9px; }
    .progress-track i { display:block; height:100%; background:linear-gradient(90deg,#14b8a6,#0891b2); transition:width .3s ease; }
    .task-current { color:#456b78; font-size:12px; padding-right:48px; }
    .task-percent { position:absolute; right:14px; bottom:13px; color:#0f766e; font-weight:800; }
    .task-progress.done { border-color:#99dfc8; background:#edfff7; }
    .selected-sequence { padding:10px 13px; background:#edf9fb; border:1px solid #c8e8ec; border-radius:10px; margin-bottom:10px; color:#315c6b; }
    .selected-sequence code { word-break:break-all; color:#0f4c5c; }
    .summary-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:8px 0 14px; }
    .summary-card { background:white; border:1px solid #d6e8ee; border-top:3px solid var(--teal); border-radius:14px; padding:14px 16px; }
    .summary-card span { display:block; color:var(--muted); font-size:12px; }
    .summary-card strong { display:block; font-size:24px; margin-top:5px; color:var(--navy); }
    .result-note { border:1px solid #cde3ea; background:#f6fbfd; color:#526d82; border-radius:10px; padding:9px 12px; margin-bottom:18px; font-size:12px; }
    .empty-result { border:1px dashed #b9d7df; background:#f8fcfd; color:#5d7885; padding:22px; border-radius:12px; text-align:center; }
    .endpoint-section h3 { color:#164e63; font-size:15px; letter-spacing:.03em; margin:21px 2px 10px; }
    .endpoint-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:11px; }
    .endpoint-card { background:white; border:1px solid #d9e8ee; border-radius:14px; padding:14px; box-shadow:0 6px 18px rgba(31,73,102,.05); }
    .endpoint-top { display:flex; justify-content:space-between; align-items:flex-start; gap:8px; color:#294e63; font-weight:700; font-size:13px; }
    .endpoint-value { font-size:27px; line-height:1; font-weight:760; color:#082f49; margin-top:14px; }
    .endpoint-unit { color:#527084; min-height:18px; font-size:11px; margin-top:5px; }
    .endpoint-meta { color:#78909f; border-top:1px solid #edf3f5; font-size:10px; margin-top:10px; padding-top:8px; }
    .history-selected { display:flex; flex-wrap:wrap; gap:10px 22px; padding:13px 15px; background:#edf9fb; border:1px solid #c8e8ec; border-radius:11px; color:#315c6b; margin:8px 0 13px; }
    .history-selected span { color:#627d8a; }
    .about-intro { padding:18px 20px; background:linear-gradient(120deg,#eefafa,#f5fbff); border:1px solid #cfe7ea; border-radius:14px; color:#315c6b; }
    .definition-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:14px 0 18px; }
    .definition-card { border:1px solid #d7e7ed; background:#fff; border-radius:14px; padding:16px; }
    .definition-card b { display:block; color:#0f766e; font-size:16px; margin-bottom:7px; }
    .definition-card p { color:#526d82; margin:0; font-size:13px; line-height:1.65; }
    .evidence-note { border-left:4px solid #0f9f91; background:#f3fbfa; color:#315c6b; padding:13px 15px; border-radius:10px; margin:12px 0; }
    footer { display:none !important; }
    @media(max-width:900px){.input-guide,.batch-summary,.summary-grid{grid-template-columns:repeat(2,1fr)}.endpoint-grid{grid-template-columns:repeat(2,1fr)}.definition-grid{grid-template-columns:1fr}}
    @media(max-width:600px){.hero{padding:24px 20px}.input-guide,.batch-summary,.summary-grid,.endpoint-grid{grid-template-columns:1fr}}
    """
    force_light = """() => {
      document.documentElement.classList.remove('dark');
      document.body.classList.remove('dark');
      document.documentElement.style.colorScheme = 'light';
      localStorage.setItem('theme', 'light');
    }"""
    theme = gr.themes.Base(primary_hue="teal", secondary_hue="cyan", neutral_hue="slate")
    with gr.Blocks(css=css, js=force_light, theme=theme, title="Peptide OmniPanel V31 肽类端点预测") as demo:
        batch_state = gr.State({})
        browser_owner = gr.BrowserState(
            "",
            storage_key="peptide_omnipanel_v31_browser_owner",
            secret=os.environ.get(
                "V31_WEB_BROWSER_STATE_SECRET",
                "peptide-omnipanel-v31-browser-history-20260723",
            ),
        )
        gr.HTML(
            "<header class='hero'><div class='hero-kicker'>序列驱动的肽类 ADMET 科研系统</div>"
            "<h1>Peptide OmniPanel <span style='color:#5eead4'>V31</span></h1>"
            "<p>输入单条或多条肽序列，一次返回 18 个肽类 ADMET、PK、安全性与稳定性端点。"
            "全部预测在本机 RTX 5080 上执行，序列不发送到外部模型 API。</p>"
            "<div class='hero-badges'><span>18 个肽类端点</span><span>ESM2-650M</span>"
            f"<span>批量上限 {MAX_BATCH_SIZE} 条</span><span>本地 GPU</span>"
            "<span>结果说明可查</span></div></header>"
        )
        gr.HTML(f"<div class='runtime-strip'>{runtime_banner}</div>")
        with gr.Tabs(selected="input", elem_id="workflow-pages") as workflow_pages:
            with gr.Tab("① 序列输入", id="input"):
                with gr.Column(elem_classes="page-panel"):
                    gr.Markdown("## 创建肽序列预测批次")
                    gr.HTML(
                        "<div class='input-guide'>"
                        "<div><b>单条序列</b>直接粘贴一条标准氨基酸序列</div>"
                        "<div><b>多条序列</b>每行一条，或使用“名称 + Tab + 序列”</div>"
                        "<div><b>FASTA</b>支持多记录和跨行序列</div>"
                        "<div><b>CSV / TSV</b>识别 sequence、序列、peptide_sequence 等列名</div>"
                        "</div>"
                    )
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=3):
                            text_input = gr.Textbox(
                                label="序列文本",
                                placeholder=(
                                    "单条：GIGKFLHSAKKFGKAFVGEIMNS\n\n"
                                    "多条 FASTA：\n>peptide_1\nGIGKFLHSAKKFGKAFVGEIMNS\n>peptide_2\nRRWWRF"
                                ),
                                lines=14,
                                max_lines=24,
                                info=f"支持单条/多条/FASTA；每批最多 {MAX_BATCH_SIZE} 条有效序列。",
                                autofocus=True,
                            )
                        with gr.Column(scale=2):
                            upload_file = gr.File(
                                label="上传序列文件",
                                file_types=[".fa", ".faa", ".fasta", ".txt", ".csv", ".tsv"],
                                type="filepath",
                                file_count="single",
                                height=190,
                            )
                            gr.Markdown(
                                "**文件合同**  \n"
                                "支持 UTF-8 的 FASTA/FA/FAA/TXT/CSV/TSV；最大 1 MB。"
                            )
                    gr.HTML(
                        "<div class='assumption'><b>表示假设：</b>裸序列统一解释为天然 L-氨基酸、"
                        "未修饰、线性、自由 N/C 端。环化、D-AA、封端、脂化等无法从裸序列识别。</div>"
                    )
                    with gr.Row():
                        submit_button = gr.Button(
                            "提交批次并进入预测面板",
                            variant="primary",
                            elem_id="predict-button",
                            scale=4,
                        )
                        gr.ClearButton(
                            value="清空输入",
                            components=[text_input, upload_file],
                            scale=1,
                        )
                    gr.Examples(
                        examples=[
                            ["GIGKFLHSAKKFGKAFVGEIMNS"],
                            [">peptide_A\nACDEFGHIKLMNPQRSTVWY\n>peptide_B\nRRWWRF"],
                            ["candidate_1\tKWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK\ncandidate_2\tFLPLLAGLAANFLPQIICKISYKC"],
                        ],
                        inputs=[text_input],
                        label="可点击的示例",
                    )
            with gr.Tab("② 预测面板", id="prediction"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7, elem_classes="result-panel"):
                        gr.Markdown("### 批次概览")
                        batch_summary = gr.HTML(
                            "<div class='empty-result'>尚未创建预测批次。</div>"
                        )
                    with gr.Column(scale=4, elem_classes="task-panel"):
                        progress_panel = gr.HTML(
                            "<aside class='task-progress queued'><div class='task-progress-top'><span>任务进展</span><strong>0 / 0</strong></div>"
                            "<div class='progress-track'><i style='width:0%'></i></div><div class='task-current'>等待提交序列</div><div class='task-percent'>0%</div></aside>"
                        )
                        with gr.Row():
                            back_button = gr.Button("返回序列输入")
                            history_nav_button = gr.Button("查看历史任务")
                        explanation_button = gr.Button("查看网站与结果说明")
                with gr.Column(elem_classes="task-panel"):
                    gr.Markdown("### 任务列表")
                    task_frame = gr.Dataframe(
                        headers=TASK_COLUMNS,
                        value=pd.DataFrame(columns=TASK_COLUMNS),
                        interactive=False,
                        wrap=True,
                        max_height=330,
                    )
                with gr.Column(elem_classes="result-panel"):
                    selected_job = gr.Dropdown(
                        label="选择已完成的肽序列",
                        choices=[],
                        interactive=False,
                        info="每完成一条序列，即可查看该序列的 18 个端点。",
                    )
                    selected_summary = gr.HTML(
                        "<div class='empty-result'>首个序列完成后，这里将显示 18 个端点。</div>"
                    )
                    with gr.Tabs():
                        with gr.Tab("端点概览"):
                            cards = gr.HTML()
                        with gr.Tab("当前序列完整结果"):
                            table = gr.Dataframe(
                                headers=TABLE_COLUMNS,
                                value=pd.DataFrame(columns=TABLE_COLUMNS),
                                interactive=False,
                                wrap=True,
                                max_height=680,
                            )
                        with gr.Tab("分类/排序图"):
                            figure = gr.Plot()
                        with gr.Tab("批量汇总表"):
                            aggregate_table = gr.Dataframe(
                                headers=BATCH_COLUMNS,
                                value=pd.DataFrame(columns=BATCH_COLUMNS),
                                interactive=False,
                                wrap=True,
                                max_height=720,
                            )
                        with gr.Tab("当前序列原始 JSON"):
                            raw_json = gr.JSON(label="可审计预测对象")
                        with gr.Tab("下载批量结果"):
                            gr.Markdown(
                                "JSON 保留全部模型、SHA、OOF 与警告；TSV 适合表格分析；ZIP 同时包含批量汇总和每条序列的独立 JSON。"
                            )
                            with gr.Row():
                                json_file = gr.File(label="下载批量 JSON")
                                tsv_file = gr.File(label="下载批量 TSV")
                                zip_file = gr.File(label="下载完整 ZIP")

            with gr.Tab("③ 历史任务", id="history"):
                with gr.Column(elem_classes="page-panel"):
                    gr.Markdown(
                        "## 历史任务\n"
                        "这里保留当前浏览器提交的最近 50 个完整批次。"
                        "历史通过浏览器随机标识隔离，不会在页面上全局展示其他访问者的序列。"
                    )
                    with gr.Row():
                        history_batch = gr.Dropdown(
                            label="选择历史批次",
                            choices=[],
                            interactive=False,
                            scale=5,
                        )
                        refresh_history_button = gr.Button("刷新历史", scale=1)
                        open_history_button = gr.Button(
                            "打开预测结果", variant="primary", scale=2
                        )
                    history_selected_summary = gr.HTML(history_summary_html(None))
                    history_frame = gr.Dataframe(
                        headers=HISTORY_COLUMNS,
                        value=pd.DataFrame(columns=HISTORY_COLUMNS),
                        interactive=False,
                        wrap=True,
                        max_height=430,
                        label="历史批次汇总",
                    )
                    history_task_frame = gr.Dataframe(
                        headers=HISTORY_TASK_COLUMNS,
                        value=pd.DataFrame(columns=HISTORY_TASK_COLUMNS),
                        interactive=False,
                        wrap=True,
                        max_height=520,
                        label="所选批次的提交序列",
                    )

            with gr.Tab("④ 网站与结果说明", id="about"):
                with gr.Column(elem_classes="page-panel"):
                    gr.HTML(
                        "<div class='about-intro'><b>为什么把这些说明单独放在一页？</b><br>"
                        "主预测面板聚焦序列与数值，不再在每个端点上重复显示警示。"
                        "证据级别、使用范围和验证阶段仍完整保留于本页、原始 JSON 与下载文件，便于审计。</div>"
                        "<div class='definition-grid'>"
                        "<div class='definition-card'><b>低置信度</b><p>这是“结果可靠性”标记。常由训练样本少、弱标签、实验条件异质或 OOF 表现不稳定引起，不等于使用权限。</p></div>"
                        "<div class='definition-card'><b>仅供科研</b><p>这是“允许用途”标记。它限定结果用于方法研究、候选排序和实验假设，不直接代表模型分数高低。</p></div>"
                        "<div class='definition-card'><b>未独立验证</b><p>这是“评估阶段”标记。当前有分组 OOF 开发评估，但尚无真正独立的外部来源或封存测试，所以泛化能力尚未被外部证明。</p></div>"
                        "</div>"
                        "<div class='evidence-note'><b>三者并不重复：</b>"
                        "低置信度讲可靠性，仅供科研讲用途，未独立验证讲评估阶段。"
                        "主面板减少重复显示，不会修改或删除原始证据字段。</div>"
                    )
                    gr.Markdown(
                        "### 18 个端点的证据与训练信息\n"
                        "下表是开发阶段的证据记录；OOF 指标不是独立外部测试成绩。"
                    )
                    gr.Dataframe(
                        headers=EVIDENCE_COLUMNS,
                        value=endpoint_evidence_table(manifest),
                        interactive=False,
                        wrap=True,
                        max_height=760,
                    )
                    gr.Markdown(
                        """
### 结果解读边界

- 分类输出是当前任务下的模型分数，不是临床风险概率。
- 正样本-未标注任务的 0–1 输出主要用于同一端点内候选肽排序，不可跨端点比较。
- 裸序列默认为天然 L-氨基酸、线性、未修饰且 N/C 端自由；环化、D-AA、二硫键拓扑、封端、脂化和 PEG 化无法从裸序列唯一确定。
- 如果用于真实实验设计，应结合端点条件、修饰信息、不确定性和正交实验复核。
"""
                    )

        prediction_outputs = [
            batch_state,
            workflow_pages,
            batch_summary,
            progress_panel,
            task_frame,
            selected_job,
            selected_summary,
            cards,
            table,
            figure,
            raw_json,
            aggregate_table,
            json_file,
            tsv_file,
            zip_file,
        ]
        history_outputs = [
            browser_owner,
            history_frame,
            history_batch,
            history_selected_summary,
            history_task_frame,
        ]
        submit_button.click(
            fn=run_batch,
            inputs=[text_input, upload_file, browser_owner],
            outputs=[*prediction_outputs, *history_outputs],
            api_name=False,
            show_api=False,
            show_progress="hidden",
            concurrency_limit=1,
            concurrency_id="v31_gpu",
        )
        selected_job.change(
            fn=show_selected,
            inputs=[batch_state, selected_job],
            outputs=[selected_summary, cards, table, figure, raw_json],
            api_name=False,
            show_api=False,
            show_progress="hidden",
            concurrency_limit=1,
        )
        back_button.click(
            fn=lambda: gr.update(selected="input"),
            inputs=None,
            outputs=workflow_pages,
            api_name=False,
            show_api=False,
        )
        history_nav_button.click(
            fn=lambda: gr.update(selected="history"),
            inputs=None,
            outputs=workflow_pages,
            api_name=False,
            show_api=False,
        )
        explanation_button.click(
            fn=lambda: gr.update(selected="about"),
            inputs=None,
            outputs=workflow_pages,
            api_name=False,
            show_api=False,
        )
        history_batch.change(
            fn=show_history_batch,
            inputs=[browser_owner, history_batch],
            outputs=[history_selected_summary, history_task_frame],
            api_name=False,
            show_api=False,
            show_progress="hidden",
        )
        refresh_history_button.click(
            fn=initialize_history,
            inputs=browser_owner,
            outputs=history_outputs,
            api_name=False,
            show_api=False,
            show_progress="hidden",
        )
        open_history_button.click(
            fn=open_history_batch,
            inputs=[browser_owner, history_batch],
            outputs=prediction_outputs,
            api_name=False,
            show_api=False,
            show_progress="hidden",
        )
        demo.load(
            fn=initialize_history,
            inputs=browser_owner,
            outputs=history_outputs,
            api_name=False,
            show_api=False,
            show_progress="hidden",
        )
    return demo.queue(default_concurrency_limit=1, max_size=8, api_open=False)
