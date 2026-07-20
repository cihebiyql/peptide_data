"""Auditable classical and masked multi-task models."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .metrics import (
    choose_binary_threshold,
    classification_metrics,
    regression_metrics,
)


@dataclass(frozen=True)
class CandidateScore:
    model_name: str
    task_kind: str
    selection_metric: str
    selection_value: float | None
    threshold: float | None
    metrics: Mapping[str, float | None]
    status: str = "success"
    failure_reason: str | None = None


@dataclass(frozen=True)
class ModelBundleMetadata:
    task_id: str
    endpoint_family: str
    task_kind: str
    model_name: str
    feature_schema: str
    seed: int
    development_rows: int
    calibration_rows: int
    selected_on: str = "calibration"


def _require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    except ImportError as exc:  # pragma: no cover - exercised on the training host.
        raise RuntimeError("scikit-learn is required for baseline training") from exc
    return {
        "DummyClassifier": DummyClassifier,
        "DummyRegressor": DummyRegressor,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "ExtraTreesRegressor": ExtraTreesRegressor,
        "LogisticRegression": LogisticRegression,
        "Ridge": Ridge,
        "KNeighborsClassifier": KNeighborsClassifier,
        "KNeighborsRegressor": KNeighborsRegressor,
    }


def candidate_estimators(task_kind: str, *, seed: int, train_rows: int) -> dict[str, Any]:
    sk = _require_sklearn()
    neighbors = max(1, min(7, int(train_rows) - 1))
    if task_kind == "regression":
        return {
            "constant_median": sk["DummyRegressor"](strategy="median"),
            "analog_knn": sk["KNeighborsRegressor"](n_neighbors=neighbors, weights="distance"),
            "ridge": sk["Ridge"](alpha=10.0),
            "extra_trees": sk["ExtraTreesRegressor"](
                n_estimators=256,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=-1,
                random_state=seed,
            ),
        }
    if task_kind == "binary_classification":
        return {
            "constant_prior": sk["DummyClassifier"](strategy="prior"),
            "analog_knn": sk["KNeighborsClassifier"](
                n_neighbors=neighbors,
                weights="distance",
            ),
            "logistic": sk["LogisticRegression"](
                C=0.1,
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            ),
            "extra_trees": sk["ExtraTreesClassifier"](
                n_estimators=256,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ),
        }
    raise ValueError(f"Unsupported task kind: {task_kind}")


def predict_estimator(estimator: Any, matrix: Any, task_kind: str) -> Any:
    if task_kind == "regression":
        return estimator.predict(matrix)
    probabilities = estimator.predict_proba(matrix)
    classes = list(estimator.classes_)
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    return probabilities[:, 0] * 0.0


def train_select_baseline(
    *,
    task_id: str,
    endpoint_family: str,
    task_kind: str,
    feature_schema: str,
    development_x: Any,
    development_y: Any,
    calibration_x: Any,
    calibration_y: Any,
    seed: int,
) -> tuple[Any, ModelBundleMetadata, list[CandidateScore]]:
    estimators = candidate_estimators(task_kind, seed=seed, train_rows=len(development_y))
    scores: list[CandidateScore] = []
    best_key: tuple[float, float, int] | None = None
    best_name: str | None = None
    for candidate_index, (name, estimator) in enumerate(estimators.items()):
        try:
            estimator.fit(development_x, development_y)
            if len(calibration_y) == 0 and len(calibration_x) == 0:
                prediction: Any = []
            else:
                prediction = predict_estimator(estimator, calibration_x, task_kind)
            if task_kind == "regression":
                metric_values = regression_metrics(calibration_y, prediction)
                raw_value = metric_values["mae"]
                selection_value = None if raw_value is None else float(raw_value)
                threshold = None
                if selection_value is None:
                    selection_metric = "fallback_no_finite_calibration_metric"
                    key = (2.0, 0.0, candidate_index)
                else:
                    selection_metric = "mae"
                    key = (0.0, selection_value, candidate_index)
            else:
                threshold = choose_binary_threshold(calibration_y, prediction)
                metric_values = classification_metrics(
                    calibration_y,
                    prediction,
                    threshold=threshold,
                )
                pr_auc = metric_values["pr_auc"]
                brier = metric_values["brier"]
                if pr_auc is not None:
                    selection_metric = "pr_auc"
                    selection_value = float(pr_auc)
                    key = (0.0, -selection_value, candidate_index)
                elif brier is not None:
                    selection_metric = "brier_fallback"
                    selection_value = float(brier)
                    key = (1.0, selection_value, candidate_index)
                else:
                    selection_metric = "fallback_no_finite_calibration_metric"
                    selection_value = None
                    key = (2.0, 0.0, candidate_index)
        except Exception as exc:
            scores.append(
                CandidateScore(
                    model_name=name,
                    task_kind=task_kind,
                    selection_metric="mae" if task_kind == "regression" else "pr_auc",
                    selection_value=None,
                    threshold=None,
                    metrics={},
                    status="failed",
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        scores.append(
            CandidateScore(
                model_name=name,
                task_kind=task_kind,
                selection_metric=selection_metric,
                selection_value=selection_value,
                threshold=threshold,
                metrics=metric_values,
            )
        )
        if best_key is None or key < best_key:
            best_key = key
            best_name = name
    if best_name is None:
        failures = "; ".join(
            f"{score.model_name}: {score.failure_reason}"
            for score in scores
            if score.status == "failed"
        )
        suffix = f"; failures: {failures}" if failures else ""
        raise RuntimeError(f"No model candidate could be selected for {task_id}{suffix}")

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required for final model fitting") from exc
    combined_x = np.concatenate([development_x, calibration_x], axis=0)
    combined_y = np.concatenate([development_y, calibration_y], axis=0)
    final_model = candidate_estimators(task_kind, seed=seed, train_rows=len(combined_y))[best_name]
    try:
        final_model.fit(combined_x, combined_y)
    except Exception as exc:
        raise RuntimeError(
            f"Selected candidate {best_name} failed final fitting for {task_id}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    metadata = ModelBundleMetadata(
        task_id=task_id,
        endpoint_family=endpoint_family,
        task_kind=task_kind,
        model_name=best_name,
        feature_schema=feature_schema,
        seed=seed,
        development_rows=len(development_y),
        calibration_rows=len(calibration_y),
    )
    return final_model, metadata, scores


def save_model_bundle(
    output_path: Path,
    *,
    model: Any,
    metadata: ModelBundleMetadata,
    scores: Iterable[CandidateScore],
) -> None:
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("joblib is required to serialize models") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "metadata": asdict(metadata),
            "candidate_scores": [asdict(score) for score in scores],
        },
        output_path,
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(asdict(metadata), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_model_bundle(
    input_path: Path,
) -> tuple[Any, ModelBundleMetadata, list[CandidateScore]]:
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("joblib is required to deserialize models") from exc
    payload = joblib.load(input_path)
    if not isinstance(payload, Mapping):
        raise ValueError("Model bundle must contain a mapping payload")
    missing = {"model", "metadata", "candidate_scores"} - set(payload)
    if missing:
        raise ValueError(f"Model bundle is missing required fields: {sorted(missing)}")
    try:
        metadata_payload = payload["metadata"]
        metadata = (
            metadata_payload
            if isinstance(metadata_payload, ModelBundleMetadata)
            else ModelBundleMetadata(**dict(metadata_payload))
        )
        candidate_scores = [
            score if isinstance(score, CandidateScore) else CandidateScore(**dict(score))
            for score in payload["candidate_scores"]
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid model bundle metadata: {exc}") from exc
    return payload["model"], metadata, candidate_scores


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on the training host.
        raise RuntimeError("PyTorch is required for multi-task training") from exc
    return torch


def train_masked_multitask_mlp(
    *,
    features: Any,
    targets: Any,
    task_indices: Any,
    task_count: int,
    task_kind: str,
    sample_weights: Any | None = None,
    hidden_dim: int = 192,
    epochs: int = 120,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 20260715,
    device: str | None = None,
    allow_fallback: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Train a shared trunk with task-specific linear heads on sparse long-form rows."""

    try:
        rows = len(targets)
    except TypeError:
        rows = 0
    if task_kind not in {"regression", "binary_classification"}:
        raise ValueError(f"Unsupported task kind: {task_kind}")
    if task_count < 1:
        raise ValueError("task_count must be positive")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    try:
        torch = _require_torch()
    except RuntimeError as exc:
        if not allow_fallback:
            raise
        return None, {
            "status": "skipped",
            "backend": "torch",
            "reason": str(exc),
            "device": None,
            "epochs": 0,
            "task_count": task_count,
            "rows": rows,
        }
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(targets, dtype=torch.float32)
    tasks = torch.as_tensor(task_indices, dtype=torch.long)
    weights = (
        torch.ones_like(y)
        if sample_weights is None
        else torch.as_tensor(sample_weights, dtype=torch.float32)
    )
    if x.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if y.ndim != 1 or tasks.ndim != 1 or weights.ndim != 1:
        raise ValueError("targets, task_indices, and sample_weights must be one-dimensional")
    if not len(y):
        raise ValueError("multi-task training requires at least one row")
    if len(x) != len(y) or len(tasks) != len(y) or len(weights) != len(y):
        raise ValueError("features, targets, task_indices, and sample_weights must have equal rows")
    if bool(torch.any(tasks < 0)) or bool(torch.any(tasks >= task_count)):
        raise ValueError("task_indices must be in [0, task_count)")

    class SparseHeadMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(x.shape[1], hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
            )
            self.head_weight = torch.nn.Parameter(torch.empty(task_count, hidden_dim))
            self.head_bias = torch.nn.Parameter(torch.zeros(task_count))
            torch.nn.init.xavier_uniform_(self.head_weight)

        def forward(self, batch_x: Any, batch_tasks: Any) -> Any:
            hidden = self.trunk(batch_x)
            return (hidden * self.head_weight[batch_tasks]).sum(dim=1) + self.head_bias[
                batch_tasks
            ]

    model = SparseHeadMLP().to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    dataset = torch.utils.data.TensorDataset(x, y, tasks, weights)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        observed = 0
        for batch_x, batch_y, batch_tasks, batch_weights in loader:
            batch_x = batch_x.to(selected_device)
            batch_y = batch_y.to(selected_device)
            batch_tasks = batch_tasks.to(selected_device)
            batch_weights = batch_weights.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x, batch_tasks)
            if task_kind == "regression":
                losses = torch.nn.functional.smooth_l1_loss(logits, batch_y, reduction="none")
            elif task_kind == "binary_classification":
                losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    batch_y,
                    reduction="none",
                )
            else:
                raise ValueError(f"Unsupported task kind: {task_kind}")
            loss = (losses * batch_weights).sum() / batch_weights.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(batch_y)
            observed += len(batch_y)
        history.append(epoch_loss / max(observed, 1))
    report = {
        "status": "trained",
        "backend": "torch",
        "device": selected_device,
        "epochs": epochs,
        "batch_size": min(batch_size, len(dataset)),
        "final_loss": history[-1],
        "minimum_loss": min(history),
        "task_count": task_count,
        "rows": len(dataset),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(selected_device)
            if selected_device.startswith("cuda")
            else None
        ),
    }
    return model, report


def train_task_balanced_multitask_mlp(
    *,
    features: Any,
    targets: Any,
    task_indices: Any,
    task_count: int,
    task_kind: str,
    validation_features: Any | None = None,
    validation_targets: Any | None = None,
    validation_task_indices: Any | None = None,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 12,
    seed: int = 20260716,
    device: str | None = None,
    allow_fallback: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Train a task-balanced residual MLP with fold-local target scaling."""

    try:
        torch = _require_torch()
    except RuntimeError as exc:
        if not allow_fallback:
            raise
        return None, {
            "status": "skipped",
            "backend": "torch",
            "reason": str(exc),
            "rows": len(targets),
            "task_count": task_count,
        }
    if task_kind not in {"regression", "binary_classification"}:
        raise ValueError(f"Unsupported task kind: {task_kind}")
    if task_count < 2:
        raise ValueError("task-balanced training requires at least two tasks")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(targets, dtype=torch.float32)
    tasks = torch.as_tensor(task_indices, dtype=torch.long)
    if x.ndim != 2 or y.ndim != 1 or tasks.ndim != 1:
        raise ValueError("features must be 2D and targets/tasks must be 1D")
    if len(x) != len(y) or len(y) != len(tasks) or not len(y):
        raise ValueError("features, targets, and tasks must have equal nonzero rows")
    if bool(torch.any(tasks < 0)) or bool(torch.any(tasks >= task_count)):
        raise ValueError("task_indices must be in [0, task_count)")
    task_counts = torch.bincount(tasks, minlength=task_count)
    if bool(torch.any(task_counts == 0)):
        raise ValueError("every task must have at least one training row")

    target_means = torch.zeros(task_count, dtype=torch.float32)
    target_scales = torch.ones(task_count, dtype=torch.float32)
    if task_kind == "regression":
        for task_index in range(task_count):
            values = y[tasks == task_index]
            target_means[task_index] = values.mean()
            scale = values.std(unbiased=False)
            target_scales[task_index] = scale if float(scale) > 1e-8 else 1.0
        model_targets = (y - target_means[tasks]) / target_scales[tasks]
    else:
        model_targets = y

    class_weights = torch.ones_like(model_targets)
    if task_kind == "binary_classification":
        for task_index in range(task_count):
            mask = tasks == task_index
            positives = torch.sum(y[mask] == 1.0)
            negatives = torch.sum(y[mask] == 0.0)
            if int(positives) and int(negatives):
                class_weights[mask & (y == 1.0)] = negatives / positives

    class TaskBalancedResidualMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layer = torch.nn.Sequential(
                torch.nn.Linear(x.shape[1], hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.SiLU(),
            )
            self.residual = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            self.output_norm = torch.nn.LayerNorm(hidden_dim)
            self.head_weight = torch.nn.Parameter(torch.empty(task_count, hidden_dim))
            self.head_bias = torch.nn.Parameter(torch.zeros(task_count))
            torch.nn.init.xavier_uniform_(self.head_weight)

        def forward(self, batch_x: Any, batch_tasks: Any) -> Any:
            hidden = self.input_layer(batch_x)
            hidden = self.output_norm(hidden + self.residual(hidden))
            return (hidden * self.head_weight[batch_tasks]).sum(dim=1) + self.head_bias[
                batch_tasks
            ]

    model = TaskBalancedResidualMLP().to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    sample_probabilities = 1.0 / task_counts[tasks].float()
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_probabilities,
        num_samples=len(y),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    dataset = torch.utils.data.TensorDataset(
        x,
        model_targets,
        tasks,
        class_weights,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        sampler=sampler,
    )

    validation = None
    if validation_features is not None:
        if validation_targets is None or validation_task_indices is None:
            raise ValueError("validation features require validation targets and tasks")
        validation_x = torch.as_tensor(validation_features, dtype=torch.float32)
        validation_y = torch.as_tensor(validation_targets, dtype=torch.float32)
        validation_tasks = torch.as_tensor(validation_task_indices, dtype=torch.long)
        if task_kind == "regression":
            validation_model_y = (
                validation_y - target_means[validation_tasks]
            ) / target_scales[validation_tasks]
        else:
            validation_model_y = validation_y
        validation = (validation_x, validation_model_y, validation_tasks)

    def macro_loss(validation_data: tuple[Any, Any, Any]) -> float:
        validation_x, validation_y, validation_tasks = validation_data
        model.eval()
        with torch.no_grad():
            logits = model(
                validation_x.to(selected_device),
                validation_tasks.to(selected_device),
            ).cpu()
        losses: list[float] = []
        for task_index in range(task_count):
            mask = validation_tasks == task_index
            if not bool(torch.any(mask)):
                continue
            if task_kind == "regression":
                loss = torch.nn.functional.smooth_l1_loss(
                    logits[mask],
                    validation_y[mask],
                )
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[mask],
                    validation_y[mask],
                )
            losses.append(float(loss))
        if not losses:
            raise ValueError("validation data has no represented tasks")
        return sum(losses) / len(losses)

    best_state = None
    best_loss = float("inf")
    best_epoch = epochs
    stale_epochs = 0
    history: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        observed_loss = 0.0
        observed_rows = 0
        for batch_x, batch_y, batch_tasks, batch_weights in loader:
            batch_x = batch_x.to(selected_device)
            batch_y = batch_y.to(selected_device)
            batch_tasks = batch_tasks.to(selected_device)
            batch_weights = batch_weights.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x, batch_tasks)
            if task_kind == "regression":
                losses = torch.nn.functional.smooth_l1_loss(
                    logits,
                    batch_y,
                    reduction="none",
                )
            else:
                losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    batch_y,
                    reduction="none",
                )
            loss = (losses * batch_weights).sum() / batch_weights.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            observed_loss += float(loss.detach().cpu()) * len(batch_y)
            observed_rows += len(batch_y)
        history.append(observed_loss / max(observed_rows, 1))

        if validation is None:
            continue
        current_loss = macro_loss(validation)
        if current_loss < best_loss - 1e-6:
            best_loss = current_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    report = {
        "status": "trained",
        "backend": "torch",
        "device": selected_device,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_macro_loss": None if validation is None else best_loss,
        "final_training_loss": history[-1],
        "minimum_training_loss": min(history),
        "task_count": task_count,
        "task_counts": [int(value) for value in task_counts.tolist()],
        "task_sampling": "inverse_frequency_weighted_random_sampler",
        "expected_task_sampling_probabilities": [1.0 / task_count] * task_count,
        "target_means": [float(value) for value in target_means.tolist()],
        "target_scales": [float(value) for value in target_scales.tolist()],
        "target_standardization": task_kind == "regression",
        "class_weighting": task_kind == "binary_classification",
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": min(batch_size, len(dataset)),
        "rows": len(dataset),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    return model, report


def predict_task_balanced_multitask_mlp(
    model: Any,
    *,
    features: Any,
    task_indices: Any,
    task_kind: str,
    train_report: Mapping[str, Any],
) -> Any:
    """Predict and invert the fold-local task target transform."""

    torch = _require_torch()
    device = next(model.parameters()).device
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    tasks = torch.as_tensor(task_indices, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        raw = model(x, tasks)
    if task_kind == "binary_classification":
        return torch.sigmoid(raw).cpu().numpy()
    if task_kind != "regression":
        raise ValueError(f"Unsupported task kind: {task_kind}")
    means = torch.as_tensor(train_report["target_means"], dtype=torch.float32, device=device)
    scales = torch.as_tensor(train_report["target_scales"], dtype=torch.float32, device=device)
    return (raw * scales[tasks] + means[tasks]).cpu().numpy()
