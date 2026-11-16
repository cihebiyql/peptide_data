"""Auditable, globally budgeted candidate search with optional Optuna TPE.

The caller owns grouped cross-validation and candidate artifact caching. This
module never sees held-out targets and never replaces a failed fit with a score.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import socket
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = "peptide_budgeted_search_v1"
MODEL_NAMES = {
    "linear",
    "lightgbm",
    "elasticnet",
    "random_forest",
    "extra_trees",
    "histgb",
    "xgboost",
    "svm",
}


class SearchBackendUnavailable(RuntimeError):
    """The explicitly requested optional search backend is not installed."""


class CandidateEvaluationError(RuntimeError):
    """An attempted candidate failed; Optuna must record FAIL, not COMPLETE."""


def _json(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):
        return _plain(value.item())
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(_plain(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _save_trial(path, evaluation):
    evaluation = _plain(evaluation)
    _write(path, {"evaluation": evaluation, "record_sha256": _digest(evaluation)})


def _read_trial(path):
    wrapped = json.loads(Path(path).read_text(encoding="utf-8"))
    value = wrapped["evaluation"]
    if _digest(value) != wrapped.get("record_sha256"):
        raise ValueError("search audit record hash mismatch: " + str(path))
    return value


@contextmanager
def _exclusive(directory):
    lock = directory / "search.lock"
    if lock.exists():
        prior = json.loads(lock.read_text(encoding="utf-8"))
        alive = True
        if prior.get("hostname") == socket.gethostname():
            try:
                os.kill(int(prior["pid"]), 0)
            except ProcessLookupError:
                alive = False
        if alive:
            raise RuntimeError("Search namespace is already locked: " + str(lock))
        lock.rename(directory / f"search.lock.stale.{time.time_ns()}")
    token = str(time.time_ns())
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"hostname": socket.gethostname(), "pid": os.getpid(), "token": token}, handle)
    try:
        yield
    finally:
        if lock.exists() and json.loads(lock.read_text(encoding="utf-8")).get("token") == token:
            lock.unlink()


def _suggest_params(trial, model, task_kind):
    """Stable named distributions; the route does not change a parameter's domain."""
    prefix = model + "."

    def categorical(name, values):
        return trial.suggest_categorical(prefix + name, values)

    def log_float(name, low, high):
        return trial.suggest_float(prefix + name, low, high, log=True)

    if model == "linear":
        return {"regularization": log_float("regularization", 1e-3, 1e3)}
    if model == "elasticnet":
        return {
            "regularization": log_float("regularization", 1e-4, 10),
            "l1_ratio": categorical("l1_ratio", [0.1, 0.5, 0.9]),
        }
    if model == "lightgbm":
        return {
            "n_estimators": categorical("n_estimators", [100, 200, 400]),
            "learning_rate": categorical("learning_rate", [0.02, 0.05, 0.1]),
            "num_leaves": categorical("num_leaves", [7, 15, 31]),
            "min_child_samples": categorical("min_child_samples", [5, 10, 20, 40]),
            "reg_lambda": categorical("reg_lambda", [0.0, 0.1, 1.0, 10.0]),
            "colsample_bytree": categorical("colsample_bytree", [0.7, 1.0]),
        }
    if model in {"random_forest", "extra_trees"}:
        return {
            "n_estimators": categorical("n_estimators", [100, 200]),
            "max_depth": categorical("max_depth", [6, 12]),
            "min_samples_leaf": categorical("min_samples_leaf", [2, 5, 10]),
            "max_features": categorical("max_features", ["sqrt", 0.1]),
        }
    if model == "histgb":
        return {
            "max_iter": categorical("max_iter", [100, 200]),
            "max_leaf_nodes": categorical("max_leaf_nodes", [15, 31]),
            "learning_rate": categorical("learning_rate", [0.03, 0.1]),
            "l2_regularization": categorical("l2_regularization", [0.0, 0.1, 1.0, 10.0]),
        }
    if model == "xgboost":
        return {
            "n_estimators": categorical("n_estimators", [100, 200]),
            "max_depth": categorical("max_depth", [3, 6]),
            "learning_rate": categorical("learning_rate", [0.03, 0.1]),
            "subsample": categorical("subsample", [0.8, 1.0]),
            "colsample_bytree": categorical("colsample_bytree", [0.7, 1.0]),
            "reg_lambda": categorical("reg_lambda", [0.1, 1.0, 10.0]),
        }
    if model == "svm":
        result = {
            "C": categorical("C", [0.1, 1.0, 10.0]),
            "gamma": categorical("gamma", ["scale", "auto"]),
        }
        if task_kind == "regression":
            result["epsilon"] = categorical("epsilon", [0.01, 0.1, 0.3])
        return result
    raise ValueError("Unsupported search model: " + model)


def _safe_params(model, task_kind):
    """Bounded initial representatives; later TPE trials explore the full space."""
    if model == "linear":
        return {"regularization": 1.0}
    if model == "elasticnet":
        return {"regularization": 0.01, "l1_ratio": 0.5}
    if model == "lightgbm":
        return {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 20,
            "reg_lambda": 1.0,
            "colsample_bytree": 0.7,
        }
    if model in {"random_forest", "extra_trees"}:
        return {"n_estimators": 100, "max_depth": 6, "min_samples_leaf": 2, "max_features": "sqrt"}
    if model == "histgb":
        return {
            "max_iter": 100,
            "max_leaf_nodes": 15,
            "learning_rate": 0.1,
            "l2_regularization": 1.0,
        }
    if model == "xgboost":
        return {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_lambda": 1.0,
        }
    if model == "svm":
        return {
            "C": 1.0,
            "gamma": "scale",
            **({"epsilon": 0.1} if task_kind == "regression" else {}),
        }
    raise ValueError("Unsupported search model: " + model)


class _RandomTrial:
    def __init__(self, rng):
        self.rng = rng

    def suggest_categorical(self, name, values):
        del name
        return self.rng.choice(values)

    def suggest_float(self, name, low, high, log=False):
        del name
        return (
            math.exp(self.rng.uniform(math.log(low), math.log(high)))
            if log
            else self.rng.uniform(low, high)
        )


def _evaluate(evaluate, route, model, params, number, signature, path):
    initial = {
        "route": route,
        "model": model,
        "params": params,
        "status": "running",
        "search_trial": number,
        "search_input_signature": signature,
    }
    _save_trial(path, initial)
    try:
        value = evaluate(route, model, params)
        if not isinstance(value, dict):
            raise ValueError("evaluate must return a candidate dictionary")
        for key, expected in (("route", route), ("model", model), ("params", params)):
            if key in value and _plain(value[key]) != expected:
                raise ValueError("evaluate candidate mismatch: " + key)
        value = {**initial, **_plain(value)}
        if value.get("status") != "completed":
            value.update(
                status="failed",
                score=None,
                reason=value.get("reason") or "candidate_evaluation_not_completed",
            )
        elif value.get("score") is None or not math.isfinite(float(value["score"])):
            value.update(status="failed", score=None, reason="candidate_score_missing_or_nonfinite")
        else:
            value["score"] = float(value["score"])
    except Exception as exc:
        value = {
            **initial,
            "status": "failed",
            "score": None,
            "reason": f"{type(exc).__name__}:{str(exc)[:1600]}",
        }
    _save_trial(path, value)
    return value


def _random_search(pairs, evaluate, config, directory, contract_hash, budget, task_kind):
    rng = random.Random(config.get("seed", 42))
    warm = list(pairs)
    rng.shuffle(warm)
    results = []
    existing = list((directory / "trials").glob("*.json"))
    if len(existing) > budget:
        raise ValueError("budget_error: stored attempts exceed requested global budget")
    for number in range(budget):
        route, model = warm[number] if number < len(warm) else rng.choice(pairs)
        params = (
            _safe_params(model, task_kind)
            if number < len(warm)
            else _suggest_params(_RandomTrial(rng), model, task_kind)
        )
        path = directory / "trials" / f"{number:06d}.json"
        if path.exists():
            value = _read_trial(path)
            if (value["route"], value["model"], value["params"]) != (route, model, params):
                raise ValueError("Stored candidate does not match deterministic search plan")
            if value["status"] == "running":
                value.update(status="failed", score=None, reason="interrupted_candidate_attempt")
                _save_trial(path, value)
        else:
            value = _evaluate(evaluate, route, model, params, number, contract_hash, path)
        results.append(value)
    return results, {}


def _optuna_search(pairs, evaluate, config, directory, contract_hash, budget, task_kind, optuna):
    path = directory / "study.sqlite3"
    study = optuna.create_study(
        storage="sqlite:///" + path.resolve().as_posix(),
        study_name="peptide_" + contract_hash[:24],
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=int(config.get("seed", 42)), n_startup_trials=max(10, len(pairs))
        ),
        pruner=optuna.pruners.NopPruner(),
    )
    identifiers = [f"pair_{i:04d}" for i in range(len(pairs))]
    mapping = dict(zip(identifiers, pairs, strict=True))
    # No other writer owns this namespace while its process lock is held.
    for trial in study.get_trials(deepcopy=False):
        audit = directory / "trials" / f"{trial.number:06d}.json"
        if trial.state == optuna.trial.TrialState.RUNNING:
            value = _read_trial(audit) if audit.exists() else None
            if value and value["status"] == "completed":
                study.tell(trial.number, value["score"])
            else:
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    finished = [t for t in study.get_trials(deepcopy=False) if t.state.is_finished()]
    if len(finished) > budget:
        raise ValueError("budget_error: stored attempts exceed requested global budget")
    queued_or_attempted = {
        t.params.get("candidate_pair")
        or t.system_attrs.get("fixed_params", {}).get("candidate_pair")
        for t in study.get_trials(deepcopy=False)
    }
    for identifier in identifiers:
        if identifier not in queued_or_attempted:
            _, model = mapping[identifier]
            safe = _safe_params(model, task_kind)
            study.enqueue_trial(
                {"candidate_pair": identifier, **{model + "." + k: v for k, v in safe.items()}}
            )

    def objective(trial):
        identifier = trial.suggest_categorical("candidate_pair", identifiers)
        route, model = mapping[identifier]
        params = _suggest_params(trial, model, task_kind)
        trial.set_user_attr("candidate", {"route": route, "model": model, "params": params})
        audit = directory / "trials" / f"{trial.number:06d}.json"
        result = _evaluate(evaluate, route, model, params, trial.number, contract_hash, audit)
        trial.set_user_attr("evaluation_result", result)
        if result["status"] != "completed":
            raise CandidateEvaluationError(result["reason"])
        return result["score"]

    study.optimize(
        objective,
        n_trials=budget - len(finished),
        catch=(CandidateEvaluationError,),
        gc_after_trial=True,
        show_progress_bar=False,
    )
    results = []
    for trial in study.get_trials(deepcopy=False):
        if not trial.state.is_finished():
            continue
        audit = directory / "trials" / f"{trial.number:06d}.json"
        value = _read_trial(audit) if audit.exists() else trial.user_attrs.get("evaluation_result")
        if value is None:
            candidate = trial.user_attrs.get("candidate", {})
            route, model = mapping.get(
                trial.params.get("candidate_pair"), ("unresolved", "unresolved")
            )
            value = {
                "route": route,
                "model": model,
                "params": candidate.get("params", {}),
                "status": "failed",
                "score": None,
                "reason": "interrupted_candidate_materialization",
                "search_trial": trial.number,
                "search_input_signature": contract_hash,
            }
        elif value["status"] == "running":
            value.update(status="failed", score=None, reason="interrupted_candidate_attempt")
        if trial.state != optuna.trial.TrialState.COMPLETE:
            value.update(status="failed", score=None)
        _save_trial(audit, value)
        results.append(value)
    return results, {
        "storage_path": str(path.resolve()),
        "study_name": study.study_name,
        "optuna_version": optuna.__version__,
        "sampler": "TPESampler",
        "pruner": "NopPruner",
        "sampler_rng_resume": "seeded_new_sampler_on_process_resume",
    }


def search_candidates(candidate_pairs, evaluate, config, search_dir, input_signature, task_kind):
    """Attempt at most ``search_budget`` candidates across all eligible pairs.

    Search scores must be computed by the caller's training-only grouped CV.
    Restarts preserve attempted trials and FAIL records; they never reset budget.
    """
    pairs = [(str(route), str(model)) for route, model in candidate_pairs]
    if len(pairs) != len(set(pairs)) or any(
        not route or model not in MODEL_NAMES for route, model in pairs
    ):
        raise ValueError("Candidate pairs must be unique with registered model names")
    budget = int(config.get("search_budget", 120))
    backend = config.get("search_backend", config.get("backend", "optuna"))
    if budget < len(pairs) or budget < 1:
        raise ValueError("budget_error: global budget must cover every eligible pair at least once")
    if task_kind not in {"regression", "classification"}:
        raise ValueError("Unsupported task kind")
    if backend not in {"random", "optuna"}:
        raise ValueError("Unsupported explicit search backend: " + str(backend))
    if not pairs:
        return {
            "status": "deferred",
            "reason": "no_eligible_candidate_pairs",
            "trials": [],
            "backend": backend,
            "budget": {"requested": budget, "used": 0, "remaining": budget, "pairs": 0},
        }
    optuna = None
    if backend == "optuna":
        try:
            import optuna
        except ImportError as exc:
            raise SearchBackendUnavailable(
                "Requested Optuna backend is not installed; no fallback was used"
            ) from exc
    contract = {
        "schema": SCHEMA,
        "input_signature": _plain(input_signature),
        "pairs": pairs,
        "task_kind": task_kind,
        "backend": backend,
        "seed": int(config.get("seed", 42)),
        "backend_version": optuna.__version__ if optuna is not None else "stdlib_random",
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    fingerprint = _digest(contract)
    directory = Path(search_dir) / fingerprint[:24]
    (directory / "trials").mkdir(parents=True, exist_ok=True)
    with _exclusive(directory):
        _write(directory / "contract.json", contract)
        if backend == "random":
            trials, details = _random_search(
                pairs, evaluate, config, directory, fingerprint, budget, task_kind
            )
        else:
            trials, details = _optuna_search(
                pairs, evaluate, config, directory, fingerprint, budget, task_kind, optuna
            )
        successful = [t for t in trials if t["status"] == "completed"]
        covered = sorted({(t["route"], t["model"]) for t in trials})
        result = {
            "schema": SCHEMA,
            "status": "completed" if successful else "no_successful_candidates",
            "backend": backend,
            "namespace": fingerprint,
            "search_dir": str(directory.resolve()),
            "trials": trials,
            "completed": len(successful),
            "failed": len(trials) - len(successful),
            "budget": {
                "requested": budget,
                "used": len(trials),
                "remaining": budget - len(trials),
                "pairs": len(pairs),
            },
            "coverage_attempted": [list(p) for p in covered],
            "contract": contract,
            "scope": "training_only_grouped_cv; heldout_not_available_to_search",
            **details,
        }
        _write(directory / "search_result.json", result)
    return result
