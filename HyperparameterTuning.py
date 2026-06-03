"""Utilities for two-stage hyperparameter tuning.

Stage 1: randomized search to quickly ballpark useful regions.
Stage 2: explicit parameter search centered around top randomized results.

The explicit stage can be run independently without the randomized stage.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter
import pickle
from typing import Any

from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from scipy.stats import expon, gamma, uniform, loguniform, randint

from math import gamma, factorial



def calculate_factorial(number:int|float):
    """
    where number can be in >=0 or float >=2
    this returns float(factorial)
    """
    if number<0:
        raise ValueError("Can't compute factorials for negative numbers")
    elif number<=1.000001:
        return 1
    elif number > 171:
        number=int(round(number))
        return int(factorial(number))
    else:
        return int(round(gamma(number-1),0))

def calculate_num_combinations(total_population, selection_size):
    """
    """
    #Combination(n,r) == Combination(total_population,selection_size) == total_population!/(selection_size!(total_population-selection_size)!)
    factorial_total_population=calculate_factorial(total_population)
    factorial_selection_size=calculate_factorial(selection_size)
    total_population_minus_selection_size=total_population - selection_size
    if total_population_minus_selection_size<0:
        raise ValueError("Selection size can't be greater than population size.")
    factorial_total_population_minus_selection_size=calculate_factorial(total_population_minus_selection_size)
    return int(round(factorial_total_population / ( factorial_selection_size * factorial_total_population_minus_selection_size ),0))


def _is_numeric(value: Any) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def _safe_float(value: Any) -> float | Any:
  if _is_numeric(value):
    return float(value)
  return value


def _unique_path(path_like: str | Path, overwrite: bool = False) -> Path:
  """Return a unique path when overwrite is disabled.

  If overwrite is False and the path exists, suffixes _1, _2, ... are added
  before the extension.
  """
  path = Path(path_like)
  if overwrite or (not path.exists()):
    return path

  stem = path.stem
  suffix = path.suffix
  parent = path.parent
  counter = 1
  while True:
    candidate = parent / f"{stem}_{counter}{suffix}"
    if not candidate.exists():
      return candidate
    counter += 1


def _normalize_for_serialization(obj: Any) -> Any:
  """Convert non-JSON-safe scalar values into plain Python types."""
  if isinstance(obj, dict):
    return {k: _normalize_for_serialization(v) for k, v in obj.items()}
  if isinstance(obj, list):
    return [_normalize_for_serialization(v) for v in obj]
  if isinstance(obj, tuple):
    return tuple(_normalize_for_serialization(v) for v in obj)
  if _is_numeric(obj):
    return _safe_float(obj)
  return obj


def _score_key(score: Any) -> str:
  """Stable key for score-index dictionaries."""
  try:
    return f"{float(score):.12g}"
  except Exception:
    return str(score)


def _coerce_iterable(value: Any) -> list[Any]:
  if isinstance(value, (list, tuple)):
    return list(value)
  return [value]


def _validate_non_empty_data(X_train: Any, y_train: Any, X_val: Any, y_val: Any) -> None:
  """Fail fast on empty train/validation inputs."""
  if len(X_train) == 0 or len(y_train) == 0:
    raise ValueError("Training data cannot be empty")
  if len(X_val) == 0 or len(y_val) == 0:
    raise ValueError("Validation data cannot be empty")


def _validate_optional_selection_data(X_select: Any, y_select: Any) -> None:
  if (X_select is None) ^ (y_select is None):
    raise ValueError("X_select and y_select must be provided together")
  if X_select is not None and (len(X_select) == 0 or len(y_select) == 0):
    raise ValueError("Selection data cannot be empty")


def _canonical_metric_name(metric_name: str) -> str:
  key = metric_name.strip().lower().replace("-", "_").replace(" ", "_")
  alias = {
    "rocauc": "roc_auc",
    "auc": "roc_auc",
  }
  return alias.get(key, key)


def _normalize_selection_metric_priority(
  selection_metric_priority: list[tuple[Any, ...]] | None,
) -> list[tuple[str, float, float | None]]:
  if selection_metric_priority is None:
    return []
  if not isinstance(selection_metric_priority, list) or len(selection_metric_priority) == 0:
    raise ValueError(
      "selection_metric_priority must be a non-empty list of (metric, pad[, floor]) tuples"
    )

  allowed = {"recall", "precision", "f1", "accuracy", "roc_auc"}
  normalized = []
  for item in selection_metric_priority:
    if not isinstance(item, (list, tuple)) or len(item) not in (2, 3):
      raise ValueError(
        "Each selection metric rule must be a tuple: (metric_name, pad) or (metric_name, pad, floor)"
      )
    metric = item[0]
    pad = item[1]
    floor = item[2] if len(item) == 3 else None
    metric_name = _canonical_metric_name(str(metric))
    if metric_name not in allowed:
      raise ValueError(f"Unsupported metric '{metric}'. Allowed metrics: {sorted(allowed)}")
    pad_val = float(pad)
    if pad_val < 0:
      raise ValueError("Metric pads must be >= 0")
    floor_val = None if floor is None else float(floor)
    if floor_val is not None and floor_val < 0:
      raise ValueError("Metric floors must be >= 0")
    normalized.append((metric_name, pad_val, floor_val))
  return normalized


def _predict_score_vector(model: Any, X_data: Any, y_pred: Any) -> Any:
  if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X_data)
    if hasattr(proba, "ndim") and proba.ndim == 2:
      if proba.shape[1] > 1:
        return proba[:, 1]
      return proba[:, 0]
    return proba
  if hasattr(model, "decision_function"):
    return model.decision_function(X_data)
  return y_pred


def _compute_selection_metrics(y_true: Any, y_pred: Any, y_score: Any) -> dict[str, float]:
  metrics = {
    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "roc_auc": float("-inf"),
  }
  try:
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
  except Exception:
    metrics["roc_auc"] = float("-inf")
  return metrics


def _select_trial_index_by_priority(
  results: list[dict[str, Any]],
  selection_metric_priority: list[tuple[str, float, float | None]],
) -> int:
  survivors = list(range(len(results)))
  if len(survivors) == 1:
    return survivors[0]

  for metric_name, pad, floor in selection_metric_priority:
    best_metric = max(results[i]["selection_metrics"][metric_name] for i in survivors)
    threshold = best_metric - pad
    if floor is not None:
      threshold = max(threshold, floor)
    candidate_survivors = [
      i for i in survivors if results[i]["selection_metrics"][metric_name] >= threshold
    ]
    if candidate_survivors:
      survivors = candidate_survivors
    if len(survivors) <= 1:
      break

  survivors = sorted(
    survivors,
    key=lambda i: (results[i]["score"], -results[i]["trial_index"]),
    reverse=True,
  )
  return survivors[0]


def _make_ranked_results(run_results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
  ranked = sorted(run_results, key=lambda x: x["score"], reverse=True)
  by_score = {}
  for item in ranked:
    key = _score_key(item["score"])
    by_score.setdefault(key, []).append(
      {
        "trial_index": item["trial_index"],
        "params": item["params"],
        "model_name": item["model_name"],
        "elapsed_seconds": item["elapsed_seconds"],
      }
    )
  return ranked, by_score


def _format_trial_error(exc: Exception) -> tuple[str, str]:
  """Return compact, notebook-friendly exception metadata for failed trials."""
  return type(exc).__name__, str(exc)


def randomized_search(
  estimator: Any,
  X_train: Any,
  y_train: Any,
  X_val: Any,
  y_val: Any,
  score_fn,
  param_distributions: dict[str, Any],
  n_iter: int = 25,
  random_state: int = 42,
  X_select: Any | None = None,
  y_select: Any | None = None,
  selection_metric_priority: list[tuple[str, float]] | None = None,
) -> dict[str, Any]:
  """Run randomized hyperparameter trials and return ranked metadata.

  score_fn signature: score_fn(y_true, y_pred) -> float
  """
  if n_iter <= 0:
    raise ValueError("n_iter must be > 0")
  if not isinstance(param_distributions, dict) or not param_distributions:
    raise ValueError("param_distributions must be a non-empty dict")

  _validate_non_empty_data(X_train, y_train, X_val, y_val)
  _validate_optional_selection_data(X_select, y_select)
  normalized_priority = _normalize_selection_metric_priority(selection_metric_priority)

  samples = list(
    ParameterSampler(
      param_distributions=param_distributions,
      n_iter=n_iter,
      random_state=random_state,
    )
  )

  results = []
  successful_models = []
  failed_trials = []
  best_estimator = None
  best_score = float("-inf")
  best_params = None

  if X_select is None:
    X_select_eval, y_select_eval = X_val, y_val
    selection_dataset_used = "validation"
  else:
    X_select_eval, y_select_eval = X_select, y_select
    selection_dataset_used = "selection_population"

  for trial_idx, params in enumerate(samples, start=1):
    model = clone(estimator)
    model.set_params(**params)

    start = perf_counter()
    try:
      model.fit(X_train, y_train)
      pred = model.predict(X_val)
      score = float(score_fn(y_val, pred))

      select_pred = model.predict(X_select_eval)
      select_score_vector = _predict_score_vector(model, X_select_eval, select_pred)
      selection_metrics = _compute_selection_metrics(y_select_eval, select_pred, select_score_vector)

      elapsed = perf_counter() - start

      trial = {
        "trial_index": trial_idx,
        "params": deepcopy(params),
        "score": score,
        "elapsed_seconds": elapsed,
        "model_name": model.__class__.__name__,
        "status": "success",
        "selection_metrics": selection_metrics,
      }
      results.append(trial)
      successful_models.append(model)

      if score > best_score:
        best_score = score
        best_params = deepcopy(params)
        best_estimator = model
    except Exception as exc:
      elapsed = perf_counter() - start
      error_type, error_message = _format_trial_error(exc)
      failed_trials.append(
        {
          "trial_index": trial_idx,
          "params": deepcopy(params),
          "score": float("-inf"),
          "elapsed_seconds": elapsed,
          "model_name": model.__class__.__name__,
          "status": "failed",
          "error_type": error_type,
          "error_message": error_message,
        }
      )

  ranked, by_score = _make_ranked_results(results)

  if ranked and normalized_priority:
    selected_idx = _select_trial_index_by_priority(ranked, normalized_priority)
    best_trial = ranked[selected_idx]
    best_score = best_trial["score"]
    best_params = deepcopy(best_trial["params"])
    model_lookup = {item["trial_index"]: model for item, model in zip(results, successful_models)}
    best_estimator = model_lookup[best_trial["trial_index"]]

  return {
    "mode": "randomized",
    "n_trials": len(samples),
    "n_successful_trials": len(results),
    "n_failed_trials": len(failed_trials),
    "results": ranked,
    "failed_trials": failed_trials,
    "score_to_model_params": by_score,
    "best_score": best_score,
    "best_params": best_params,
    "best_estimator": best_estimator,
    "selection_metric_priority": normalized_priority,
    "selection_dataset_used": selection_dataset_used,
  }


def _build_numeric_candidates(center_value: Any, multipliers: tuple[float, ...] | list[float]) -> list[Any]:
  values = []
  for mult in multipliers:
    candidate = center_value * mult
    if isinstance(center_value, int) and not isinstance(center_value, bool):
      candidate = int(round(candidate))
      if candidate < 1:
        candidate = 1
    values.append(candidate)

  deduped = []
  seen = set()
  for val in sorted(values):
    key = str(val)
    if key not in seen:
      seen.add(key)
      deduped.append(val)
  return deduped


def _center_subset(values: list[Any], keep_count: int) -> list[Any]:
    """Keep a centered subset of values with deterministic trimming."""
    n = len(values)
    if keep_count >= n:
        return values
    if keep_count <= 0:
        return []

    left = (n - keep_count) // 2
    right = left + keep_count
    return values[left:right]


def _estimate_grid_size(param_grid: dict[str, list[Any]]) -> int:
    total = 1
    for values in param_grid.values():
        total *= max(len(values), 1)
    return total


def _limit_numeric_candidates(values: list[Any], max_combinations_tested: int | None) -> list[Any]:
    """Limit numeric neighborhood width using calculate_num_combinations.

    We cap pairwise neighborhood combinations C(k, 2) to stay under the provided
    combination limit, then keep a centered subset of size k.
    """
    n = len(values)
    if n <= 2:
        return values
    if max_combinations_tested is None or max_combinations_tested <= 0:
        return values

    keep = n
    while keep > 2 and calculate_num_combinations(keep, 2) > max_combinations_tested:
        keep -= 1

    return _center_subset(values, keep_count=keep)


def _downsample_grid_to_max(param_grid: dict[str, list[Any]], max_combinations: int | None) -> dict[str, list[Any]]:
    """Reduce candidate counts deterministically until grid size and bound are <= cap."""
    if max_combinations is None or max_combinations <= 0:
        return param_grid

    new_grid = {k: list(v) for k, v in param_grid.items()}
    max_iterations = 2000
    iterations = 0

    while _estimate_grid_size(new_grid) > max_combinations:
        iterations += 1
        if iterations > max_iterations:
            raise RuntimeError(
                "Could not reduce parameter grid below max_combinations within iteration limit"
            )

        key = max(new_grid, key=lambda kk: len(new_grid[kk]))
        vals = new_grid[key]
        if len(vals) <= 1:
            break

        new_grid[key] = _center_subset(vals, keep_count=len(vals) - 1)

    return new_grid


def build_refined_grid_from_random_results(
    random_results: dict[str, Any],
    numeric_multipliers: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
    top_k_random: int = 1,
    max_top_k_random: int = 5,
    max_combinations_tested: int = 300,
) -> dict[str, list[Any]]:
    """Build explicit grid from top randomized trials.

    Numeric params are expanded around their centers with multiplicative
    factors. Non-numeric params are carried forward from top trials.
    """
    ranked = random_results.get("results", [])
    if not ranked:
        raise ValueError("random_results has no trials to refine")

    if top_k_random < 1:
        raise ValueError("top_k_random must be >= 1")
    if top_k_random > max_top_k_random:
        raise ValueError(f"top_k_random cannot exceed {max_top_k_random}")

    if len(ranked) < top_k_random:
      raise ValueError(
        f"Requested top_k_random={top_k_random} but only {len(ranked)} randomized trials are available"
      )

    top_trials = ranked[:top_k_random]
    grid_candidates: dict[str, list[Any]] = {}

    for trial in top_trials:
      params = trial["params"]
      for pname, pval in params.items():
        grid_candidates.setdefault(pname, [])
        if _is_numeric(pval):
          expanded = _build_numeric_candidates(pval, numeric_multipliers)
          grid_candidates[pname].extend(expanded)
        else:
          grid_candidates[pname].append(pval)

    # De-duplicate while preserving sorted order where possible.
    cleaned: dict[str, list[Any]] = {}
    for pname, values in grid_candidates.items():
        unique = []
        seen = set()
        sortable = all(_is_numeric(v) for v in values)
        iter_values = sorted(values) if sortable else values
        for value in iter_values:
            key = str(value)
            if key not in seen:
                seen.add(key)
                unique.append(value)
        if sortable:
            cleaned[pname] = _limit_numeric_candidates(unique, max_combinations_tested)
        else:
            cleaned[pname] = unique

    cleaned = _downsample_grid_to_max(cleaned, max_combinations=max_combinations_tested)
    return cleaned


def grid_search(
  estimator: Any,
  X_train: Any,
  y_train: Any,
  X_val: Any,
  y_val: Any,
  score_fn,
  param_grid: dict[str, Any],
  X_select: Any | None = None,
  y_select: Any | None = None,
  selection_metric_priority: list[tuple[str, float]] | None = None,
) -> dict[str, Any]:
  """Run explicit parameter grid evaluation.

  score_fn signature: score_fn(y_true, y_pred) -> float
  """
  if not isinstance(param_grid, dict) or not param_grid:
    raise ValueError("param_grid must be a non-empty dict")

  _validate_non_empty_data(X_train, y_train, X_val, y_val)
  _validate_optional_selection_data(X_select, y_select)
  normalized_priority = _normalize_selection_metric_priority(selection_metric_priority)

  normalized_grid = {k: _coerce_iterable(v) for k, v in param_grid.items()}
  combos = list(ParameterGrid(normalized_grid))

  results = []
  successful_models = []
  failed_trials = []
  best_estimator = None
  best_score = float("-inf")
  best_params = None

  if X_select is None:
    X_select_eval, y_select_eval = X_val, y_val
    selection_dataset_used = "validation"
  else:
    X_select_eval, y_select_eval = X_select, y_select
    selection_dataset_used = "selection_population"

  for trial_idx, params in enumerate(combos, start=1):
    model = clone(estimator)
    model.set_params(**params)

    start = perf_counter()
    try:
      model.fit(X_train, y_train)
      pred = model.predict(X_val)
      score = float(score_fn(y_val, pred))

      select_pred = model.predict(X_select_eval)
      select_score_vector = _predict_score_vector(model, X_select_eval, select_pred)
      selection_metrics = _compute_selection_metrics(y_select_eval, select_pred, select_score_vector)

      elapsed = perf_counter() - start

      trial = {
        "trial_index": trial_idx,
        "params": deepcopy(params),
        "score": score,
        "elapsed_seconds": elapsed,
        "model_name": model.__class__.__name__,
        "status": "success",
        "selection_metrics": selection_metrics,
      }
      results.append(trial)
      successful_models.append(model)

      if score > best_score:
        best_score = score
        best_params = deepcopy(params)
        best_estimator = model
    except Exception as exc:
      elapsed = perf_counter() - start
      error_type, error_message = _format_trial_error(exc)
      failed_trials.append(
        {
          "trial_index": trial_idx,
          "params": deepcopy(params),
          "score": float("-inf"),
          "elapsed_seconds": elapsed,
          "model_name": model.__class__.__name__,
          "status": "failed",
          "error_type": error_type,
          "error_message": error_message,
        }
      )

  ranked, by_score = _make_ranked_results(results)

  if ranked and normalized_priority:
    selected_idx = _select_trial_index_by_priority(ranked, normalized_priority)
    best_trial = ranked[selected_idx]
    best_score = best_trial["score"]
    best_params = deepcopy(best_trial["params"])
    model_lookup = {item["trial_index"]: model for item, model in zip(results, successful_models)}
    best_estimator = model_lookup[best_trial["trial_index"]]

  return {
    "mode": "grid",
    "n_trials": len(combos),
    "n_successful_trials": len(results),
    "n_failed_trials": len(failed_trials),
    "results": ranked,
    "failed_trials": failed_trials,
    "score_to_model_params": by_score,
    "best_score": best_score,
    "best_params": best_params,
    "best_estimator": best_estimator,
    "selection_metric_priority": normalized_priority,
    "selection_dataset_used": selection_dataset_used,
  }


def save_tuning_results(results: dict[str, Any], save_results: str | None = None, overwrite: bool = False) -> str | None:
    """Persist tuning results as pickle when requested.

    - save_results=None: do not persist.
    - save_results=str: use as filename.
    - overwrite=False: append suffixes to avoid clobbering files.
    """
    if save_results is None:
        return None

    if not isinstance(save_results, str) or not save_results.strip():
        raise ValueError("save_results must be None or a non-empty filename string")

    path = Path(save_results)
    if path.suffix == "":
        path = path.with_suffix(".pkl")

    target = _unique_path(path, overwrite=overwrite)

    payload = {
        "schema_version": 1,
        "results": _normalize_for_serialization(results),
    }

    # Validate serializability before writing any file.
    try:
        pickle.dumps(payload)
    except Exception as exc:
        raise RuntimeError(f"Results payload is not picklable: {exc}") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
      pickle.dump(payload, f)

    return str(target)


def tune_hyperparameters(
  estimator: Any,
  X_train: Any,
  y_train: Any,
  X_val: Any,
  y_val: Any,
  score_fn,
  X_select: Any | None = None,
  y_select: Any | None = None,
  selection_metric_priority: list[tuple[str, float]] | None = None,
  random_param_distributions: dict[str, Any] | None = None,
  explicit_param_grid: dict[str, Any] | None = None,
  run_randomized: bool = True,
  run_explicit: bool = True,
  n_random_iterations: int = 25,
  random_state: int = 42,
  top_k_random: int = 1,
  max_top_k_random: int = 5,
  numeric_multipliers: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
  max_combinations_tested: int = 300,
  save_results: str | None = None,
  overwrite: bool = False,
) -> dict[str, Any]:
  """Wrapper for randomized-only, explicit-only, or two-stage tuning."""
  if not callable(score_fn):
    raise ValueError("score_fn must be callable")

  if (not run_randomized) and (not run_explicit):
    raise ValueError("At least one of run_randomized or run_explicit must be True")

  start_total = perf_counter()
  output = {
    "randomized": None,
    "refined_grid": None,
    "grid": None,
    "best_overall_score": float("-inf"),
    "best_overall_params": None,
    "best_overall_estimator": None,
    "selection_metric_priority": _normalize_selection_metric_priority(selection_metric_priority),
    "selection_dataset_used": "selection_population" if X_select is not None else "validation",
  }

  if run_randomized:
    if not random_param_distributions:
      raise ValueError("random_param_distributions is required when run_randomized=True")

    random_output = randomized_search(
      estimator=estimator,
      X_train=X_train,
      y_train=y_train,
      X_val=X_val,
      y_val=y_val,
      score_fn=score_fn,
      X_select=X_select,
      y_select=y_select,
      selection_metric_priority=selection_metric_priority,
      param_distributions=random_param_distributions,
      n_iter=n_random_iterations,
      random_state=random_state,
    )
    output["randomized"] = random_output

    if random_output["best_score"] > output["best_overall_score"]:
      output["best_overall_score"] = random_output["best_score"]
      output["best_overall_params"] = deepcopy(random_output["best_params"])
      output["best_overall_estimator"] = random_output["best_estimator"]

  if run_explicit:
    if explicit_param_grid is None:
      if not run_randomized:
        raise ValueError(
          "explicit_param_grid is required when run_explicit=True and run_randomized=False"
        )

      if output["randomized"].get("n_successful_trials", 0) == 0:
        raise ValueError(
          "Randomized search completed with zero successful trials. "
          "Provide an explicit_param_grid or narrow the randomized search space."
        )

      explicit_param_grid = build_refined_grid_from_random_results(
        output["randomized"],
        numeric_multipliers=numeric_multipliers,
        top_k_random=top_k_random,
        max_top_k_random=max_top_k_random,
        max_combinations_tested=max_combinations_tested,
      )
      output["refined_grid"] = explicit_param_grid

    grid_output = grid_search(
      estimator=estimator,
      X_train=X_train,
      y_train=y_train,
      X_val=X_val,
      y_val=y_val,
      score_fn=score_fn,
      X_select=X_select,
      y_select=y_select,
      selection_metric_priority=selection_metric_priority,
      param_grid=explicit_param_grid,
    )
    output["grid"] = grid_output

    if grid_output["best_score"] > output["best_overall_score"]:
      output["best_overall_score"] = grid_output["best_score"]
      output["best_overall_params"] = deepcopy(grid_output["best_params"])
      output["best_overall_estimator"] = grid_output["best_estimator"]

  if output["best_overall_estimator"] is None:
    raise ValueError(
      "Hyperparameter tuning finished with zero successful trials across all enabled stages. "
      "Narrow parameter ranges or provide a safer explicit grid."
    )

  output["total_elapsed_seconds"] = perf_counter() - start_total
  output["save_path"] = save_tuning_results(
    output,
    save_results=save_results,
    overwrite=overwrite,
  )

  return output


# allow direct imports through the module
__all__ = [
  "expon",
  "gamma",
  "uniform",
  "loguniform",
  "randint",
  "randomized_search",
  "build_refined_grid_from_random_results",
  "grid_search",
  "save_tuning_results",
  "tune_hyperparameters",
]