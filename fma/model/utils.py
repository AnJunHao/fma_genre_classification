from collections.abc import Iterable
from typing import Any, Literal, TypedDict, cast

import pandas as pd
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsRestClassifier

from fma.data import FMADataset
from fma.types import DataFrame


class GlobalMetrics(TypedDict):
    """Aggregated metrics for a prediction run."""

    precision: float
    recall: float
    f1: float
    support: int
    accuracy: float


class ModelResultBase[T](TypedDict):
    """A generic representation of a single model result for some 'averaging' metric."""

    score: float
    params: T
    metrics: GlobalMetrics
    estimator: OneVsRestClassifier
    results: DataFrame[str, int, float | str]


class BestModelResults[T](TypedDict):
    """Tracks the best models for macro/micro/weighted F1."""

    micro: ModelResultBase[T]
    macro: ModelResultBase[T]
    weighted: ModelResultBase[T]


def ensure_iterable_option(value: Any) -> list[Any]:
    """Normalize a scalar or iterable option into a list, preserving strings/bytes as scalars."""
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def evaluate_predictions_table(
    Y_true: pd.DataFrame,
    Y_pred: pd.DataFrame | Any,
    label_titles: list[str],
) -> DataFrame[str, int, float | str]:
    """Create the per-label and aggregated metrics table.

    Returns a DataFrame with:
    - One row per label containing precision/recall/f1/support
    - Three extra rows for MACRO/MICRO/WEIGHTED aggregates (indexed at -3, -2, -1)
    - Column 'genre' contains the display title for the label (or aggregate name)
    """
    # Per-class metrics
    prec, rec, f1, support = precision_recall_fscore_support(
        Y_true,
        Y_pred,
        average=None,
        zero_division=0,  # type: ignore
    )

    # Calculate per-class accuracy
    per_class_accuracy = (Y_true == Y_pred).mean(axis=0).values

    df = pd.DataFrame(
        {
            "genre": label_titles,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
            "accuracy": per_class_accuracy,
        },
        index=Y_true.columns,  # keep the same index as labels/columns
    )

    # Aggregates
    total_support = int(support.sum())  # type: ignore[arg-type]
    overall_accuracy = accuracy_score(Y_true, Y_pred)
    for index, avg_type in enumerate(["MACRO", "MICRO", "WEIGHTED"], start=-3):
        avg = cast(Literal["macro", "micro", "weighted"], avg_type.lower())
        df.loc[index] = [
            avg_type,
            precision_score(Y_true, Y_pred, average=avg, zero_division=0),  # type: ignore[arg-type]
            recall_score(Y_true, Y_pred, average=avg, zero_division=0),  # type: ignore[arg-type]
            f1_score(Y_true, Y_pred, average=avg, zero_division=0),  # type: ignore[arg-type]
            total_support,
            overall_accuracy,
        ]

    return cast(DataFrame[str, int, float | str], df)


def evaluation_dataframe_from_dataset(
    dataset: FMADataset,
    Y_true: DataFrame[int, int, bool],
    Y_pred: DataFrame[int, int, bool],
) -> DataFrame[str, int, float | str]:
    """Build the evaluation table using the dataset mapping for label titles."""
    titles = [dataset.id_to_genre[c].title for c in Y_true.columns]
    return evaluate_predictions_table(Y_true, Y_pred, titles)


def extract_global_metrics(
    df: pd.DataFrame, label: Literal["MACRO", "MICRO", "WEIGHTED"]
) -> GlobalMetrics:
    """Extract an aggregate metrics row from an evaluation DataFrame."""
    row = df.loc[df["genre"] == label]
    if row.empty:
        raise ValueError(
            f"Aggregated metrics for {label} were not found in the evaluation DataFrame."
        )
    r = row.iloc[0]
    return {
        "precision": float(r["precision"]),
        "recall": float(r["recall"]),
        "f1": float(r["f1"]),
        "support": int(r["support"]),
        "accuracy": float(r["accuracy"]),
    }


def init_best_model_results() -> BestModelResults:
    """Create an empty best-by-aggregate structure with -inf sentinel scores."""
    empty: ModelResultBase = {
        "score": float("-inf"),
        "params": {},  # will be set
        "metrics": {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "support": 0,
            "accuracy": 0.0,
        },
        "estimator": cast(OneVsRestClassifier, None),  # will be set
        "results": cast(DataFrame[str, int, float | str], None),  # will be set
    }
    return {
        "macro": empty.copy(),
        "micro": empty.copy(),
        "weighted": empty.copy(),
    }


def maybe_update_best(
    best: BestModelResults,
    metric_name: Literal["macro", "micro", "weighted"],
    score: float,
    params: dict[str, Any],
    estimator: OneVsRestClassifier,
    results_df: DataFrame[str, int, float | str],
    metrics: GlobalMetrics,
) -> None:
    """Update the best tracker in-place if a new score is better."""
    if score > best[metric_name]["score"]:
        best[metric_name] = {
            "score": score,
            "params": params.copy(),
            "metrics": metrics.copy(),
            "estimator": estimator,
            "results": results_df.copy(),
        }


def build_ovr_with_optional_oversampling(
    base_estimator: BaseEstimator,
    oversampler_cls: type[BaseOverSampler] | None,
    *,
    random_state: int | None = None,
    n_jobs: int = -1,
) -> OneVsRestClassifier:
    """Wrap a base estimator with optional oversampling and OneVsRest.

    Note: This function does not modify classifier-specific parameters like
    class_weight. That conditional logic (e.g. toggling 'balanced' when not
    oversampling) should be handled by the caller when constructing the
    base_estimator.
    """
    if oversampler_cls is not None:
        pipeline = Pipeline(
            [
                ("smote", oversampler_cls(random_state=random_state)),  # type: ignore
                ("clf", base_estimator),
            ]
        )
        base: BaseEstimator = pipeline
    else:
        base = base_estimator
    return OneVsRestClassifier(base, n_jobs=n_jobs)
