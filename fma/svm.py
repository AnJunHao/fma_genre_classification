from collections.abc import Iterable
from itertools import product
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from fma.data import FMADataset, PathLike
from fma.plain import console, with_status
from fma.types import DataFrame

KernelType = Literal["linear", "poly", "rbf", "sigmoid"]


@with_status(transient=False)
def svm_train_eval(
    dataset: FMADataset,
    genre_set: Literal["all", "root", "non-root"] | Iterable[int] = "all",
    random_state: int = 42,
    test_size: float = 0.2,
    oversampler: type[BaseOverSampler] | None = SMOTE,
    kernel: KernelType = "rbf",
    C: float = 1.0,
    *,
    verbose: bool = True,
) -> tuple[OneVsRestClassifier, DataFrame[int, int, float | str]]:
    X_train, X_test, Y_train, Y_test, _ = dataset.prepare_train_test_multi(
        genre_set, test_size=test_size, random_state=random_state, verbose=verbose
    )
    if oversampler is not None:
        base = Pipeline(
            [
                ("smote", oversampler(random_state=random_state)),  # type: ignore
                (
                    "clf",
                    SVC(
                        kernel=kernel,
                        C=C,
                        class_weight=None,
                        probability=False,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    else:
        base = SVC(
            kernel=kernel,
            C=C,
            class_weight="balanced",
            probability=False,
            random_state=random_state,
        )
    with console.status("Training model...", disable=not verbose):
        clf = OneVsRestClassifier(base, n_jobs=-1).fit(X_train, Y_train)
    with console.status("Evaluating model...", disable=not verbose):
        Y_pred = clf.predict(X_test)
        prec, rec, f1, support = precision_recall_fscore_support(
            Y_test,
            Y_pred,
            average=None,
            zero_division=0,  # type: ignore
        )
        df = pd.DataFrame(
            {
                "genre": [dataset.id_to_genre[c].title for c in Y_test.columns],
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": support,
            },
            index=Y_test.columns,
        )

        total_support = support.sum()  # type: ignore
        for index, avg_type in enumerate(["MACRO", "MICRO", "WEIGHTED"], start=-3):
            prec_global = precision_score(
                Y_test,
                Y_pred,
                average=avg_type.lower(),
                zero_division=0,  # type: ignore
            )
            rec_global = recall_score(
                Y_test,
                Y_pred,
                average=avg_type.lower(),
                zero_division=0,  # type: ignore
            )
            f1_global = f1_score(
                Y_test,
                Y_pred,
                average=avg_type.lower(),
                zero_division=0,  # type: ignore
            )
            df.loc[index] = [
                avg_type,
                prec_global,
                rec_global,
                f1_global,
                total_support,
            ]

    df = cast(DataFrame[int, int, float | str], df)
    return clf, df


class GlobalMetrics(TypedDict):
    precision: float
    recall: float
    f1: float
    support: int


class ModelParams(TypedDict):
    oversampler: type[BaseOverSampler] | None
    oversampler_name: str
    kernel: KernelType
    C: float


class ModelResult(TypedDict):
    score: float
    params: ModelParams
    metrics: GlobalMetrics
    estimator: OneVsRestClassifier
    results: DataFrame[int, int, float | str]


class BestModelResults(TypedDict):
    micro: ModelResult
    macro: ModelResult
    weighted: ModelResult


@with_status(transient=False)
def svm_grid_search(
    dataset: FMADataset,
    oversampler: Iterable[type[BaseOverSampler] | None] | type[BaseOverSampler] | None,
    kernel: Iterable[KernelType] | KernelType,
    C: Iterable[float] | float,
    genre_set: Literal["all", "root", "non-root"] | Iterable[int] = "all",
    random_state: int = 42,
    test_size: float = 0.2,
    save_file: PathLike | None = None,
    *,
    verbose: bool = True,
) -> tuple[DataFrame[str, int, str | int | float], BestModelResults]:
    def _ensure_iterable_option(value: Any) -> list[Any]:
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return list(value)
        return [value]

    def _extract_global_metrics(df: pd.DataFrame, label: str) -> dict[str, float | int]:
        row = df.loc[df["genre"] == label]
        if row.empty:
            raise ValueError(
                f"Aggregated metrics for {label} were not found in the evaluation DataFrame."
            )
        row = row.iloc[0]
        return {
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "support": int(row["support"]),
        }

    oversampler_options = _ensure_iterable_option(oversampler)
    kernel_options = _ensure_iterable_option(kernel)
    c_options = _ensure_iterable_option(C)

    param_grid = list(product(oversampler_options, kernel_options, c_options))
    if not param_grid:
        raise ValueError(
            "No hyperparameter configurations were provided for the grid search."
        )

    results_records: list[dict[str, Any]] = []
    best_by_metric: BestModelResults = {
        "macro": {
            "score": float("-inf"),
            "params": None,  # type: ignore
            "metrics": None,  # type: ignore
            "estimator": None,  # type: ignore
            "results": None,  # type: ignore
        },
        "micro": {
            "score": float("-inf"),
            "params": None,  # type: ignore
            "metrics": None,  # type: ignore
            "estimator": None,  # type: ignore
            "results": None,  # type: ignore
        },
        "weighted": {
            "score": float("-inf"),
            "params": None,  # type: ignore
            "metrics": None,  # type: ignore
            "estimator": None,  # type: ignore
            "results": None,  # type: ignore
        },
    }

    for oversampler_cls, kernel_, C_ in console.track(
        param_grid,
        disable=not verbose,
        desc=f"Searching {len(param_grid)} parameter sets",
    ):
        oversampler_label = getattr(oversampler_cls, "__name__", str(oversampler_cls))

        with console.status(
            f"Running with (oversampler={oversampler_label}, kernel={kernel_}, C={C_})",
            disable=not verbose,
        ) as status:
            clf, df_scores = svm_train_eval(
                dataset,
                genre_set=genre_set,
                random_state=random_state,
                test_size=test_size,
                oversampler=oversampler_cls,
                kernel=kernel_,
                C=float(C_),
                verbose=False,
            )
            status.update(
                f"Evaluating with (oversampler={oversampler_label}, kernel={kernel_}, C={C_})"
            )

            macro_metrics = _extract_global_metrics(df_scores, "MACRO")
            micro_metrics = _extract_global_metrics(df_scores, "MICRO")
            weighted_metrics = _extract_global_metrics(df_scores, "WEIGHTED")

            record: dict[str, Any] = {
                "oversampler": oversampler_label,
                "kernel": kernel_,
                "C": float(C_),
                "support_total": macro_metrics["support"],
            }
            for metric_name, metrics in (
                ("macro", macro_metrics),
                ("micro", micro_metrics),
                ("weighted", weighted_metrics),
            ):
                record[f"precision_{metric_name}"] = metrics["precision"]
                record[f"recall_{metric_name}"] = metrics["recall"]
                record[f"f1_{metric_name}"] = metrics["f1"]
            results_records.append(record)

            params = {
                "oversampler": oversampler_cls,
                "oversampler_name": oversampler_label,
                "kernel": kernel_,
                "C": float(C_),
            }
            for metric_name, metrics in (
                ("macro", macro_metrics),
                ("micro", micro_metrics),
                ("weighted", weighted_metrics),
            ):
                if metrics["f1"] > best_by_metric[metric_name]["score"]:
                    best_by_metric[metric_name] = {
                        "score": metrics["f1"],
                        "params": params.copy(),
                        "metrics": metrics.copy(),
                        "estimator": clf,
                        "results": df_scores.copy(),
                    }

    results_df = pd.DataFrame(results_records)
    ordered_columns = [
        "oversampler",
        "kernel",
        "C",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "support_total",
    ]
    results_df = (
        results_df[ordered_columns]
        .sort_values(by=["f1_macro", "f1_micro", "f1_weighted"], ascending=False)
        .reset_index(drop=True)
    )
    results_df = cast(DataFrame[str, int, str | int | float], results_df)

    if save_file is not None:
        save_file = Path(save_file).absolute()
        results_df.to_csv(save_file, index=False)
        if verbose:
            console.done(f"Saved hyperparameter grid search results to {save_file}")

    if verbose:
        console.table(
            results_df[
                [
                    c
                    for c in results_df.columns
                    if all(term not in c for term in ["precision", "recall", "support"])
                ]
            ],
            fmt_spec=["^", "^", ".1e"] + [".2%"] * 3,
            title="Hyperparameter Grid Search Results",
        )

    return results_df, best_by_metric
