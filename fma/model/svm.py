from collections.abc import Iterable
from itertools import product
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling.base import BaseOverSampler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from fma.data import FMADataset, PathLike
from fma.model.utils import (
    BestModelResults,
    build_ovr_with_optional_oversampling,
    ensure_iterable_option,
    evaluation_dataframe_from_dataset,
    extract_global_metrics,
    init_best_model_results,
    maybe_update_best,
)
from fma.plain import console, with_status
from fma.types import DataFrame, MetricsDF

KernelType = Literal["linear", "poly", "rbf", "sigmoid"]


@with_status(transient=False)
def svm_train_eval(
    dataset: FMADataset,
    genre_set: Literal["all", "root", "non_root"] | Iterable[int] = "all",
    random_state: int = 42,
    test_size: float = 0.2,
    oversampler: type[BaseOverSampler] | None = SMOTE,
    kernel: KernelType = "rbf",
    C: float = 1.0,
    gamma: float | Literal["scale", "auto"] = "scale",
    *,
    n_jobs: int = -1,
    cache_size: int = 200,
    verbose: bool = True,
) -> tuple[OneVsRestClassifier, MetricsDF]:
    X_train, X_test, Y_train, Y_test, _ = dataset.prepare_train_test(
        genre_set, test_size=test_size, random_state=random_state, verbose=verbose
    )

    base_clf = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,  # type: ignore
        cache_size=cache_size,
        class_weight=None if oversampler is not None else "balanced",
        probability=False,
        random_state=random_state,
    )

    with console.status("Training model...", disable=not verbose):
        clf = build_ovr_with_optional_oversampling(
            base_estimator=base_clf,
            oversampler_cls=oversampler,
            random_state=random_state,
            n_jobs=n_jobs,
        ).fit(X_train, Y_train)
    with console.status("Evaluating model...", disable=not verbose):
        Y_pred = clf.predict(X_test)

        df = evaluation_dataframe_from_dataset(dataset, Y_test, Y_pred)  # type: ignore
    return clf, df


class ModelParams(TypedDict):
    oversampler: type[BaseOverSampler] | None
    oversampler_name: str
    kernel: KernelType
    C: float
    gamma: float | Literal["scale", "auto"]


@with_status(transient=False)
def svm_grid_search(
    dataset: FMADataset,
    oversampler: Iterable[type[BaseOverSampler] | None] | type[BaseOverSampler] | None,
    kernel: Iterable[KernelType] | KernelType,
    C: Iterable[float] | float,
    gamma: Iterable[float | Literal["scale", "auto"]]
    | float
    | Literal["scale", "auto"],
    genre_set: Literal["all", "root", "non_root"] | Iterable[int] = "all",
    random_state: int = 42,
    test_size: float = 0.2,
    save_file: PathLike | None = None,
    *,
    n_jobs: int = -1,
    cache_size: int = 200,
    verbose: bool = True,
) -> tuple[DataFrame[str, int, str | int | float], BestModelResults[ModelParams]]:
    oversampler_options = ensure_iterable_option(oversampler)
    kernel_options = ensure_iterable_option(kernel)
    c_options = ensure_iterable_option(C)
    gamma_options = ensure_iterable_option(gamma)

    param_grid = list(
        product(oversampler_options, kernel_options, c_options, gamma_options)
    )

    if not param_grid:
        raise ValueError(
            "No hyperparameter configurations were provided for the grid search."
        )

    results_records: list[dict[str, Any]] = []

    best_by_metric: BestModelResults = init_best_model_results()

    for oversampler_cls, kernel_, C_, gamma_ in console.track(
        param_grid,
        disable=not verbose,
        desc=f"Searching {len(param_grid)} parameter sets",
    ):
        oversampler_label = getattr(oversampler_cls, "__name__", str(oversampler_cls))

        with console.status(
            f"Running with (oversampler={oversampler_label}, kernel={kernel_}, C={C_}, gamma={gamma_})",
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
                gamma=gamma_,
                n_jobs=n_jobs,
                verbose=verbose,
                cache_size=cache_size,
            )

            status.update(
                f"Evaluating with (oversampler={oversampler_label}, kernel={kernel_}, C={C_}, gamma={gamma_})"
            )

            macro_metrics = extract_global_metrics(df_scores, "MACRO")
            micro_metrics = extract_global_metrics(df_scores, "MICRO")
            weighted_metrics = extract_global_metrics(df_scores, "WEIGHTED")

            record: dict[str, Any] = {
                "oversampler": oversampler_label,
                "kernel": kernel_,
                "C": float(C_),
                "gamma": gamma_ if isinstance(gamma_, str) else float(gamma_),
                "support_total": macro_metrics["support"],
            }

            metric_name: Literal["macro", "micro", "weighted"]

            for metric_name, metrics in (  # type: ignore
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
                "gamma": gamma_,
            }

            for metric_name, metrics in (  # type: ignore
                ("macro", macro_metrics),
                ("micro", micro_metrics),
                ("weighted", weighted_metrics),
            ):
                maybe_update_best(
                    best_by_metric,
                    metric_name,
                    metrics["f1"],
                    params,
                    clf,
                    df_scores,
                    metrics,
                )

    results_df = pd.DataFrame(results_records)

    ordered_columns = [
        "oversampler",
        "kernel",
        "C",
        "gamma",
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
        .sort_values(by=["f1_macro", "f1_micro", "f1_weighted"], ascending=False)  # ty: ignore
        .reset_index(drop=True)
    )

    results_df = cast(DataFrame[str, int, str | int | float], results_df)

    if save_file is not None:
        save_file = Path(save_file).absolute()
        save_file.parent.mkdir(parents=True, exist_ok=True)
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
            fmt_spec=["^", "^", ".1e", ".1e"] + [".2%"] * 3,
            title="Hyperparameter Grid Search Results",
        )

    return results_df, best_by_metric
