from collections.abc import Iterable
from itertools import product
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling.base import BaseOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

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
from fma.types import DataFrame


@with_status(transient=False)
def rf_train_eval(
    dataset: FMADataset,
    genre_set: Literal["all", "root", "non-root"] | Iterable[int] = "all",
    random_state: int = 42,
    test_size: float = 0.2,
    oversampler: type[BaseOverSampler] | None = SMOTE,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    n_estimators: int = 100,
    *,
    verbose: bool = True,
) -> tuple[OneVsRestClassifier, DataFrame[int, int, float | str]]:
    X_train, X_test, Y_train, Y_test, _ = dataset.prepare_train_test_multi(
        genre_set, test_size=test_size, random_state=random_state, verbose=verbose
    )

    # Construct the base classifier with appropriate class_weight depending on oversampling
    base_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=None if oversampler is not None else "balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    with console.status("Training model...", disable=not verbose):
        clf = build_ovr_with_optional_oversampling(
            base_estimator=base_clf,
            oversampler_cls=oversampler,
            random_state=random_state,
        ).fit(X_train, Y_train)

    with console.status("Evaluating model...", disable=not verbose):
        Y_pred = clf.predict(X_test)

        df = evaluation_dataframe_from_dataset(dataset, Y_test, Y_pred)  # type: ignore
    return clf, df


class ModelParams(TypedDict):
    oversampler: type[BaseOverSampler] | None
    oversampler_name: str
    max_depth: int | None
    min_samples_leaf: int
    n_estimators: int


@with_status(transient=False)
def rf_grid_search(
    dataset: FMADataset,
    oversampler: Iterable[type[BaseOverSampler] | None] | type[BaseOverSampler] | None,
    max_depth: Iterable[int | None] | int | None,
    min_samples_leaf: Iterable[int] | int,
    n_estimators: Iterable[int] | int,
    genre_set: Literal["all", "root", "non-root"] | Iterable[int] = "all",
    random_state: int = 42,
    test_size: float = 0.2,
    save_file: PathLike | None = None,
    *,
    verbose: bool = True,
) -> tuple[DataFrame[str, int, str | int | float], BestModelResults[ModelParams]]:
    oversampler_options = ensure_iterable_option(oversampler)
    max_depth_options = ensure_iterable_option(max_depth)
    min_samples_leaf_options = ensure_iterable_option(min_samples_leaf)
    n_estimators_options = ensure_iterable_option(n_estimators)

    param_grid = list(
        product(
            oversampler_options,
            max_depth_options,
            min_samples_leaf_options,
            n_estimators_options,
        )
    )

    if not param_grid:
        raise ValueError(
            "No hyperparameter configurations were provided for the grid search."
        )

    results_records: list[dict[str, Any]] = []

    best_by_metric: BestModelResults = init_best_model_results()

    for (
        oversampler_cls,
        max_depth_,
        min_samples_leaf_,
        n_estimators_,
    ) in console.track(
        param_grid,
        disable=not verbose,
        desc=f"Searching {len(param_grid)} parameter sets",
    ):
        oversampler_label = getattr(oversampler_cls, "__name__", str(oversampler_cls))
        max_depth_label = "None" if max_depth_ is None else max_depth_

        with console.status(
            (
                f"Running with (oversampler={oversampler_label}, max_depth={max_depth_label}, "
                f"min_samples_leaf={min_samples_leaf_}, n_estimators={n_estimators_})"
            ),
            disable=not verbose,
        ) as status:
            clf, df_scores = rf_train_eval(
                dataset,
                genre_set=genre_set,
                random_state=random_state,
                test_size=test_size,
                oversampler=oversampler_cls,
                max_depth=max_depth_,
                min_samples_leaf=min_samples_leaf_,
                n_estimators=n_estimators_,
                verbose=False,
            )

            status.update(
                (
                    f"Evaluating with (oversampler={oversampler_label}, max_depth={max_depth_label}, "
                    f"min_samples_leaf={min_samples_leaf_}, n_estimators={n_estimators_})"
                )
            )

            macro_metrics = extract_global_metrics(df_scores, "MACRO")
            micro_metrics = extract_global_metrics(df_scores, "MICRO")
            weighted_metrics = extract_global_metrics(df_scores, "WEIGHTED")

            record: dict[str, Any] = {
                "oversampler": oversampler_label,
                "max_depth": max_depth_label,
                "min_samples_leaf": min_samples_leaf_,
                "n_estimators": n_estimators_,
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

            params: dict[str, Any] = {
                "oversampler": oversampler_cls,
                "oversampler_name": oversampler_label,
                "max_depth": max_depth_,
                "min_samples_leaf": min_samples_leaf_,
                "n_estimators": n_estimators_,
            }

            for metric_name, metrics in (
                ("macro", macro_metrics),
                ("micro", micro_metrics),
                ("weighted", weighted_metrics),
            ):
                maybe_update_best(
                    best_by_metric,
                    metric_name,  # type: ignore[arg-type]
                    metrics["f1"],
                    params,
                    clf,
                    df_scores,
                    metrics,
                )

    results_df = pd.DataFrame(results_records)

    ordered_columns = [
        "oversampler",
        "max_depth",
        "min_samples_leaf",
        "n_estimators",
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
            fmt_spec=["^", "^", "^", "^"] + [".2%"] * 3,
            title="Hyperparameter Grid Search Results",
        )

    return results_df, best_by_metric
