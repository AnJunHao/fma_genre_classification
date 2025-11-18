from pathlib import Path
from time import time
from typing import Literal, TypedDict

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling.base import BaseOverSampler
from joblib import dump, load
from sklearn.multiclass import OneVsRestClassifier

from fma.data import FMADataset
from fma.model.svm import KernelType, svm_train_eval
from fma.plain import console
from fma.types import DataFrame


class BestModelParams(TypedDict):
    oversampler: type[BaseOverSampler] | None
    kernel: KernelType
    C: float


best_model_params: dict[Literal["root", "non_root"], BestModelParams] = {
    "root": {
        "oversampler": RandomOverSampler,
        "kernel": "rbf",
        "C": 1.8,
    },
    "non_root": {
        "oversampler": None,
        "kernel": "rbf",
        "C": 1.0,
    },
}


def _get_cache_paths(
    genre_set: Literal["root", "non_root"], hash_key: int
) -> tuple[Path, Path]:
    """Get cache directory and file paths for model and dataframe."""
    cache_dir = Path.cwd() / ".cache"
    model_path = cache_dir / f"best_model_{genre_set}_{hash_key}.joblib"
    df_path = cache_dir / f"best_model_{genre_set}_{hash_key}_df.csv"
    return model_path, df_path


def _load_from_cache(
    genre_set: Literal["root", "non_root"],
    hash_key: int,
    verbose: bool = True,
) -> tuple[OneVsRestClassifier, DataFrame[str, int, float | str]] | None:
    """Load model and dataframe from cache if they exist."""
    model_path, df_path = _get_cache_paths(genre_set, hash_key)

    if model_path.exists() and df_path.exists():
        try:
            with console.status(
                f"Loading best model for {genre_set=} from {model_path}",
                disable=not verbose,
            ):
                start_time = time()
                model = load(model_path)
                df = DataFrame(pd.read_csv(df_path))
            if verbose:
                console.done(
                    f"Loaded best model for {genre_set=} from {model_path} in {time() - start_time:.2f} seconds"
                )
            return model, df
        except Exception as e:
            if verbose:
                console.warn(f"Failed to load cache: {e}")
            return None
    else:
        if verbose:
            console.info(f"No cache found for {genre_set=} from {model_path}")
        return None


def _save_to_cache(
    model: OneVsRestClassifier,
    df: DataFrame[str, int, float | str],
    genre_set: Literal["root", "non_root"],
    hash_key: int,
    verbose: bool = True,
) -> None:
    """Save model and dataframe to cache."""
    model_path, df_path = _get_cache_paths(genre_set, hash_key)

    # Create cache directory if it doesn't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with console.status(
            f"Saving best model for {genre_set=} to {model_path}", disable=not verbose
        ):
            start_time = time()
            dump(model, model_path)
            df.to_csv(df_path, index=False)
        if verbose:
            console.done(
                f"Saved best model for {genre_set=} to {model_path} in {time() - start_time:.2f} seconds"
            )
    except Exception as e:
        if verbose:
            console.warn(f"Failed to save cache: {e}")


def get_best_model(
    dataset: FMADataset,
    genre_set: Literal["root", "non_root"],
    n_jobs: int = -1,
    *,
    verbose: bool = True,
    cache: bool = True,
) -> tuple[OneVsRestClassifier, DataFrame[str, int, float | str]]:
    # Try to load from cache if caching is enabled
    hash_key = abs(hash(tuple(dataset.genre_sets[genre_set])))
    if cache:
        cached = _load_from_cache(genre_set, hash_key, verbose=verbose)
        if cached is not None:
            if cached[0]._genre_set == tuple(dataset.genre_sets[genre_set]):  # type: ignore
                return cached
            else:
                console.warn(f"Cache for {genre_set=} does not match dataset")

    # Train model if not in cache
    params = best_model_params[genre_set]
    oversampler = params["oversampler"]
    kernel = params["kernel"]
    C = params["C"]
    model, df = svm_train_eval(
        dataset=dataset,
        genre_set=genre_set,
        random_state=42,
        oversampler=oversampler,
        kernel=kernel,
        C=C,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # Save to cache if caching is enabled
    if cache:
        model._genre_set = tuple(dataset.genre_sets[genre_set])  # type: ignore
        _save_to_cache(model, df, genre_set, hash_key, verbose=verbose)

    return model, df
