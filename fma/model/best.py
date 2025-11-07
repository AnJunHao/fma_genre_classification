from typing import Literal, TypedDict

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling.base import BaseOverSampler
from sklearn.multiclass import OneVsRestClassifier

from fma.data import FMADataset
from fma.model.svm import KernelType, svm_train_eval
from fma.types import DataFrame


class BestModelParams(TypedDict):
    oversampler: type[BaseOverSampler] | None
    kernel: KernelType
    C: float


best_model_params: dict[Literal["root", "non-root"], BestModelParams] = {
    "root": {
        "oversampler": RandomOverSampler,
        "kernel": "rbf",
        "C": 1.0,
    },
    "non-root": {
        "oversampler": None,
        "kernel": "rbf",
        "C": 1.0,
    },
}


def get_best_model(
    dataset: FMADataset,
    genre_set: Literal["root", "non-root"],
    n_jobs: int = -1,
    *,
    verbose: bool = True,
) -> tuple[OneVsRestClassifier, DataFrame[int, int, float | str]]:
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
    return model, df
