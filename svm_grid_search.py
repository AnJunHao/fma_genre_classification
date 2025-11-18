# %%
import numpy as np
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SVMSMOTE,
    BorderlineSMOTE,
    RandomOverSampler,
)

from fma import read_dataset, svm_grid_search

dataset = read_dataset("fma_metadata", cache=True, verbose=True)
dataset.remove_rare_genres()
oversampler_grid = [SMOTE, SVMSMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler, None]

# %%
for subset in ("root", "non_root"):
    df, result = svm_grid_search(
        dataset,
        genre_set=subset,
        oversampler=oversampler_grid,
        kernel="rbf",
        gamma="scale",
        C=1.0,
        save_file=f"result/{subset.replace('-', '_')}_svm_01.csv",
    )
    best_oversampler = result["micro"]["params"]["oversampler"]
    df, result = svm_grid_search(
        dataset,
        genre_set=subset,
        oversampler=best_oversampler,
        kernel="rbf",
        gamma="scale",
        C=np.logspace(-5, 4, 10),
        save_file=f"result/{subset.replace('-', '_')}_svm_02.csv",
    )
    best_c = result["micro"]["params"]["C"]
    df, result = svm_grid_search(
        dataset,
        genre_set=subset,
        oversampler=best_oversampler,
        kernel=["rbf", "linear", "poly", "sigmoid"],
        gamma="scale",
        C=best_c,
        save_file=f"result/{subset.replace('-', '_')}_svm_03.csv",
    )
