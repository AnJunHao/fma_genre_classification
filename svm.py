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

# %% Grid Search for Best Oversampler (root genres)
df, result = svm_grid_search(
    dataset,
    genre_set="root",
    oversampler=oversampler_grid,
    kernel="rbf",
    C=1.0,
    save_file="result/root_svm_01.csv",
)

best_oversampler = result["micro"]["params"]["oversampler"]

# %% Sweep through C values (root genres)
df, result = svm_grid_search(
    dataset,
    genre_set="root",
    oversampler=best_oversampler,
    kernel="rbf",
    C=np.logspace(-5, 4, 10),
    save_file="result/root_svm_02.csv",
)

best_c = result["micro"]["params"]["C"]

# %% Sweep through kernels (root genres)
df, result = svm_grid_search(
    dataset,
    genre_set="root",
    oversampler=best_oversampler,
    kernel=["rbf", "linear", "poly", "sigmoid"],
    C=best_c,
    save_file="result/root_svm_03.csv",
)

best_kernel = result["micro"]["params"]["kernel"]

# %% Grid Search for Best Oversampler (non-root genres)
df, result = svm_grid_search(
    dataset,
    genre_set="non-root",
    oversampler=oversampler_grid,
    kernel="rbf",
    C=1.0,
    save_file="result/non_root_svm_01.csv",
)

best_oversampler = result["micro"]["params"]["oversampler"]

# %% Sweep through C values (non-root genres)
df, result = svm_grid_search(
    dataset,
    genre_set="non-root",
    oversampler=best_oversampler,
    kernel="rbf",
    C=np.logspace(-5, 4, 10),
    save_file="result/non_root_svm_02.csv",
)

best_c = result["micro"]["params"]["C"]

# %% Sweep through kernels (non-root genres)
df, result = svm_grid_search(
    dataset,
    genre_set="non-root",
    oversampler=best_oversampler,
    kernel=["rbf", "linear", "poly", "sigmoid"],
    C=best_c,
    save_file="result/non_root_svm_03.csv",
)

best_kernel = result["micro"]["params"]["kernel"]
