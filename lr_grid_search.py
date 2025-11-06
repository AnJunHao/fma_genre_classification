# %%
import numpy as np
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SVMSMOTE,
    BorderlineSMOTE,
    RandomOverSampler,
)

from fma import lr_grid_search, read_dataset

dataset = read_dataset("fma_metadata", cache=True, verbose=True)
dataset.remove_rare_genres()
oversampler_grid = [SMOTE, SVMSMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler, None]

# %% Grid Search for Best Oversampler (root genres)
df, result = lr_grid_search(
    dataset,
    genre_set="root",
    oversampler=oversampler_grid,
    penalty="l2",
    C=1.0,
    save_file="result/root_lr_01.csv",
)

best_oversampler = result["micro"]["params"]["oversampler"]

# %% Sweep through C values (root genres)
df, result = lr_grid_search(
    dataset,
    genre_set="root",
    oversampler=best_oversampler,
    penalty="l2",
    C=np.logspace(-5, 4, 10),
    save_file="result/root_lr_02.csv",
)

# %% Grid Search for Best Oversampler (non-root genres)
df, result = lr_grid_search(
    dataset,
    genre_set="non-root",
    oversampler=oversampler_grid,
    penalty="l2",
    C=0.1,
    save_file="result/non_root_lr_01.csv",
)

best_oversampler = result["micro"]["params"]["oversampler"]

# %% Sweep through C values (non-root genres)
df, result = lr_grid_search(
    dataset,
    genre_set="non-root",
    oversampler=best_oversampler,
    penalty="l2",
    C=np.logspace(-5, 4, 10),
    save_file="result/non_root_lr_02.csv",
)
