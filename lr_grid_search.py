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

# %%
for subset in ("root", "non-root"):
    df, result = lr_grid_search(
        dataset,
        genre_set=subset,
        oversampler=oversampler_grid,
        penalty="l2",
        C=1.0,
        save_file=f"result/{subset.replace('-', '_')}_lr_01.csv",
    )
    best_oversampler = result["micro"]["params"]["oversampler"]
    df, result = lr_grid_search(
        dataset,
        genre_set=subset,
        oversampler=best_oversampler,
        penalty="l2",
        C=np.logspace(-5, 4, 10),
        save_file=f"result/{subset.replace('-', '_')}_lr_02.csv",
    )
