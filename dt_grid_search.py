# %%
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SVMSMOTE,
    BorderlineSMOTE,
    RandomOverSampler,
)

from fma import dt_grid_search, read_dataset

dataset = read_dataset("fma_metadata", cache=True, verbose=True)
dataset.remove_rare_genres()
oversampler_grid = [SMOTE, SVMSMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler, None]

# %%
for subset in ("root", "non_root"):
    df, result = dt_grid_search(
        dataset,
        genre_set=subset,
        oversampler=oversampler_grid,
        min_samples_leaf=1,
        max_features=None,
        max_depth=None,
        save_file=f"result/{subset.replace('-', '_')}_dt_01.csv",
        n_jobs=-1,
    )
    best_oversampler = result["micro"]["params"]["oversampler"]
    df, result = dt_grid_search(
        dataset,
        genre_set=subset,
        oversampler=best_oversampler,
        min_samples_leaf=[1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32],
        max_features=None,
        max_depth=None,
        save_file=f"result/{subset.replace('-', '_')}_dt_02.csv",
        n_jobs=-1,
    )
    best_min_samples_leaf = result["micro"]["params"]["min_samples_leaf"]
    df, result = df, result = dt_grid_search(
        dataset,
        genre_set=subset,
        oversampler=best_oversampler,
        min_samples_leaf=best_min_samples_leaf,
        max_features=["log2", "sqrt", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        max_depth=None,
        save_file=f"result/{subset.replace('-', '_')}_dt_03.csv",
        n_jobs=-1,
    )
