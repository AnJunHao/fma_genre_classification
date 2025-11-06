# %%
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SVMSMOTE,
    BorderlineSMOTE,
    RandomOverSampler,
)

from fma import read_dataset, rf_grid_search

dataset = read_dataset("fma_metadata", cache=True, verbose=True)
dataset.remove_rare_genres()
oversampler_grid = [SMOTE, SVMSMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler, None]

# %%
for subset in ("root", "non-root"):
    df, result = rf_grid_search(
        dataset,
        genre_set=subset,
        oversampler=oversampler_grid,
        min_samples_leaf=1,
        n_estimators=100,
        max_depth=None,
        save_file=f"result/{subset.replace('-', '_')}_rf_01.csv",
    )
    best_oversampler = result["micro"]["params"]["oversampler"]
    df, result = rf_grid_search(
        dataset,
        genre_set=subset,
        oversampler=best_oversampler,
        min_samples_leaf=1,
        n_estimators=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
        max_depth=None,
        save_file=f"result/{subset.replace('-', '_')}_rf_02.csv",
    )
    best_n_estimators = result["micro"]["params"]["n_estimators"]
    df, result = rf_grid_search(
        dataset,
        genre_set=subset,
        oversampler=best_oversampler,
        min_samples_leaf=[1, 2, 3, 4, 5, 6, 7, 8],
        n_estimators=best_n_estimators,
        max_depth=None,
        save_file=f"result/{subset.replace('-', '_')}_rf_03.csv",
    )
