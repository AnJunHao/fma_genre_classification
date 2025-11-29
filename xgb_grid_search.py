# %%
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SVMSMOTE,
    BorderlineSMOTE,
    RandomOverSampler,
)

from fma import read_dataset, xgb_grid_search

dataset = read_dataset("fma_metadata", cache=True, verbose=True)
dataset.remove_rare_genres()
oversampler_grid = [SMOTE, SVMSMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler, None]

# %%
for subset in ("root", "non_root"):
    df, result = xgb_grid_search(
        dataset,
        genre_set=subset,
        oversampler=oversampler_grid,
        max_depth=6,
        learning_rate=0.3,
        n_estimators=100,
        save_file=f"result/{subset.replace('-', '_')}_xgb_01.csv",
    )
    best_oversampler = result["micro"]["params"]["oversampler"]
    df, result = xgb_grid_search(
        dataset,
        genre_set=subset,
        oversampler=best_oversampler,
        max_depth=6,
        learning_rate=0.3,
        n_estimators=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
        save_file=f"result/{subset.replace('-', '_')}_xgb_02.csv",
    )
    best_n_estimators = result["micro"]["params"]["n_estimators"]
    df, result = xgb_grid_search(
        dataset,
        genre_set=subset,
        oversampler=best_oversampler,
        max_depth=[3, 4, 5, 6, 7, 8, 9, 10],
        learning_rate=0.3,
        n_estimators=best_n_estimators,
        save_file=f"result/{subset.replace('-', '_')}_xgb_03.csv",
    )
    best_max_depth = result["micro"]["params"]["max_depth"]
    df, result = xgb_grid_search(
        dataset,
        genre_set=subset,
        oversampler=best_oversampler,
        max_depth=best_max_depth,
        learning_rate=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
        n_estimators=best_n_estimators,
        save_file=f"result/{subset.replace('-', '_')}_xgb_04.csv",
    )
