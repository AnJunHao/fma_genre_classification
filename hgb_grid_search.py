# %%
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SVMSMOTE,
    BorderlineSMOTE,
    RandomOverSampler,
)

from fma import hgb_grid_search, read_dataset

dataset = read_dataset("fma_metadata", cache=True, verbose=True)
dataset.remove_rare_genres()
oversampler_grid = [SMOTE, SVMSMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler, None]

# %%
for subset in ("root", "non_root"):
    df, result = hgb_grid_search(
        dataset,
        genre_set=subset,
        oversampler=oversampler_grid,
        learning_rate=0.1,
        max_iter=100,
        max_depth=None,
        save_file=f"result/{subset.replace('-', '_')}_hgb_01.csv",
        n_jobs=-1,
    )
    best_oversampler = result["micro"]["params"]["oversampler"]
    df, result = hgb_grid_search(
        dataset,
        genre_set=subset,
        oversampler=best_oversampler,
        learning_rate=0.1,
        max_iter=[50, 100, 200, 500, 1000, 2000, 5000],
        max_depth=None,
        save_file=f"result/{subset.replace('-', '_')}_hgb_02.csv",
        n_jobs=-1,
    )
    best_max_iter = result["micro"]["params"]["max_iter"]
    df, result = hgb_grid_search(
        dataset,
        genre_set=subset,
        oversampler=best_oversampler,
        learning_rate=[0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
        max_iter=best_max_iter,
        max_depth=None,
        save_file=f"result/{subset.replace('-', '_')}_hgb_03.csv",
        n_jobs=-1,
    )
    best_learning_rate = result["micro"]["params"]["learning_rate"]
    df, result = hgb_grid_search(
        dataset,
        genre_set=subset,
        oversampler=best_oversampler,
        learning_rate=best_learning_rate,
        max_iter=best_max_iter,
        max_depth=[3, 5, 7, 10, 15, 20, None],
        save_file=f"result/{subset.replace('-', '_')}_hgb_04.csv",
        n_jobs=-1,
    )
