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
