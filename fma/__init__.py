from fma import data, eda, model, plain
from fma.data import read_dataset
from fma.eda import describe_tracks, draw_genre_tree, plot_pca
from fma.model.best import get_best_model
from fma.model.dt import dt_grid_search
from fma.model.hgb import hgb_grid_search
from fma.model.lr import lr_grid_search
from fma.model.rf import rf_grid_search
from fma.model.svm import svm_grid_search
from fma.model.xgb import xgb_grid_search
from fma.plain import console

__all__ = [
    "data",
    "eda",
    "model",
    "plain",
    "draw_genre_tree",
    "describe_tracks",
    "plot_pca",
    "read_dataset",
    "lr_grid_search",
    "svm_grid_search",
    "dt_grid_search",
    "rf_grid_search",
    "hgb_grid_search",
    "xgb_grid_search",
    "get_best_model",
    "console",
]
