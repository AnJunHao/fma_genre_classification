from fma import data, eda, linear, plain, svm
from fma.data import read_dataset
from fma.eda import describe_tracks, draw_genre_tree
from fma.linear import lr_grid_search
from fma.plain import console
from fma.svm import svm_grid_search

__all__ = [
    "data",
    "eda",
    "svm",
    "linear",
    "plain",
    "draw_genre_tree",
    "describe_tracks",
    "read_dataset",
    "lr_grid_search",
    "svm_grid_search",
    "console",
]
