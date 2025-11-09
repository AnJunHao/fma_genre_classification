# %%
from fma import read_dataset
from fma.eda import describe_tracks, draw_genre_tree, plot_pca

dataset = read_dataset("fma_metadata", cache=True, verbose=True)
plot_pca(dataset, save_file="pca_plot.png")
draw_genre_tree(dataset, save_file="genre_tree.png")
describe_tracks(dataset, save_file="track_describe.png")
