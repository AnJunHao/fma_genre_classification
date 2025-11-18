# %%
from fma import get_best_model, read_dataset
from fma.eda import describe_tracks, draw_genre_tree, plot_pca

dataset = read_dataset("fma_metadata", cache=True, verbose=True)
dataset.remove_rare_genres(divider=200)
# plot_pca(dataset, save_file="pca_plot.png")
draw_genre_tree(
    dataset,
    save_file="genre_tree.png",
    num_rows=3,
    width_variance_threshold=1,
    debug=True,
)
# describe_tracks(dataset, save_file="track_describe.png")


root_model, root_df = get_best_model(dataset, "root", verbose=True)
non_root_model, non_root_df = get_best_model(dataset, "non_root", verbose=True)
