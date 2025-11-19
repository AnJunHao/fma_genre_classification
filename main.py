# %%
from fma import get_best_model, read_dataset
from fma.eda import (
    describe_tracks,
    draw_genre_tree,
    plot_classification_report,
    plot_pca,
)

dataset = read_dataset("fma_metadata", cache=True, verbose=True)
dataset.remove_rare_genres(divider=1000)
plot_pca(dataset, save_file="pca_plot.png")
draw_genre_tree(
    dataset,
    save_file="genre_tree.png",
    num_rows=3,
    width_variance_threshold=1,
)
describe_tracks(dataset, save_file="track_describe.png")


root_model, root_df = get_best_model(dataset, "root", verbose=True)
non_root_model, non_root_df = get_best_model(dataset, "non_root", verbose=True)

plot_classification_report(
    root_df,
    "root_classification_report.png",
    "Classification Report for Root Genres",
    sort_by="f1",
    top_n=10,
)
plot_classification_report(
    non_root_df,
    "non_root_classification_report.png",
    "Classification Report for Non-Root Genres",
    sort_by="f1",
    top_n=10,
)
