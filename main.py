# %%
from fma import read_dataset
from fma.eda import plot_pca

dataset = read_dataset("fma_metadata", cache=True, verbose=True)
plot_pca(dataset)
