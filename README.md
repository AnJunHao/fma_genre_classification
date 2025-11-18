# FMA: Free Music Archive Dataset Analysis

This project performs music genre classification on the Free Music Archive (FMA) dataset using various machine learning models including Support Vector Machines (SVM), Logistic Regression, Decision Trees, and Random Forests. It includes comprehensive data loading, exploratory data analysis (EDA), and hyperparameter tuning capabilities.

## Table of Contents

1. [Installation and Environment Setup](#1-installation-and-environment-setup)
2. [Dataset Download](#2-dataset-download)
3. [Project Structure](#3-project-structure)
4. [Usage Examples](#4-usage-examples)

---

## 1. Installation and Environment Setup

This project uses **`uv`** for fast and reliable package and environment management.

### Step 1: Install `uv`

Follow the official installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

Quick install (macOS/Linux):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Quick install (Windows):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Clone Repository and Navigate to Project

```bash
git clone <repository-url>
cd fma
```

### Step 3: Install Dependencies

```bash
uv sync
```

That's it! `uv sync` will automatically:
- Create a virtual environment
- Install Python 3.14 if needed
- Install all dependencies from `uv.lock`

The environment is ready to use immediately after `uv sync` completes.

---

## 2. Dataset Download

The project requires the FMA metadata files for analysis.

### Download Instructions

1. **Download Link**: [fma_metadata.zip](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip)
2. **Dataset Information**: For comprehensive details about the Free Music Archive dataset, visit the official repository: [mdeff/fma](https://github.com/mdeff/fma)

### Setup

After downloading, extract the ZIP file and place the contents in a directory named `fma_metadata` at the project root:

```
project_root/
├── fma_metadata/          # Place extracted files here
│   ├── tracks.csv         # Track metadata (bit_rate, duration, listens, etc.)
│   ├── genres.csv         # Genre hierarchy information
│   ├── features.csv       # Librosa audio features
│   ├── echonest.csv       # Echonest audio features
│   ├── README.txt         # Dataset documentation
│   └── ...
├── main.py
├── pyproject.toml
└── ...
```

**Note**: The dataset reader will automatically create a `.cache` directory inside `fma_metadata/` to store processed Parquet files for faster subsequent loads.

---

## 3. Project Structure

```
.
├── fma/                           # Main source package
│   ├── __init__.py                # Package initialization and exports
│   ├── data.py                    # Data loading, parsing, and FMADataset class
│   ├── eda.py                     # Exploratory Data Analysis functions
│   ├── plain.py                   # Console output utilities using Rich library
│   ├── types.py                   # Custom type definitions and aliases
│   └── model/                     # Machine learning models
│       ├── __init__.py            # Model package initialization
│       ├── best.py                # Best model selection and parameters
│       ├── dt.py                  # Decision Tree classifier
│       ├── lr.py                  # Logistic Regression classifier
│       ├── rf.py                  # Random Forest classifier
│       ├── svm.py                 # Support Vector Machine classifier
│       └── utils.py               # Shared model utilities (evaluation, metrics)
│
├── fma_metadata/                  # Dataset directory (download required)
│   ├── .cache/                    # Auto-generated cache (Parquet files)
│   ├── tracks.csv
│   ├── genres.csv
│   ├── features.csv
│   └── echonest.csv
│
├── result/                        # Output directory for results
│
├── main.py                        # Main entry point for EDA visualizations
├── dt_grid_search.py              # Hyperparameter tuning for Decision Trees
├── lr_grid_search.py              # Hyperparameter tuning for Logistic Regression
├── rf_grid_search.py              # Hyperparameter tuning for Random Forest
├── svm_grid_search.py             # Hyperparameter tuning for SVM
│
├── pyproject.toml                 # Project metadata and dependencies
├── uv.lock                        # Locked dependency versions
└── README.md                      # This file
```

### Root Directory Files

- **`main.py`**: Primary entry point for exploratory data analysis
  - Loads the FMA dataset using `read_dataset()`
  - Generates PCA analysis plot (`pca_plot.png`)
  - Creates genre hierarchy tree visualization (`genre_tree.png`)
  - Produces track feature distribution plots (`track_describe.png`)

- **`*_grid_search.py`**: Hyperparameter tuning scripts
  - `dt_grid_search.py`: Decision Tree hyperparameter optimization
  - `lr_grid_search.py`: Logistic Regression hyperparameter optimization
  - `rf_grid_search.py`: Random Forest hyperparameter optimization
  - `svm_grid_search.py`: SVM hyperparameter optimization
  - Each script performs multi-stage grid search with cross-validation
  - Results are saved to `result/` directory as CSV files

- **`pyproject.toml`**: Project configuration file
  - Defines project metadata (name, version, description)
  - Lists all dependencies with version constraints
  - Specifies Python version requirement (>=3.14)
  - Contains tool configurations (basedpyright, uv)

---

## 4. Usage Examples

### Data Loading and EDA

The `FMADataset` class (in `fma/data.py`) is the central data container holding all dataset components. It provides methods like `prepare_train_test()` for creating stratified splits and `remove_rare_genres()` for filtering.

```python
from fma import read_dataset
from fma.eda import plot_pca, draw_genre_tree, describe_tracks

# Load dataset (first run converts CSV to Parquet cache for 10-100x faster subsequent loads)
dataset = read_dataset("fma_metadata", cache=True, verbose=True)

# Generate visualizations
plot_pca(dataset, save_file="pca_plot.png")                    # PCA variance analysis
draw_genre_tree(dataset, save_file="genre_tree.png")           # Genre hierarchy tree
describe_tracks(dataset, save_file="track_describe.png")       # Track statistics (6-panel)
```

### Model Training

Each model module (`fma/model/svm.py`, `lr.py`, `dt.py`, `rf.py`) provides `<model>_train_eval()` for single training runs and `<model>_grid_search()` for hyperparameter tuning.

**Single model training:**

```python
from fma import read_dataset
from fma.model.svm import svm_train_eval
from imblearn.over_sampling import SMOTE

dataset = read_dataset("fma_metadata", cache=True)
dataset.remove_rare_genres()  # Remove infrequent genres

# Train and evaluate SVM on root genres
clf, metrics_df = svm_train_eval(
    dataset,
    genre_set="root",        # Options: "all", "root", "non_root", or list of genre IDs
    oversampler=SMOTE,       # Handle class imbalance
    kernel="rbf",
    C=1.0
)

print(metrics_df)  # Shows per-genre and aggregate (MACRO/MICRO/WEIGHTED) metrics
```

**Hyperparameter grid search:**

```python
from fma import read_dataset, svm_grid_search
from imblearn.over_sampling import SMOTE, ADASYN

dataset = read_dataset("fma_metadata", cache=True)
dataset.remove_rare_genres()

# Perform grid search across multiple hyperparameters
results_df, best_models = svm_grid_search(
    dataset,
    genre_set="root",
    oversampler=[SMOTE, ADASYN, None],
    kernel=["rbf", "linear"],
    C=[0.1, 1.0, 10.0],
    gamma=["scale", "auto"],
    save_file="result/svm_grid_search.csv"
)

# Access best model by metric
best_micro_f1 = best_models["micro"]
print(f"Best Micro F1: {best_micro_f1['score']:.4f}")
print(f"Best params: {best_micro_f1['params']}")
```

**Using pre-tuned best models:**

```python
from fma import read_dataset
from fma.model.best import get_best_model

dataset = read_dataset("fma_metadata", cache=True)
dataset.remove_rare_genres()

# Get best pre-tuned model for root genres
model, metrics_df = get_best_model(dataset, genre_set="root")

print("Best Model Performance:")
print(metrics_df.loc[metrics_df["genre"] == "MICRO"])
```
