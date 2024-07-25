"""Test the exploratory data analysis methods."""

import pandas as pd
from ab_tools.data import Preprocessing, ExploreFeatures

import time

TIMESTAMP = time.strftime("%Y_%m_%d-%H:%M:%S")

TEST_DATA = pd.DataFrame(
    {
        "ids": [1, 2, 3, 4, 5],
        "num feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "num feature2": [0.1, 0.3, 0.5, 0.7, 0.9],
        "cat feature1": ["a", "b", "c", "d", "e"],
        "cat feature2": ["yes", "no", "yes", "yes", "no"],
        "target labels": [1, 0, 0, 1, 0],
    }
)

preprocessing = Preprocessing(TEST_DATA, ["ids"], ["target labels"])

NUMERIC_COLS = preprocessing.numeric_cols
CATEGORICAL_COLS = preprocessing.categorical_cols
TARGET_COL = preprocessing.target_columns

def test_correlation_matrix():
    """Test the correlation matrix plot function."""
    explore = ExploreFeatures(NUMERIC_COLS, CATEGORICAL_COLS, TARGET_COL)
    explore.correlation_matrix(TEST_DATA, plot_name=f"tests/.testfigs/test_{TIMESTAMP}_Correlation_matrix.png")

def test_violin_plot():
    explore = ExploreFeatures(NUMERIC_COLS, CATEGORICAL_COLS, TARGET_COL)
    explore.violin_plots(TEST_DATA, plot_name=f"tests/.testfigs/test_{TIMESTAMP}_Violin_plot.png")

def test_category_heatmap():
    explore = ExploreFeatures(NUMERIC_COLS, CATEGORICAL_COLS,TARGET_COL)
    explore.category_heatmap(TEST_DATA, plot_name=f"tests/.testfigs/test_{TIMESTAMP}_Category_heatmap.png")

def test_scatter_plot():
    """Test the scatter plot function."""
    explore = ExploreFeatures(NUMERIC_COLS, CATEGORICAL_COLS, TARGET_COL)
    explore.scatter_plot(TEST_DATA, plot_name=f"tests/.testfigs/test_{TIMESTAMP}_scatter_plot.png")

def test_perform_eda():
    explore = ExploreFeatures(NUMERIC_COLS, CATEGORICAL_COLS, TARGET_COL)
    explore.perform_eda(TEST_DATA, plot_name=f"tests/.testfigs/test_{TIMESTAMP}_EDA.png")