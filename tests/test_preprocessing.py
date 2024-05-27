"""Test the data preprocessing methods."""

import pandas as pd
from ab_tools.data import Preprocessing


def test_preprocessing_pipeline():
    test_data = pd.DataFrame(
        {
            "ids": [1, 2, 3, 4, 5],
            "num feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "num feature2": [0.1, 0.3, 0.5, 0.7, 0.9],
            "cat feature1": ["a", "b", "c", "d", "e"],
            "cat feature2": ["yesy", "no", "yes", "yes", "no"],
            "target labels": [1, 0, 0, 1, 0],
        }
    )

    id_col = ["ids"]
    target_col = ["target labels"]

    preprocessing = Preprocessing(test_data, id_col, target_col)
    # set pipeline
    preprocessing.set_pipeline()

    # transform data
    preprocessing.pipeline.fit_transform(test_data)

    # check numeric columns
    assert preprocessing.numeric_cols[0] == "num feature1"
    assert preprocessing.numeric_cols[1] == "num feature2"
    # check cateogrical columns
    assert preprocessing.categorical_cols[0] == "cat feature1"
    assert preprocessing.categorical_cols[1] == "cat feature2"
