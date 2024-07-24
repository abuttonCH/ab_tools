"""Methods to preprocess data for ML tasks."""
from typing import List, Tuple, Optional, Literal


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

_TASK_TYPES = Literal["Regression", "Categorical"]


class Preprocessing:

    """Class for creating the data preprocessing pipeline."""

    def __init__(
        self,
        data: pd.DataFrame,
        id_columns: List[str],
        target_columns: List[str],
        task_type: _TASK_TYPES = "Categorical",
    ) -> None:
        """Initialize the pipeline.

        Args:
            data: Data to be procesed.
            id_columns: Column(s) containing data point identifiers
            target_columns: Column(s) containing the taget label(s)
            task_type: The type of learning task. If categorical perform
                one-hot encoding on the label values.
        """
        # set initial value of the pipeline
        self.__pipeline = None
        self.id_columns = id_columns
        self.target_columns = target_columns
        self.label_encoder = None
        # remove the ids from the dataframe
        data = self.remove_ids(data, id_columns)
        # separate the features from the target labels
        self.input_features, self.target_labels = self.separate_features_and_target(
            data, target_columns
        )

        # one hot encode target labels if the task is categorical
        if task_type == "Categorical":
            self.label_encoder = LabelEncoder()
            self.target_labels = self.label_encoder.fit_transform(
                self.target_labels.values.ravel()
            )

        # extract numeric and categorical columns from input features
        self.numeric_cols = self.input_features.select_dtypes(
            include=["number"]
        ).columns
        self.categorical_cols = self.input_features.select_dtypes(
            include=["object"]
        ).columns

    def remove_ids(self, data: pd.DataFrame, id_columns: List[str]) -> pd.DataFrame:
        """Remove the columns containing the ids.

        Args:
            data: Data to be procesed.
            id_columns: Column(s) containing data point identifiers

        Returns:
            The dataframe with the id column(s) removed.
        """
        return data.drop(labels=id_columns, axis=1)

    def separate_features_and_target(
        self, dataframe: pd.DataFrame, target_columns: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Separate a pandas dataframe into input features and target labels.

        Args:
            data: Data to be procesed.
            target_columns: Column(s) containing the taget label(s)

        Returns:
            Two separate dataframes
                1. containing the inputs features
                2. containing the target values
        """
        target_labels = dataframe[target_columns]
        input_features = dataframe.drop(labels=target_columns, axis=1)
        return input_features, target_labels

    @property
    def pipeline(self) -> ColumnTransformer:
        """Getter method for pipeline.

        Returns:
            The pipeline ColumnTransformer object.
        """
        if self.__pipeline is None:
            print("Pipeline has not been set!")
            raise ValueError
        else:
            return self.__pipeline

    def set_pipeline(
        self, num_steps: Optional[Pipeline] = None, cat_steps: Optional[Pipeline] = None
    ) -> None:
        """Create a data preprocessing pipeline.

        Using the numeric and categorical column headers this
        method creates pipeline to process data.

        Args:
            num_steps: The processing steps for the numeric columns
            cat_steps: The processing steps for the categorical columns
        """
        if num_steps is None:
            num_steps = self.default_num_steps()

        if cat_steps is None:
            cat_steps = self.default_cat_steps()

        self.__pipeline = self.create_pipeline(num_steps, cat_steps)

    def create_pipeline(
        self, numeric_transformer: Pipeline, categorical_transformer: Pipeline
    ) -> ColumnTransformer:
        """Set the preprocessing pipeline.

        Args:
            numeric_transformer: The pipeline for the numeric features
            categorical_transformer: The pipeline for the categorical features

        Returns:
            The set data preprocessing pipeline object.
        """
        # data conversion pipeline
        created_pipeline = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_cols),
                ("cat", categorical_transformer, self.categorical_cols),
            ]
        )
        return created_pipeline

    def default_num_steps(self) -> Pipeline:
        """Set default steps for the numeric features.

        Returns:
            The pipe for the numeric columns. By default the only
            step is sklearn's standard scaler.
        """
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        return numeric_transformer

    def default_cat_steps(self) -> Pipeline:
        """Set default steps for the categorical features.

        Returns:
            The pipe for the categorical columns. By default the only
            step is sklearn's one hjot encoder.
        """
        categorical_transformer = Pipeline(
            steps=[("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))]
        )
        return categorical_transformer
