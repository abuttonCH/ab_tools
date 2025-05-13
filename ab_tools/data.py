"""Methods for preprocessing and exploring data for ML tasks."""
from typing import List, Tuple, Optional, Literal


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import math

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
            data: Data to be processed.
            id_columns: Column(s) containing data point identifiers
            target_columns: Column(s) containing the target label(s)
            task_type: The type of learning task. If categorical perform
                encoding on the label values.
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

        # encode target labels if the task is categorical
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
        """Remove the column(s) containing the ids.

        Args:
            data: Data to be processed.
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
            data: Data to be processed.
            target_columns: Column(s) containing the target label(s)

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
        method creates a pipeline to process data.

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
            The initialized preprocessing pipeline.
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
            The pipeline for the numeric columns. By default the only
            step is sklearn's standard scaler.
        """
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        return numeric_transformer

    def default_cat_steps(self) -> Pipeline:
        """Set default steps for the categorical features.

        Returns:
            The pipeline for the categorical columns. By default the only
            step is sklearn's one-hot encoder.
        """
        categorical_transformer = Pipeline(
            steps=[("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))]
        )
        return categorical_transformer


class ExploreFeatures:

    """Class for performing exploratory data analysis on the input features."""

    def __init__(
        self, numeric_cols: pd.Series, categoric_cols: pd.Series, target_col: pd.Series
    ) -> None:
        """Initialize class.

        Args:
            numeric_cols: Column names for the numeric input features.
            categoric_cols: Column names for the categorical input features.
            target_col: Column name for the target labels.
        """
        self.numeric_cols = numeric_cols
        self.categoric_cols = categoric_cols
        self.target_col = target_col

    def show_plot(self, plot_name: str) -> None:
        """Display or save the plot using either plt.show() or plt.savefig().

        Args:
            plot_name: The name for the plot file. If plot_name is an empty string,
                    the plot is displayed using plt.show(). Otherwise, the plot is
                    saved as a file with the given name using plt.savefig().
        """
        plt.savefig(plot_name, bbox_inches="tight") if plot_name != "" else plt.show()
        # close the plot object
        plt.close()

    def update_plotname(
        self, plot_name: str, appended_string: str, replaced_string: str
    ) -> str:
        """Replace a substring within the plot file name with an updated substring.

        Args:
            plot_name: The name of the plot file. If plot_name is an empty string,
                    the function returns an empty string without any updates.
            appended_string: The new substring to replace the old substring in plot_name.
            replaced_string: The substring to be replaced in plot_name.

        Returns:
            The updated plot file name with the replaced substring.
        """
        if plot_name != "":
            plot_name = plot_name.replace(replaced_string, appended_string)
        return plot_name

    def correlation_matrix(
        self,
        data: pd.DataFrame,
        plot_name: str = "",
        figsize: Tuple[int, int] = (15, 10),
    ) -> None:
        """Plot correlation matrices for each numeric feature.

        Args:
            data: Data to perform analysis on.
            plot_name: The name for the plot file. If empty then display plot
                using plt.show()
            figsize: A tuple containing the plot dimensions.
        """
        if len(self.numeric_cols) == 0:
            correlation_matrix = data[self.numeric_cols].corr()
            plt.figure(figsize=figsize)
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
            plt.title("Correlation Matrix")
            self.show_plot(plot_name)

    def violin_plots(
        self,
        data: pd.DataFrame,
        plot_name: str = "",
        figsize: Tuple[int, int] = (15, 10),
    ):
        """Plot distribution of numeric features for each target label.

        Args:
            data: Data to perform analysis on.
            plot_name: The name for the plot file. If empty then display plot
                using plt.show()
            figsize: A tuple containing the plot dimensions.
        """
        num_plots = len(self.numeric_cols)
        cols = math.ceil(math.sqrt(num_plots))
        rows = math.ceil(num_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if len(axes) != 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        for i, column in enumerate(self.numeric_cols):
            sns.violinplot(x=self.target_col[0], y=column, data=data, ax=axes[i])
            axes[i].set_title(f"Violin plot: {column}")
            axes[i].tick_params(axis="x", rotation=45)

        # Hide any remaining empty subplots
        for i in range(num_plots, rows * cols):
            fig.delaxes(axes[i])

        self.show_plot(plot_name)

    def category_heatmap(
        self,
        data: pd.DataFrame,
        plot_name: str = "",
        figsize: Tuple[int, int] = (15, 10),
    ) -> None:
        """Plot a target label heatmap for each categorical feature."""
        num_plots = len(self.categoric_cols)
        cols = math.ceil(math.sqrt(num_plots))
        rows = math.ceil(num_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        # Ensure axes is always iterable
        if isinstance(axes, np.ndarray):
            axes = axes.ravel()
        else:
            axes = [axes]

        for i, column in enumerate(self.categoric_cols):
            if column not in data.columns or self.target_col[0] not in data.columns:
                continue

            cross_table_plot = pd.crosstab(
                data[column].dropna(), data[self.target_col[0]].dropna()
            )

            sns.heatmap(
                cross_table_plot,
                annot=True,
                cmap="YlGnBu",
                fmt="d",
                cbar_kws={"label": "Count"},
                ax=axes[i],
            )
            axes[i].set_title(f"Categoric features - Heatmap: {column}")
            axes[i].tick_params(axis="x", rotation=45)

        # Hide unused subplots
        for i in range(num_plots, rows * cols):
            fig.delaxes(axes[i])

        plt.tight_layout()
        self.show_plot(plot_name)

    def scatter_plot(
        self,
        data: pd.DataFrame,
        plot_name: str = "",
    ) -> None:
        """Plot scatter plots between each of the numeric features.

        Args:
            data: Data to perform analysis on.
            plot_name: The name for the plot file. If empty then display plot
                using plt.show()
        """
        sns.pairplot(data[self.numeric_cols])
        plt.suptitle("Pair Plot of Numeric Variables", y=1.02)
        self.show_plot(plot_name)

    def perform_eda(self, data: pd.DataFrame, plot_name="") -> None:
        """Perform all exploratory data analysis methods.

        Args:
            data: Data to perform analysis on.
            plot_name: The name for the plot file. If empty then display plot
                using plt.show()
        """
        self.correlation_matrix(
            data, self.update_plotname(plot_name, "_Correlation_matrix.png", ".png")
        )
        self.violin_plots(
            data, self.update_plotname(plot_name, "_Violin_plot.png", ".png")
        )
        self.category_heatmap(
            data, self.update_plotname(plot_name, "_Category_map.png", ".png")
        )
        self.scatter_plot(
            data, self.update_plotname(plot_name, "_Scatter_plot.png", ".png")
        )
