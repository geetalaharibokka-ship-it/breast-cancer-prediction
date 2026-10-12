import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_column_groups(dataframe: pd.DataFrame, feature_columns: list[str]):
    """Separate selected features into numeric and categorical groups."""
    selected = dataframe[feature_columns]

    numeric_columns = list(
        selected.select_dtypes(include="number").columns
    )

    categorical_columns = [
        column
        for column in feature_columns
        if column not in numeric_columns
    ]

    return numeric_columns, categorical_columns


def build_preprocessor(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Create preprocessing steps for selected dataset features."""
    numeric_columns, categorical_columns = get_column_groups(
        dataframe, feature_columns
    )

    numeric_steps = [
        ("fill_missing", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_steps = [
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        (
            "encode",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]

    transformers = []

    if numeric_columns:
        transformers.append(
            ("numeric", Pipeline(numeric_steps), numeric_columns)
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(categorical_steps),
                categorical_columns,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def create_preparation_config(
    dataframe: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    scale_numeric: bool,
) -> dict:
    """Save all selected preparation settings in one dictionary."""
    numeric_columns, categorical_columns = get_column_groups(
        dataframe, feature_columns
    )

    return {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "scale_numeric": scale_numeric,
    }