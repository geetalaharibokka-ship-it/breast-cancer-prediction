from io import BytesIO, StringIO

import pandas as pd
from scipy.io import arff


SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".arff"}


def load_dataset(uploaded_file) -> pd.DataFrame:
    """Load an uploaded CSV, TSV, Excel, JSON, or ARFF file."""
    filename = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if filename.endswith(".csv"):
        dataframe = pd.read_csv(BytesIO(raw))

    elif filename.endswith((".tsv", ".txt")):
        dataframe = pd.read_csv(BytesIO(raw), sep="\t")

    elif filename.endswith((".xlsx", ".xls")):
        dataframe = pd.read_excel(BytesIO(raw))

    elif filename.endswith(".json"):
        dataframe = pd.read_json(BytesIO(raw))

    elif filename.endswith(".arff"):
        records, _ = arff.loadarff(StringIO(raw.decode("utf-8-sig")))
        dataframe = pd.DataFrame(records)

    else:
        raise ValueError(
            "Unsupported file. Upload CSV, TSV, Excel, JSON, or ARFF."
        )

    # Convert ARFF byte values into normal text.
    for column in dataframe.columns:
        if dataframe[column].dtype == object:
            dataframe[column] = dataframe[column].apply(
                lambda value: value.decode("utf-8")
                if isinstance(value, bytes)
                else value
            )

    if dataframe.empty:
        raise ValueError("The uploaded dataset has no rows.")

    return dataframe


def get_dataset_profile(dataframe: pd.DataFrame) -> dict:
    """Create summary information for the upload page."""
    return {
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "missing_values": int(dataframe.isna().sum().sum()),
        "numeric_columns": len(
            dataframe.select_dtypes(include="number").columns
        ),
        "categorical_columns": len(
            dataframe.select_dtypes(exclude="number").columns
        ),
    }