import pandas as pd
import streamlit as st


def show_dataset_preview(dataframe: pd.DataFrame, file_name: str) -> None:
    """Show uploaded dataset summary, preview, and column information."""
    st.subheader("Dataset overview")
    st.caption(f"Loaded file: **{file_name}**")

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric("Rows", len(dataframe))
    metric_2.metric("Columns", len(dataframe.columns))
    metric_3.metric("Missing values", int(dataframe.isna().sum().sum()))
    metric_4.metric(
        "Numeric columns",
        len(dataframe.select_dtypes(include="number").columns),
    )

    st.markdown("### Dataset preview")
    st.dataframe(dataframe.head(15), use_container_width=True, height=400)

    st.markdown("### Column information")

    column_info = pd.DataFrame(
        {
            "Column": dataframe.columns,
            "Data type": dataframe.dtypes.astype(str).values,
            "Missing values": dataframe.isna().sum().values,
            "Unique values": dataframe.nunique().values,
        }
    )

    st.dataframe(column_info, use_container_width=True, hide_index=True)