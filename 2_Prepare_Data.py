import streamlit as st
from pathlib import Path
import base64


from core.preprocessing import create_preparation_config


st.set_page_config(page_title="Prepare Data", page_icon="⚙️", layout="wide")
# =========================================================
# SAME BACKGROUND IMAGE AS HOME PAGE
# =========================================================

image_path = Path("assets/breast_cancer_bg.png")

if image_path.exists():

    image_base64 = base64.b64encode(
        image_path.read_bytes()
    ).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image:
                linear-gradient(
                    rgba(255, 245, 250, 0.78),
                    rgba(255, 245, 250, 0.78)
                ),
                url("data:image/png;base64,{image_base64}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Main title */
        h1 {{
            color: #7f123f !important;
            font-weight: 800 !important;
        }}

        /* Sub headings */
        h2, h3 {{
            color: #5c0a2d !important;
            font-weight: 700 !important;
        }}

        /* Normal text */
        p {{
            color: #111827 !important;
            font-size: 17px !important;
            font-weight: 500 !important;
        }}

        /* Labels */
        label {{
            color: #111827 !important;
            font-weight: 600 !important;
        }}

        /* Info / warning / success boxes */
        .stAlert {{
            background-color: rgba(255, 255, 255, 0.95) !important;
            border-radius: 15px !important;
        }}
                /* DARK TEXT - ALL CONTENT */

        body {{
            color: #111827 !important;
        }}

        p {{
            color: #111827 !important;
        }}

        label {{
            color: #111827 !important;
            font-weight: 600 !important;
        }}

        /* Selectbox / Multiselect / Checkbox text */
        .stSelectbox label,
        .stMultiSelect label,
        .stCheckbox label {{
            color: #111827 !important;
        }}

        /* Caption */
        [data-testid="stCaptionContainer"] {{
            color: #1f2937 !important;
        }}

        /* Markdown text */
        [data-testid="stMarkdownContainer"] {{
            color: #111827 !important;
        }}

        [data-testid="stMarkdownContainer"] p {{
            color: #111827 !important;
        }}

        /* Headings */
        h1 {{
            color: #7f123f !important;
        }}

        h2, h3 {{
            color: #5c0a2d !important;
        }}

        /* Data table text */
        [data-testid="stDataFrame"] {{
            color: #111827 !important;
        }}

        /* Alert text */
        .stAlert p {{
            color: #111827 !important;
        }}
                [data-testid="stMetricLabel"] {{
            color: #111827 !important;
            font-weight: 800 !important;
        }}

        [data-testid="stMetricValue"] {{
            color: #5c0a2d !important;
            font-weight: 900 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
st.title("Prepare data")

if "dataset" not in st.session_state:
    st.warning("Upload a dataset first from the Upload Dataset page.")
    st.stop()

dataframe = st.session_state["dataset"]

st.subheader("1. Choose the target column")

target_column = st.selectbox(
    "Which column should the model predict?",
    options=list(dataframe.columns),
    index=len(dataframe.columns) - 1,
)

target_values = dataframe[target_column].dropna()

if target_values.nunique() < 2:
    st.error(
        "The target column needs at least two different class values."
    )
    st.stop()

st.caption(
    f"Target classes found: {target_values.nunique()}"
)

st.subheader("2. Select input features")

available_features = [
    column
    for column in dataframe.columns
    if column != target_column
]

feature_columns = st.multiselect(
    "Choose columns used by the algorithm",
    options=available_features,
    default=available_features,
)

if not feature_columns:
    st.warning("Select at least one feature column.")
    st.stop()

st.subheader("3. Cleaning and transformation")

scale_numeric = st.checkbox(
    "Scale numeric features",
    value=True,
    help=(
        "Recommended for KNN, SVM, K-Means, DBSCAN, "
        "and hierarchical clustering."
    ),
)

missing_values = dataframe[feature_columns].isna().sum()
missing_values = missing_values[missing_values > 0]

left, right = st.columns(2)

with left:
    st.markdown("### Selected features")
    feature_table = dataframe[feature_columns].dtypes.astype(str)
    st.dataframe(
        feature_table.rename("Data type"),
        use_container_width=True,
    )

with right:
    st.markdown("### Missing values")

    if missing_values.empty:
        st.success("No missing values in selected features.")
    else:
        st.dataframe(
            missing_values.rename("Missing values"),
            use_container_width=True,
        )
        st.caption(
            "Numeric missing values will use the median. "
            "Text/categorical missing values will use the most frequent value."
        )

if st.button("Save preparation settings", type="primary"):
    st.session_state["preparation_config"] = (
        create_preparation_config(
            dataframe=dataframe,
            target_column=target_column,
            feature_columns=feature_columns,
            scale_numeric=scale_numeric,
        )
    )

    st.success("Preparation settings saved successfully.")

    st.info(
        "Next: open Classification for ID3, J48, KNN, "
        "or Naive Bayes."
    )