import streamlit as st
from pathlib import Path
import base64
from io import BytesIO

from components.dataset_preview import show_dataset_preview
from core.dataset_loader import load_dataset


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Upload Dataset",
    page_icon="📁",
    layout="wide"
)


# =========================================================
# BACKGROUND IMAGE
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
                    rgba(255, 245, 250, 0.70),
                    rgba(255, 245, 250, 0.70)
                ),
                url("data:image/png;base64,{image_base64}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        h1 {{
            color: #7f123f !important;
            font-weight: 800 !important;
        }}

        h2, h3 {{
            color: #5c0a2d !important;
            font-weight: 700 !important;
        }}

        p {{
            color: #111827 !important;
            font-size: 17px !important;
            font-weight: 500 !important;
        }}

        label {{
            color: #111827 !important;
            font-weight: 600 !important;
        }}

        [data-testid="stFileUploader"] * {{
            color: #111827 !important;
        }}

        .stAlert {{
            background-color: rgba(255, 255, 255, 0.95) !important;
            border-radius: 15px !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# TITLE
# =========================================================

st.title("Upload dataset")

st.write(
    "Upload a structured dataset to begin machine-learning analysis."
)


# =========================================================
# EXISTING DATASET
# =========================================================

existing_dataset_path = Path(
    "assets/breast_cancer_wisconsin_100.arff"
)


# Load automatically ONLY if dataset is not already in session
if "dataset" not in st.session_state:

    if existing_dataset_path.exists():

        try:

            raw_data = existing_dataset_path.read_bytes()

            file_object = BytesIO(raw_data)

            file_object.name = (
                existing_dataset_path.name
            )

            dataframe = load_dataset(file_object)

            st.session_state["dataset"] = dataframe

            st.session_state["dataset_name"] = (
                existing_dataset_path.name
            )

        except Exception as error:

            st.error(
                f"Could not load existing dataset: {error}"
            )


# =========================================================
# UPLOAD NEW DATASET
# =========================================================

uploaded_file = st.file_uploader(
    "Choose a file",
    type=[
        "csv",
        "tsv",
        "txt",
        "xlsx",
        "xls",
        "json",
        "arff"
    ],
    help="Supported formats: CSV, TSV, Excel, JSON, and ARFF."
)


# =========================================================
# NEW FILE UPLOADED
# =========================================================

if uploaded_file is not None:

    try:

        dataframe = load_dataset(uploaded_file)

        st.session_state["dataset"] = dataframe

        st.session_state["dataset_name"] = (
            uploaded_file.name
        )

        st.success(
            f"Dataset loaded successfully: "
            f"{uploaded_file.name}"
        )

        show_dataset_preview(
            dataframe,
            uploaded_file.name
        )

        st.info(
            "Next: open **Prepare Data** from the sidebar "
            "to select the target column and clean the dataset."
        )

    except Exception as error:

        st.error(
            f"Could not load the dataset: {error}"
        )


# =========================================================
# EXISTING DATASET DISPLAY
# =========================================================

elif "dataset" in st.session_state:

    dataframe = st.session_state["dataset"]

    dataset_name = st.session_state.get(
        "dataset_name",
        "Existing Dataset"
    )

    st.success(
        f"Existing dataset ready: {dataset_name}"
    )

    show_dataset_preview(
        dataframe,
        dataset_name
    )

    st.info(
        "Dataset is already loaded. "
        "You can continue to Prepare Data."
    )
