import streamlit as st
from pathlib import Path
import base64

from core.classification import train_classifier


st.set_page_config(
    page_title="Classification Lab",
    page_icon="🧠",
    layout="wide",
)
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
                /* Classification sliders - dark text */
        [data-testid="stSlider"] label {{
            color: #111827 !important;
            font-weight: 700 !important;
        }}

        [data-testid="stSlider"] p {{
            color: #111827 !important;
            font-weight: 700 !important;
        }}

        [data-testid="stSlider"] [data-testid="stThumbValue"] {{
            color: #111827 !important;
            font-weight: 800 !important;
        }}

        [data-testid="stSlider"] div {{
            color: #111827 !important;
        }}

        /* Caption text */
        .stCaption {{
            color: #1f2937 !important;
        }}

        /* Info / warning / success boxes */
        .stAlert {{
            background-color: rgba(255, 255, 255, 0.95) !important;
            border-radius: 15px !important;
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
st.title("Classification lab")

if "dataset" not in st.session_state:
    st.warning("Upload a dataset first.")
    st.stop()

if "preparation_config" not in st.session_state:
    st.warning(
        "Open Prepare Data, choose a target column, "
        "and save the preparation settings first."
    )
    st.stop()

dataframe = st.session_state["dataset"]
config = st.session_state["preparation_config"]

st.caption(
    f"Target column: **{config['target_column']}** · "
    f"Features selected: **{len(config['feature_columns'])}**"
)

left, right = st.columns(2)

with left:
    algorithm = st.selectbox(
        "Classification algorithm",
        [
            "ID3 decision tree",
            "J48 / C4.5-style tree",
            "K-nearest neighbours",
            "Naive Bayes",
            "Random Forest",
        ],
    )

    evaluation_method = st.radio(
        "Evaluation method",
        [
            "Holdout test set",
            "Stratified cross-validation",
        ],
    )

with right:
    neighbours = st.slider(
        "K neighbours",
        min_value=1,
        max_value=25,
        value=5,
        disabled=algorithm != "K-nearest neighbours",
    )

    max_depth = st.slider(
        "Maximum tree depth",
        min_value=2,
        max_value=15,
        value=6,
        disabled=algorithm
        not in {"ID3 decision tree", "J48 / C4.5-style tree"},
    )

    test_size = st.slider(
        "Test-set percentage",
        min_value=15,
        max_value=40,
        value=20,
        disabled=evaluation_method
        != "Holdout test set",
    )

    folds = st.slider(
        "Cross-validation folds",
        min_value=2,
        max_value=10,
        value=5,
        disabled=evaluation_method
        != "Stratified cross-validation",
    )

if st.button("Run classification", type="primary"):
    try:
        with st.spinner("Training model and calculating results..."):
            result = train_classifier(
                dataframe=dataframe,
                preparation_config=config,
                algorithm=algorithm,
                evaluation_method=evaluation_method,
                test_size=test_size / 100,
                folds=folds,
                neighbours=neighbours,
                max_depth=max_depth,
            )

        st.session_state["classification_result"] = result

        st.success("Classification completed successfully.")

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)

        metric_1.metric("Accuracy", f"{result['accuracy']:.2%}")
        metric_2.metric("Weighted F1", f"{result['f1_score']:.3f}")
        metric_3.metric("Kappa", f"{result['kappa']:.3f}")
        metric_4.metric(
            "Evaluated rows",
            result["evaluated_rows"],
        )

        st.info(
            "Open the Results page to see ROC curves, "
            "confusion matrix, detailed reports, and decision trees."
        )

    except Exception as error:
        st.error(f"Classification could not run: {error}")