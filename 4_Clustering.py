import streamlit as st
from pathlib import Path
import base64

from core.clustering import train_clusterer


st.set_page_config(
    page_title="Clustering Lab",
    page_icon="🔷",
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

        /* Caption */
        .stCaption {{
            color: #1f2937 !important;
        }}

        /* Info / warning / success boxes */
        .stAlert {{
            background-color: rgba(255, 255, 255, 0.95) !important;
            border-radius: 15px !important;
        }}
                /* DARK CLUSTERING CONTROLS */

        .stSlider label {{
            color: #111827 !important;
            font-weight: 800 !important;
            font-size: 16px !important;
        }}

        .stSlider p {{
            color: #111827 !important;
            font-weight: 700 !important;
        }}

        /* Slider numbers */
        [data-testid="stSlider"] {{
            color: #111827 !important;
        }}

        [data-testid="stSlider"] * {{
            color: #111827 !important;
        }}

        /* Selectbox label */
        .stSelectbox label {{
            color: #111827 !important;
            font-weight: 800 !important;
        }}

        /* Multiselect label */
        .stMultiSelect label {{
            color: #111827 !important;
            font-weight: 800 !important;
        }}

        /* Metrics */
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
st.title("Clustering lab")

if "dataset" not in st.session_state:
    st.warning("Upload a dataset first.")
    st.stop()

dataframe = st.session_state["dataset"]

numeric_columns = list(
    dataframe.select_dtypes(include="number").columns
)

if not numeric_columns:
    st.error(
        "This dataset has no numeric columns. "
        "Clustering requires numeric data."
    )
    st.stop()

selected_features = st.multiselect(
    "Choose numeric features used for clustering",
    options=numeric_columns,
    default=numeric_columns,
)

algorithm = st.selectbox(
    "Clustering algorithm",
    [
        "K-Means",
        "Hierarchical clustering",
        "DBSCAN",
    ],
)

left, right = st.columns(2)

with left:
    cluster_count = st.slider(
        "Number of clusters",
        min_value=2,
        max_value=min(12, max(2, len(dataframe) - 1)),
        value=3,
        disabled=algorithm == "DBSCAN",
    )

with right:
    epsilon = st.slider(
        "DBSCAN neighbourhood radius",
        min_value=0.1,
        max_value=3.0,
        value=0.5,
        step=0.1,
        disabled=algorithm != "DBSCAN",
    )

    minimum_samples = st.slider(
        "DBSCAN minimum samples",
        min_value=2,
        max_value=min(20, max(2, len(dataframe) - 1)),
        value=5,
        disabled=algorithm != "DBSCAN",
    )

if st.button("Run clustering", type="primary"):
    try:
        with st.spinner("Finding groups in your dataset..."):
            result = train_clusterer(
                dataframe=dataframe,
                feature_columns=selected_features,
                algorithm=algorithm,
                cluster_count=cluster_count,
                epsilon=epsilon,
                minimum_samples=minimum_samples,
            )

        st.session_state["clustering_result"] = result

        st.success("Clustering completed successfully.")

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)

        metric_1.metric(
            "Clusters found",
            result["cluster_count"],
        )

        metric_2.metric(
            "Silhouette score",
            (
                f"{result['silhouette_score']:.3f}"
                if result["silhouette_score"] is not None
                else "Not available"
            ),
        )

        metric_3.metric(
            "Noise points",
            result["noise_points"],
        )

        metric_4.metric(
            "Features used",
            len(result["feature_columns"]),
        )

        st.markdown("### Cluster sizes")
        st.dataframe(
            result["cluster_sizes"],
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "Open Results to see the colourful PCA cluster map "
            "and download the dataset with cluster labels."
        )

    except Exception as error:
        st.error(f"Clustering could not run: {error}")