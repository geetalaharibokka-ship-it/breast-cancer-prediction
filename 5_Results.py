import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import base64

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import label_binarize
from sklearn.tree import plot_tree, export_text


st.set_page_config(
    page_title="Results Dashboard",
    page_icon="📊",
    layout="wide",
)
# ============================================================
# SAME BACKGROUND IMAGE AS HOME PAGE
# ============================================================

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

        /* Headings */
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

        /* Captions */
        .stCaption {{
            color: #1f2937 !important;
            font-weight: 500 !important;
        }}

        /* Labels */
        label {{
            color: #111827 !important;
            font-weight: 600 !important;
        }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: rgba(255, 220, 235, 0.95) !important;
        }}

        [data-testid="stSidebar"] * {{
            color: #5c0a2d !important;
        }}

        /* Alert boxes */
        .stAlert {{
            background-color: rgba(255, 255, 255, 0.95) !important;
            border-radius: 15px !important;
        }}

        /* Dataframes / tables */
        [data-testid="stDataFrame"] {{
            background-color: rgba(255, 255, 255, 0.92) !important;
            border-radius: 12px !important;
        }}
                /* Dark text for result values */
        [data-testid="stMetricLabel"] {{
            color: #111827 !important;
            font-weight: 800 !important;
        }}

        [data-testid="stMetricValue"] {{
            color: #5c0a2d !important;
            font-weight: 900 !important;
        }}

        [data-testid="stWidgetLabel"] {{
            color: #111827 !important;
            font-weight: 700 !important;
        }}

        [data-testid="stSlider"] label {{
            color: #111827 !important;
            font-weight: 700 !important;
        }}

        [data-testid="stRadio"] label {{
            color: #111827 !important;
            font-weight: 700 !important;
        }}

        [data-baseweb="select"] * {{
            color: #111827 !important;
        }}

        .stMarkdown p {{
            color: #111827 !important;
        }}

        [data-testid="stDataFrame"] * {{
            color: #111827 !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SESSION STATE
# ============================================================

dataset = st.session_state.get("dataset")
preparation_config = st.session_state.get("preparation_config")
classification_result = st.session_state.get(
    "classification_result"
)
clustering_result = st.session_state.get(
    "clustering_result"
)


# ============================================================
# HEADER
# ============================================================

st.title("📊 Results Dashboard")

st.caption(
    "WEKA-style machine-learning analysis, "
    "decision trees, ROC curves and clustering results."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Results")

section = st.sidebar.radio(
    "Select analysis",
    [
        "Overview",
        "Classification",
        "Decision Tree",
        "ROC Curve",
        "WEKA Report",
        "Clustering",
        "Dataset",
        "Exports",
    ],
)


# ============================================================
# OVERVIEW
# ============================================================

if section == "Overview":

    st.header("📊 Results Overview")

    c1, c2, c3, c4 = st.columns(4)

    if classification_result is not None:

        accuracy = classification_result.get(
            "accuracy"
        )

        f1 = classification_result.get(
            "f1_score"
        )

        kappa = classification_result.get(
            "kappa"
        )

        c1.metric(
            "Accuracy",
            f"{accuracy:.2%}"
            if accuracy is not None
            else "—",
        )

        c2.metric(
            "Weighted F1",
            f"{f1:.3f}"
            if f1 is not None
            else "—",
        )

        c3.metric(
            "Kappa",
            f"{kappa:.3f}"
            if kappa is not None
            else "—",
        )

    else:

        c1.metric("Accuracy", "—")
        c2.metric("Weighted F1", "—")
        c3.metric("Kappa", "—")

    if clustering_result is not None:

        clusters = clustering_result.get(
            "cluster_count"
        )

        c4.metric(
            "Clusters",
            clusters
            if clusters is not None
            else "—",
        )

    else:

        c4.metric("Clusters", "—")

    st.divider()

    st.subheader("🚦 Analysis Status")

    s1, s2, s3 = st.columns(3)

    if dataset is not None:
        s1.success("✅ Dataset loaded")
    else:
        s1.warning("⚠️ Dataset not loaded")

    if classification_result is not None:
        s2.success("✅ Classification completed")
    else:
        s2.info("Classification not run")

    if clustering_result is not None:
        s3.success("✅ Clustering completed")
    else:
        s3.info("Clustering not run")

    if dataset is not None:

        st.divider()

        st.subheader("📁 Dataset Summary")

        d1, d2, d3 = st.columns(3)

        d1.metric(
            "Rows",
            dataset.shape[0],
        )

        d2.metric(
            "Columns",
            dataset.shape[1],
        )

        d3.metric(
            "Missing Values",
            int(
                dataset.isnull().sum().sum()
            ),
        )


# ============================================================
# CLASSIFICATION
# ============================================================

elif section == "Classification":

    st.header("🧠 Classification Results")

    if classification_result is None:

        st.warning(
            "No classification results found."
        )

        st.info(
            "Go to Classification Lab and "
            "click Run classification."
        )

        st.stop()

    algorithm = classification_result.get(
        "algorithm",
        "Unknown",
    )

    evaluation_note = classification_result.get(
        "evaluation_note",
        "",
    )

    st.subheader(
        f"Algorithm: {algorithm}"
    )

    if evaluation_note:
        st.caption(evaluation_note)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{classification_result['accuracy']:.2%}",
    )

    c2.metric(
        "Precision",
        f"{classification_result['precision']:.3f}",
    )

    c3.metric(
        "Recall",
        f"{classification_result['recall']:.3f}",
    )

    c4.metric(
        "F1 Score",
        f"{classification_result['f1_score']:.3f}",
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "Kappa",
        f"{classification_result['kappa']:.3f}",
    )

    c6.metric(
        "Evaluated Rows",
        classification_result["evaluated_rows"],
    )

    c7.metric(
        "Correct",
        classification_result["correct_rows"],
    )

    c8.metric(
        "Incorrect",
        (
            classification_result["evaluated_rows"]
            - classification_result["correct_rows"]
        ),
    )

    st.divider()

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    actual = np.asarray(
        classification_result["actual"]
    )

    predicted = np.asarray(
        classification_result["predicted"]
    )

    class_names = classification_result.get(
        "class_names"
    )

    cm = confusion_matrix(
        actual,
        predicted,
    )

    st.subheader("🧩 Confusion Matrix")

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    image = ax.imshow(cm)

    ax.set_title(
        "Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "Actual Class"
    )

    if class_names is not None:

        ax.set_xticks(
            range(len(class_names))
        )

        ax.set_yticks(
            range(len(class_names))
        )

        ax.set_xticklabels(
            class_names,
            rotation=45,
            ha="right",
        )

        ax.set_yticklabels(
            class_names
        )

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    st.pyplot(fig)

    st.subheader(
        "📋 Detailed Classification Report"
    )

    report = classification_report(
        actual,
        predicted,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True,
    )


# ============================================================
# DECISION TREE
# ============================================================

elif section == "Decision Tree":

    st.header("🌳 Decision Tree Visualization")

    if classification_result is None:

        st.warning(
            "Run classification first."
        )

        st.stop()

    algorithm = classification_result.get(
        "algorithm",
        "",
    )

    model = classification_result.get(
        "model"
    )

    if model is None:

        st.error(
            "The trained model was not found."
        )

        st.stop()

    if algorithm not in [
        "ID3 decision tree",
        "J48 / C4.5-style tree",
        "Random Forest",
    ]:

        st.info(
            f"The selected algorithm is **{algorithm}**. "
            "A decision-tree visualization is available "
            "for ID3, J48 and Random Forest."
        )

        st.stop()

    # Get classifier from pipeline
    try:

        classifier = model.named_steps[
            "classifier"
        ]

    except Exception:

        st.error(
            "Could not access the classifier "
            "inside the model pipeline."
        )

        st.stop()

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    if algorithm == "Random Forest":

        st.subheader(
            "🌲 Random Forest"
        )

        st.info(
            "Random Forest contains many decision trees. "
            "The visualization below shows the first tree."
        )

        try:

            tree_model = classifier.estimators_[0]

            st.write(
                f"Number of trees: "
                f"**{len(classifier.estimators_)}**"
            )

            st.write(
                f"Selected tree depth: "
                f"**{tree_model.get_depth()}**"
            )

        except Exception as error:

            st.error(
                f"Could not access Random Forest tree: {error}"
            )

            st.stop()

    else:

        tree_model = classifier

        st.subheader(
            f"🌳 {algorithm}"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Tree Depth",
            tree_model.get_depth(),
        )

        c2.metric(
            "Number of Nodes",
            tree_model.tree_.node_count,
        )

        c3.metric(
            "Leaves",
            tree_model.get_n_leaves(),
        )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = None

    try:

        preprocessor = model.named_steps[
            "preprocessor"
        ]

        feature_names = (
            preprocessor.get_feature_names_out()
        )

    except Exception:

        feature_names = None

    if feature_names is None:

        feature_names = [
            f"Feature {i + 1}"
            for i in range(
                tree_model.n_features_in_
            )
        ]

    # --------------------------------------------------------
    # Tree display controls
    # --------------------------------------------------------

    max_display_depth = st.slider(
        "Tree display depth",
        min_value=1,
        max_value=min(
            10,
            tree_model.get_depth(),
        ),
        value=min(
            4,
            tree_model.get_depth(),
        ),
    )

    st.subheader(
        "🌳 Visual Tree"
    )

    fig, ax = plt.subplots(
        figsize=(24, 14)
    )

    plot_tree(
        tree_model,
        ax=ax,
        feature_names=feature_names,
        class_names=(
            classification_result.get(
                "class_names"
            )
        ),
        filled=True,
        rounded=True,
        max_depth=max_display_depth,
        fontsize=8,
    )

    st.pyplot(fig)

    # --------------------------------------------------------
    # Tree rules
    # --------------------------------------------------------

    st.subheader(
        "📜 Decision Rules"
    )

    try:

        rules = export_text(
            tree_model,
            feature_names=list(
                feature_names
            ),
        )

        st.code(
            rules,
            language="text",
        )

    except Exception as error:

        st.warning(
            f"Could not generate tree rules: {error}"
        )


# ============================================================
# ROC CURVE
# ============================================================

elif section == "ROC Curve":

    st.header("📈 ROC Curve & AUC")

    if classification_result is None:

        st.warning(
            "Run classification first."
        )

        st.stop()

    actual = np.asarray(
        classification_result["actual"]
    )

    probabilities = np.asarray(
        classification_result["probabilities"]
    )

    class_names = classification_result.get(
        "class_names"
    )

    if probabilities.ndim != 2:

        st.error(
            "Prediction probabilities are not "
            "available in the expected format."
        )

        st.stop()

    number_of_classes = probabilities.shape[1]

    st.write(
        f"Detected **{number_of_classes} classes**."
    )

    # --------------------------------------------------------
    # Binary ROC
    # --------------------------------------------------------

    if number_of_classes == 2:

        positive_probability = probabilities[
            :, 1
        ]

        fpr, tpr, _ = roc_curve(
            actual,
            positive_probability,
        )

        roc_auc = auc(
            fpr,
            tpr,
        )

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        ax.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc:.4f}",
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Random classifier",
        )

        ax.set_xlabel(
            "False Positive Rate"
        )

        ax.set_ylabel(
            "True Positive Rate"
        )

        ax.set_title(
            "Receiver Operating Characteristic"
        )

        ax.legend()

        st.pyplot(fig)

        st.metric(
            "ROC-AUC",
            f"{roc_auc:.4f}",
        )

    # --------------------------------------------------------
    # Multiclass ROC
    # --------------------------------------------------------

    else:

        binary_actual = label_binarize(
            actual,
            classes=range(
                number_of_classes
            ),
        )

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        auc_values = []

        for i in range(
            number_of_classes
        ):

            try:

                fpr, tpr, _ = roc_curve(
                    binary_actual[:, i],
                    probabilities[:, i],
                )

                roc_auc = auc(
                    fpr,
                    tpr,
                )

                auc_values.append(
                    roc_auc
                )

                label = (
                    class_names[i]
                    if class_names is not None
                    and i < len(class_names)
                    else f"Class {i}"
                )

                ax.plot(
                    fpr,
                    tpr,
                    label=(
                        f"{label} "
                        f"(AUC={roc_auc:.3f})"
                    ),
                )

            except ValueError:
                pass

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Random classifier",
        )

        ax.set_xlabel(
            "False Positive Rate"
        )

        ax.set_ylabel(
            "True Positive Rate"
        )

        ax.set_title(
            "Multiclass ROC Curve"
        )

        ax.legend()

        st.pyplot(fig)

        if auc_values:

            st.metric(
                "Mean AUC",
                f"{np.mean(auc_values):.4f}",
            )


# ============================================================
# WEKA REPORT
# ============================================================

elif section == "WEKA Report":

    st.header("📄 WEKA-Style Classification Result")

    if classification_result is None:

        st.warning(
            "Run a classification model first."
        )

        st.stop()

    actual = np.asarray(
        classification_result["actual"]
    )

    predicted = np.asarray(
        classification_result["predicted"]
    )

    probabilities = np.asarray(
        classification_result["probabilities"]
    )

    class_names = classification_result.get(
        "class_names"
    )

    correct = int(
        np.sum(actual == predicted)
    )

    incorrect = int(
        len(actual) - correct
    )

    accuracy = (
        correct / len(actual)
        if len(actual) > 0
        else 0
    )

    # --------------------------------------------------------
    # Error metrics
    # --------------------------------------------------------

    try:

        number_of_classes = probabilities.shape[1]

        one_hot_actual = label_binarize(
            actual,
            classes=range(
                number_of_classes
            ),
        )

        mae = mean_absolute_error(
            one_hot_actual,
            probabilities,
        )

        rmse = np.sqrt(
            mean_squared_error(
                one_hot_actual,
                probabilities,
            )
        )

    except Exception:

        mae = None
        rmse = None

    kappa = classification_result.get(
        "kappa"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    st.subheader(
        "=== Summary ==="
    )

    summary = f"""
Correctly Classified Instances    {correct}
Incorrectly Classified Instances  {incorrect}
Accuracy                          {accuracy * 100:.2f}%
Kappa statistic                   {kappa:.4f}
Mean absolute error               {
    f"{mae:.4f}" if mae is not None else "N/A"
}
Root mean squared error           {
    f"{rmse:.4f}" if rmse is not None else "N/A"
}
Total Instances                   {len(actual)}
"""

    st.code(
        summary,
        language="text",
    )

    # --------------------------------------------------------
    # Detailed Accuracy
    # --------------------------------------------------------

    st.subheader(
        "=== Detailed Accuracy By Class ==="
    )

    report = classification_report(
        actual,
        predicted,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    st.subheader(
        "=== Confusion Matrix ==="
    )

    cm = confusion_matrix(
        actual,
        predicted,
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    )

    st.dataframe(
        cm_df,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Full report
    # --------------------------------------------------------

    st.subheader(
        "=== Complete Report ==="
    )

    st.code(
        classification_report(
            actual,
            predicted,
            target_names=class_names,
            zero_division=0,
        ),
        language="text",
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    st.download_button(
        "📥 Download WEKA-style Report",
        data=summary
        + "\n\n"
        + classification_report(
            actual,
            predicted,
            target_names=class_names,
            zero_division=0,
        ),
        file_name="weka_style_result.txt",
        mime="text/plain",
    )


# ============================================================
# CLUSTERING
# ============================================================

elif section == "Clustering":

    st.header("🔷 Clustering Results")

    if clustering_result is None:

        st.warning(
            "Run clustering first."
        )

        st.stop()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Clusters",
        clustering_result.get(
            "cluster_count",
            "—",
        ),
    )

    silhouette = clustering_result.get(
        "silhouette_score"
    )

    c2.metric(
        "Silhouette Score",
        (
            f"{silhouette:.3f}"
            if silhouette is not None
            else "N/A"
        ),
    )

    c3.metric(
        "Noise Points",
        clustering_result.get(
            "noise_points",
            "—",
        ),
    )

    features = clustering_result.get(
        "feature_columns"
    )

    c4.metric(
        "Features",
        len(features)
        if features is not None
        else "—",
    )

    st.divider()

    cluster_sizes = clustering_result.get(
        "cluster_sizes"
    )

    if cluster_sizes is not None:

        st.subheader(
            "📊 Cluster Sizes"
        )

        st.dataframe(
            cluster_sizes,
            use_container_width=True,
            hide_index=True,
        )

        try:
            st.bar_chart(
                cluster_sizes
            )
        except Exception:
            pass


# ============================================================
# DATASET
# ============================================================

elif section == "Dataset":

    st.header("📁 Dataset Information")

    if dataset is None:

        st.warning(
            "No dataset uploaded."
        )

    else:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Rows",
            dataset.shape[0],
        )

        c2.metric(
            "Columns",
            dataset.shape[1],
        )

        c3.metric(
            "Missing Values",
            int(
                dataset.isnull().sum().sum()
            ),
        )

        st.subheader(
            "Dataset Preview"
        )

        st.dataframe(
            dataset,
            use_container_width=True,
        )

        st.subheader(
            "Column Information"
        )

        information = pd.DataFrame(
            {
                "Column": dataset.columns,
                "Type": [
                    str(dtype)
                    for dtype in dataset.dtypes
                ],
                "Missing": [
                    int(dataset[c].isna().sum())
                    for c in dataset.columns
                ],
                "Unique": [
                    int(dataset[c].nunique())
                    for c in dataset.columns
                ],
            }
        )

        st.dataframe(
            information,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# EXPORTS
# ============================================================

elif section == "Exports":

    st.header("📥 Export Results")

    if classification_result is not None:

        actual = np.asarray(
            classification_result["actual"]
        )

        predicted = np.asarray(
            classification_result["predicted"]
        )

        classification_export = pd.DataFrame(
            {
                "Actual": actual,
                "Predicted": predicted,
            }
        )

        st.subheader(
            "Classification Predictions"
        )

        st.dataframe(
            classification_export,
            use_container_width=True,
        )

        st.download_button(
            "📥 Download Predictions CSV",
            data=classification_export.to_csv(
                index=False
            ),
            file_name="classification_predictions.csv",
            mime="text/csv",
        )

    else:

        st.info(
            "No classification results available."
        )

    st.divider()

    if clustering_result is not None:

        cluster_sizes = clustering_result.get(
            "cluster_sizes"
        )

        if cluster_sizes is not None:

            st.subheader(
                "Clustering Results"
            )

            st.dataframe(
                cluster_sizes,
                use_container_width=True,
            )

            st.download_button(
                "📥 Download Cluster Results",
                data=cluster_sizes.to_csv(
                    index=False
                ),
                file_name="clustering_results.csv",
                mime="text/csv",
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ML Workspace • DWDM Analytics • WEKA-style Results"
)