from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def train_clusterer(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    algorithm: str,
    cluster_count: int = 3,
    epsilon: float = 0.5,
    minimum_samples: int = 5,
) -> dict:
    """Train a clustering algorithm and return visualisation-ready results."""

    if not feature_columns:
        raise ValueError("Select at least one numeric feature.")

    features = dataframe[feature_columns].copy()

    # Convert selected columns to numeric and fill missing values.
    features = features.apply(pd.to_numeric, errors="coerce")

    if features.dropna(axis=1, how="all").empty:
        raise ValueError(
            "Clustering requires numeric feature columns."
        )

    features = features.dropna(axis=1, how="all")

    imputer = SimpleImputer(strategy="median")
    cleaned_features = imputer.fit_transform(features)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cleaned_features)

    if len(scaled_features) < 3:
        raise ValueError(
            "Clustering needs at least three dataset rows."
        )

    started = perf_counter()

    if algorithm == "K-Means":
        model = KMeans(
            n_clusters=cluster_count,
            n_init=10,
            random_state=42,
        )
        labels = model.fit_predict(scaled_features)

    elif algorithm == "Hierarchical clustering":
        model = AgglomerativeClustering(
            n_clusters=cluster_count
        )
        labels = model.fit_predict(scaled_features)

    elif algorithm == "DBSCAN":
        model = DBSCAN(
            eps=epsilon,
            min_samples=minimum_samples,
        )
        labels = model.fit_predict(scaled_features)

    else:
        raise ValueError("Unsupported clustering algorithm.")

    duration = perf_counter() - started

    # Create PCA coordinates for a clear two-dimensional cluster plot.
    if scaled_features.shape[1] == 1:
        coordinates = np.column_stack(
            [
                scaled_features[:, 0],
                np.zeros(len(scaled_features)),
            ]
        )
    else:
        coordinates = PCA(
            n_components=2,
            random_state=42,
        ).fit_transform(scaled_features)

    non_noise_mask = labels != -1
    non_noise_labels = labels[non_noise_mask]
    found_clusters = len(set(non_noise_labels))

    if (
        found_clusters > 1
        and len(non_noise_labels) > found_clusters
    ):
        silhouette = silhouette_score(
            scaled_features[non_noise_mask],
            non_noise_labels,
        )
    else:
        silhouette = None

    plot_data = pd.DataFrame(
        {
            "PC 1": coordinates[:, 0],
            "PC 2": coordinates[:, 1],
            "Cluster": [
                "Noise" if label == -1 else f"Cluster {label + 1}"
                for label in labels
            ],
        }
    )

    cluster_sizes = (
        plot_data["Cluster"]
        .value_counts()
        .rename_axis("Cluster")
        .reset_index(name="Rows")
    )

    output_data = dataframe.copy()
    output_data["cluster"] = plot_data["Cluster"]

    return {
        "model": model,
        "algorithm": algorithm,
        "feature_columns": list(features.columns),
        "labels": labels,
        "plot_data": plot_data,
        "cluster_sizes": cluster_sizes,
        "output_data": output_data,
        "cluster_count": found_clusters,
        "noise_points": int(np.sum(labels == -1)),
        "silhouette_score": silhouette,
        "build_time": duration,
    }