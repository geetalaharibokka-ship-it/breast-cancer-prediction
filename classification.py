from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from core.preprocessing import build_preprocessor


def get_classifier(
    algorithm: str,
    neighbours: int = 5,
    max_depth: int = 6,
):
    """Return the selected classification algorithm."""

    if algorithm == "ID3 decision tree":
        return DecisionTreeClassifier(
            criterion="entropy",
            max_depth=max_depth,
            random_state=42,
        )

    if algorithm == "J48 / C4.5-style tree":
        return DecisionTreeClassifier(
            criterion="entropy",
            max_depth=max_depth,
            min_samples_leaf=2,
            random_state=42,
        )

    if algorithm == "K-nearest neighbours":
        return KNeighborsClassifier(n_neighbors=neighbours)

    if algorithm == "Naive Bayes":
        return GaussianNB()

    if algorithm == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        )

    raise ValueError("Unsupported classification algorithm.")


def train_classifier(
    dataframe: pd.DataFrame,
    preparation_config: dict,
    algorithm: str,
    evaluation_method: str,
    test_size: float = 0.20,
    folds: int = 5,
    neighbours: int = 5,
    max_depth: int = 6,
) -> dict:
    """Train and evaluate a classifier."""

    target_column = preparation_config["target_column"]
    feature_columns = preparation_config["feature_columns"]

    features = dataframe[feature_columns].copy()
    target = dataframe[target_column].astype(str).copy()

    # Remove rows where the target value is missing.
    valid_rows = target.notna()
    features = features.loc[valid_rows]
    target = target.loc[valid_rows]

    if target.nunique() < 2:
        raise ValueError(
            "The target column must contain at least two classes."
        )

    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(target)
    class_names = list(label_encoder.classes_)

    preprocessor = build_preprocessor(
        dataframe=features,
        feature_columns=feature_columns,
        scale_numeric=preparation_config["scale_numeric"],
    )

    classifier = get_classifier(
        algorithm=algorithm,
        neighbours=neighbours,
        max_depth=max_depth,
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    started = perf_counter()

    if evaluation_method == "Stratified cross-validation":
        smallest_class = int(target.value_counts().min())
        usable_folds = min(folds, smallest_class)

        if usable_folds < 2:
            raise ValueError(
                "Cross-validation needs at least two rows in every class."
            )

        splitter = StratifiedKFold(
            n_splits=usable_folds,
            shuffle=True,
            random_state=42,
        )

        predicted = cross_val_predict(
            model,
            features,
            encoded_target,
            cv=splitter,
            method="predict",
        )

        probabilities = cross_val_predict(
            model,
            features,
            encoded_target,
            cv=splitter,
            method="predict_proba",
        )

        actual = encoded_target

        # Fit the final model using every row.
        model.fit(features, encoded_target)

        evaluation_note = (
            f"{usable_folds}-fold stratified cross-validation"
        )

    else:
        if target.value_counts().min() < 2:
            raise ValueError(
                "Holdout testing needs at least two rows in every class."
            )

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            encoded_target,
            test_size=test_size,
            random_state=42,
            stratify=encoded_target,
        )

        if (
            algorithm == "K-nearest neighbours"
            and neighbours > len(x_train)
        ):
            raise ValueError(
                "K must be smaller than the number of training rows."
            )

        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        probabilities = model.predict_proba(x_test)
        actual = y_test

        evaluation_note = (
            f"{int(test_size * 100)}% stratified holdout test set"
        )

    duration = perf_counter() - started

    return {
        "model": model,
        "algorithm": algorithm,
        "actual": actual,
        "predicted": predicted,
        "probabilities": probabilities,
        "class_names": class_names,
        "feature_columns": feature_columns,
        "evaluation_note": evaluation_note,
        "build_time": duration,
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual,
            predicted,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            actual,
            predicted,
            average="weighted",
            zero_division=0,
        ),
        "f1_score": f1_score(
            actual,
            predicted,
            average="weighted",
            zero_division=0,
        ),
        "kappa": cohen_kappa_score(actual, predicted),
        "evaluated_rows": len(actual),
        "correct_rows": int(np.sum(actual == predicted)),
    }