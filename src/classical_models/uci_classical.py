import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
)

def train_text_model(
        model,
        X_text,
        y,
        max_features=5000,
        ngram_range=(1, 2),
        oversample_method="none",  # "none" | "random" | "borderline"
):
    """
    Train a classical ML text classification model on the UCI stance dataset.

    Dataset labels (fixed):
        - pos  : tweet agrees with / propagates misconception
        - neg  : tweet contradicts the misconception
        - na   : neutral / no stance

    Parameters:
        model: sklearn model instance (LogisticRegression, SVM, etc.)
        X_text: raw text list/Series
        y: label list/Series containing only {pos, neg, na}
        max_features: TF-IDF vocabulary size
        ngram_range: TF-IDF n-gram range
        oversample_method: "none", "random", "borderline"

    Returns:
        dict containing model, vectorizer, metrics, y_test, y_pred
    """

    # Convert to Series for convenience
    y = pd.Series(y).copy()

    # -------------------------------------------------------
    # Step 1 — Ensure only the 3 valid labels are present
    # -------------------------------------------------------
    valid_labels = {"pos", "neg", "na"}
    invalid = set(y.unique()) - valid_labels

    if invalid:
        print(f"[WARN] Dropping invalid labels: {invalid}")
        mask = y.isin(valid_labels)
        y = y[mask]
        X_text = pd.Series(X_text)[mask]

    print("[INFO] Final label distribution:")
    print(y.value_counts())

    # -------------------------------------------------------
    # Step 2 — TF-IDF vectorization
    # -------------------------------------------------------
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )
    X_vec = vectorizer.fit_transform(X_text)

    # -------------------------------------------------------
    # Step 3 — Train-test split
    # -------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------------------------------------------------------
    # Step 4 — Optional oversampling
    # -------------------------------------------------------
    if oversample_method != "none":

        print(f"[INFO] Oversampling method selected: {oversample_method}")

        before = y_train.value_counts().to_dict()
        print("  Before oversampling:", before)

        if oversample_method == "random":
            sampler = RandomOverSampler(random_state=42)

        elif oversample_method == "borderline":
            class_counts = y_train.value_counts()
            if (class_counts < 2).any():
                print("[WARN] BorderlineSMOTE skipped — class has <2 samples.")
                sampler = None
            else:
                sampler = BorderlineSMOTE(random_state=42)

        if sampler is not None:
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            after = y_train.value_counts().to_dict()
            print("  After oversampling:", after)

    # -------------------------------------------------------
    # Step 5 — Train model
    # -------------------------------------------------------
    model.fit(X_train, y_train)

    # -------------------------------------------------------
    # Step 6 — Evaluate
    # -------------------------------------------------------
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred)

    # print("\n===== Evaluation Results =====")
    # print("Accuracy:", accuracy)
    # print("Macro F1:", f1_macro)
    # print(report)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["pos", "neg", "na"])
    }
