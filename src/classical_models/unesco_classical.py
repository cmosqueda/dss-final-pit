from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE
import pandas as pd


def train_text_model(
        model,
        X_text,
        y,
        max_features=5000,
        ngram_range=(1, 2),
        merge_small_classes=True,
        min_samples=5,
        oversample_method="none",  # "none" | "random" | "borderline"
):
    """
    Train a text classification model on TF-IDF features with optional class merging 
    and oversampling.

    Parameters:
        model: sklearn model instance (LogisticRegression, NaiveBayes, SVM, etc.)
        X_text: raw text list/Series
        y: label list/Series
        max_features: TF-IDF max features
        ngram_range: TF-IDF ngram range
        merge_small_classes: merge rare classes into the largest class
        min_samples: rare-class threshold for merging
        oversample_method: oversampling strategy ("none", "random", "borderline")

    Returns:
        dict containing model, vectorizer, metrics, y_test, and y_pred
    """

    y = pd.Series(y).copy()

    # Merge rare classes if needed
    if merge_small_classes:
        value_counts = y.value_counts()
        small_classes = value_counts[value_counts < min_samples].index.tolist()
        if small_classes:
            print(f"[INFO] Merging small classes: {small_classes}")
            largest_class = value_counts.idxmax()
            y = y.replace(small_classes, largest_class)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_vec = vectorizer.fit_transform(X_text)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------
    # Oversampling Logic
    # -------------------------
    if oversample_method != "none":
        print(f"[INFO] Oversampling method selected: {oversample_method}")

        before_dist = pd.Series(y_train).value_counts().to_dict()
        print(f"  Before oversampling: {before_dist}")

        if oversample_method == "random":
            sampler = RandomOverSampler(random_state=42, sampling_strategy="not majority")

        elif oversample_method == "borderline":
            # Safe mode: prevent SMOTE errors when samples are too low
            class_counts = pd.Series(y_train).value_counts()
            if (class_counts < 2).any():
                print("[WARN] Some classes have < 2 samples; BorderlineSMOTE skipped.")
                sampler = None
            else:
                sampler = BorderlineSMOTE(random_state=42, sampling_strategy="not majority")

        else:
            raise ValueError("oversample_method must be 'none', 'random', or 'borderline'")

        if sampler is not None:
            X_train, y_train = sampler.fit_resample(X_train, y_train)

            after_dist = pd.Series(y_train).value_counts().to_dict()
            print(f"  After oversampling: {after_dist}")
        else:
            print("[INFO] No oversampling applied due to insufficient samples.")

    # Fit the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
    }
