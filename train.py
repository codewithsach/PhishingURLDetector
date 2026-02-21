import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATASET_PATH = "dataset_phishing.csv"
MODEL_PATH = "phishing_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"


def main():
    # 1) Load data
    df = pd.read_csv(DATASET_PATH)

    # Keep only required columns (safe in case dataset has extra columns)
    df = df[["url", "status"]].dropna()

    # Remove duplicates (helps reduce noise / leakage)
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    # 2) Encode labels
    df["status"] = df["status"].map({"legitimate": 0, "phishing": 1})

    # Drop anything that did not map correctly
    df = df.dropna(subset=["status"])
    df["status"] = df["status"].astype(int)

    X = df["url"].astype(str)
    y = df["status"]

    print("Dataset size:", len(df))
    print("Label distribution:\n", y.value_counts(), "\n")

    # 3) Vectorize URLs using TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2)  # helps capture patterns like "secure login"
    )
    X_vectors = vectorizer.fit_transform(X)

    # 4) Train/Test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectors,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 5) Train model (balanced helps reduce phishing false negatives)
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # 6) Evaluate
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))

    # 7) Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved vectorizer -> {VECTORIZER_PATH}")

    # Small sanity test
    test_url = "http://secure-paypal-login.com/login"
    test_vec = vectorizer.transform([test_url])
    pred = model.predict(test_vec)[0]
    print("\nTest URL:", test_url)
    print("Prediction:", "PHISHING" if pred == 1 else "LEGITIMATE")


if __name__ == "__main__":
    main()
