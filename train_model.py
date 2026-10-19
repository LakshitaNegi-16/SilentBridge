
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# File locations
DATASET_PATH = Path("dataset/landmarks.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "sign_model.joblib"


def main():
    # Check that the dataset exists
    if not DATASET_PATH.exists():
        print("Error: Dataset not found.")
        print("Run collect_data.py first to collect hand-sign samples.")
        return

    # Load the dataset
    features = []
    labels = []

    with DATASET_PATH.open("r", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "label" not in reader.fieldnames:
            print("Error: CSV must contain a 'label' column.")
            return

        feature_columns = [
            column for column in reader.fieldnames
            if column.startswith("feature_")
        ]

        if len(feature_columns) != 63:
            print("Error: Expected 63 hand-landmark features.")
            print(f"Found: {len(feature_columns)}")
            return

        for row in reader:
            try:
                sample = [float(row[column]) for column in feature_columns]
                label = row["label"].strip()

                if label and np.isfinite(sample).all():
                    features.append(sample)
                    labels.append(label)

            except (ValueError, TypeError):
                continue

    if not features:
        print("Error: No valid training samples found.")
        return

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)

    # Ensure every class has enough samples for a stratified split
    unique_labels, counts = np.unique(y, return_counts=True)

    if len(unique_labels) < 2:
        print("Error: Collect samples for at least two different signs.")
        return

    if np.min(counts) < 2:
        print("Error: Each sign needs at least two samples.")
        print("Collect more samples for every sign before training.")
        return

    print(f"Total samples: {len(X)}")
    print(f"Number of signs: {len(unique_labels)}")
    print("Signs:", ", ".join(unique_labels))

    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    print("\nTraining SilentBridge AI...")
    model.fit(X_train, y_train)

    # Evaluate on the held-out test samples
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    # Save the trained model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print("Training complete!")


if __name__ == "__main__":
    main()
