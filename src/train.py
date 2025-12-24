import json
import os
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from src.features import make_features


def train(data_path="data/sample.csv", out_model="models/model.joblib", out_metrics="metrics/metrics.json"):
    df = pd.read_csv(data_path)

    X = make_features(df)
    y = df["difficulty"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    labels = sorted(y.unique().tolist())

    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)

    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    os.makedirs(os.path.dirname(out_metrics), exist_ok=True)

    dump(clf, out_model)

    payload = {
        "data": data_path,
        "n_rows": int(len(df)),
        "labels": labels,
        "confusion_matrix": cm,
        "report": report,
    }
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


if __name__ == "__main__":
    train()
