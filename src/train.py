from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.features import make_features


@dataclass
class TrainConfig:
    data_path: str = "data/sample.csv"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    out_model: str = "models/model.joblib"
    out_metrics: str = "metrics/metrics.json"


def _validate_dataset(df: pd.DataFrame) -> None:
    required = {"question_text", "avg_time_sec", "pct_correct", "difficulty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")


def train_and_evaluate(cfg: TrainConfig) -> Dict:
    df = pd.read_csv(cfg.data_path)
    _validate_dataset(df)

    X = make_features(df)
    y = df["difficulty"].astype(str)

    # stratify требует минимум по 2 примера на класс, иначе упадет.
    # для маленьких датасетов можно снять stratify, но в sample.csv классы повторяются.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_state,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    labels = sorted(y.unique().tolist())

    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
    report = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    os.makedirs(os.path.dirname(cfg.out_model), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.out_metrics), exist_ok=True)

    dump(clf, cfg.out_model)

    payload = {
        "config": asdict(cfg),
        "n_rows": int(len(df)),
        "labels": labels,
        "confusion_matrix": cm,
        "report": report,
    }

    with open(cfg.out_metrics, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload
