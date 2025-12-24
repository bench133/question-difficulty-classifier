import pandas as pd
from src.features import make_features


def test_make_features_basic():
    df = pd.DataFrame(
        {
            "question_text": ["Hi", "Hello world"],
            "avg_time_sec": [10, 20],
            "pct_correct": [0.9, 0.2],
        }
    )
    X = make_features(df)
    assert len(X) == 2
    assert all(c in X.columns for c in ["char_len", "word_count", "fk_grade", "avg_time_sec", "pct_correct"])
