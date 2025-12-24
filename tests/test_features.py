import pandas as pd
from src.features import make_features


def test_make_features_columns():
    df = pd.DataFrame(
        {
            "question_text": ["hi", "hello world"],
            "avg_time_sec": [10, 20],
            "pct_correct": [0.9, 0.2],
        }
    )
    X = make_features(df)
    assert set(["char_len", "word_count", "fk_grade", "avg_time_sec", "pct_correct"]).issubset(X.columns)
    assert len(X) == 2
