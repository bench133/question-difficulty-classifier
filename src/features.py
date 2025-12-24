from __future__ import annotations

import pandas as pd

try:
    import textstat
except Exception:  # pragma: no cover
    textstat = None


def safe_fk_grade(text: str) -> float:
    """Flesch-Kincaid grade level. If textstat not installed, returns 0.0."""
    if textstat is None:
        return 0.0
    try:
        return float(textstat.flesch_kincaid_grade(text or ""))
    except Exception:
        return 0.0


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature table from raw dataframe."""
    required = {"question_text", "avg_time_sec", "pct_correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for features: {sorted(missing)}")

    text = df["question_text"].fillna("").astype(str)

    X = pd.DataFrame(
        {
            "char_len": text.str.len(),
            "word_count": text.str.split().map(len),
            "fk_grade": text.map(safe_fk_grade),
            "avg_time_sec": pd.to_numeric(df["avg_time_sec"], errors="coerce").fillna(0.0),
            "pct_correct": pd.to_numeric(df["pct_correct"], errors="coerce").fillna(0.0),
        }
    )
    return X
