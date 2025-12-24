import pandas as pd

try:
    import textstat
except Exception:
    textstat = None


def safe_fk_grade(text: str) -> float:
    if textstat is None:
        return 0.0
    try:
        return float(textstat.flesch_kincaid_grade(text or ""))
    except Exception:
        return 0.0


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df["question_text"].fillna("").astype(str)

    feats = pd.DataFrame(
        {
            "char_len": text.str.len(),
            "word_count": text.str.split().map(len),
            "fk_grade": text.map(safe_fk_grade),
            "avg_time_sec": pd.to_numeric(df["avg_time_sec"], errors="coerce").fillna(0.0),
            "pct_correct": pd.to_numeric(df["pct_correct"], errors="coerce").fillna(0.0),
        }
    )
    return feats
