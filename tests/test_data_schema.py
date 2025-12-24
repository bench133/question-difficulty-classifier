import pandas as pd


def test_sample_data_has_required_columns():
    df = pd.read_csv("data/sample.csv")
    for col in ["question_text", "avg_time_sec", "pct_correct", "difficulty"]:
        assert col in df.columns
