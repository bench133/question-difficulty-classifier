import json
from src.train import TrainConfig, train_and_evaluate


def test_train_and_evaluate_writes_outputs(tmp_path):
    out_model = tmp_path / "model.joblib"
    out_metrics = tmp_path / "metrics.json"

    cfg = TrainConfig(
        data_path="data/sample.csv",
        out_model=str(out_model),
        out_metrics=str(out_metrics),
        test_size=0.4,
    )

    payload = train_and_evaluate(cfg)

    assert out_model.exists()
    assert out_metrics.exists()

    saved = json.loads(out_metrics.read_text(encoding="utf-8"))
    assert "confusion_matrix" in saved
    assert "report" in saved
    assert saved["labels"] == payload["labels"]

