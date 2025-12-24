import json
from src.train import train


def test_train_writes_outputs(tmp_path):
    out_model = tmp_path / "model.joblib"
    out_metrics = tmp_path / "metrics.json"

    payload = train(out_model=str(out_model), out_metrics=str(out_metrics))

    assert out_model.exists()
    assert out_metrics.exists()

    saved = json.loads(out_metrics.read_text(encoding="utf-8"))
    assert "confusion_matrix" in saved
    assert "report" in saved
    assert saved["labels"] == payload["labels"]
