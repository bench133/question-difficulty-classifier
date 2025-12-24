from __future__ import annotations

import argparse

from src.train import TrainConfig, train_and_evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sample.csv")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--out_model", default="models/model.joblib")
    ap.add_argument("--out_metrics", default="metrics/metrics.json")
    args = ap.parse_args()

    cfg = TrainConfig(
        data_path=args.data,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        out_model=args.out_model,
        out_metrics=args.out_metrics,
    )

    train_and_evaluate(cfg)
    print(f"OK: wrote {cfg.out_model} and {cfg.out_metrics}")


if __name__ == "__main__":
    main()

