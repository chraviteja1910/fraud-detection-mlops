"""
MLflow Training Pipeline for Fraud Detection
Full experiment tracking, model registry, and artifact logging
Author: Ravi Teja Chittaluri
"""

from __future__ import annotations
import argparse
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from .ensemble import StackingEnsemble, XGBoostFraudClassifier, NeuralNetClassifier, GraphAnomalyDetector

EXPERIMENT_NAME = "fraud-detection"
REGISTERED_MODEL_NAME = "fraud-detection-ensemble"


def train(
    data_path: str = "data/features/",
    experiment_name: str = EXPERIMENT_NAME,
    run_name: str = "ensemble-v1",
    register_model: bool = True,
):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        # ── Load data ──────────────────────────────────────────
        print("Loading feature data...")
        X = np.load(f"{data_path}/X.npy")
        y = np.load(f"{data_path}/y.npy")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # ── Log params ─────────────────────────────────────────
        config = {
            "n_base_models": 3,
            "xgb_n_estimators": 500,
            "xgb_max_depth": 8,
            "nn_hidden_dims": "256-128-64",
            "nn_dropout": 0.3,
            "graph_contamination": 0.01,
            "n_folds": 5,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "fraud_rate": float(y.mean()),
        }
        mlflow.log_params(config)

        # ── Train ──────────────────────────────────────────────
        print("Training stacking ensemble...")
        ensemble = StackingEnsemble(
            base_models=[
                XGBoostFraudClassifier(n_estimators=500, max_depth=8),
                NeuralNetClassifier(hidden_dims=[256, 128, 64]),
                GraphAnomalyDetector(contamination=0.01),
            ]
        )
        ensemble.fit(X_train, y_train)

        # ── Evaluate ───────────────────────────────────────────
        print("Evaluating...")
        metrics = ensemble.evaluate(X_test, y_test)
        mlflow.log_metrics({
            "test_auc": metrics.auc,
            "test_f1": metrics.f1,
            "test_precision": metrics.precision,
            "test_recall": metrics.recall,
            "test_false_positive_rate": metrics.false_positive_rate,
        })
        print(f"  AUC: {metrics.auc:.4f} | F1: {metrics.f1:.4f} | Precision: {metrics.precision:.4f}")

        # ── Log model ──────────────────────────────────────────
        mlflow.sklearn.log_model(
            ensemble,
            artifact_path="fraud_ensemble",
            registered_model_name=REGISTERED_MODEL_NAME if register_model else None,
        )

        # ── Transition to staging if AUC > threshold ───────────
        if register_model and metrics.auc > 0.95:
            client = mlflow.MlflowClient()
            latest = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["None"])
            if latest:
                client.transition_model_version_stage(
                    name=REGISTERED_MODEL_NAME,
                    version=latest[0].version,
                    stage="Staging",
                )
                print(f"Model v{latest[0].version} promoted to Staging (AUC={metrics.auc:.4f})")

        print(f"Run ID: {run.info.run_id}")
        return run.info.run_id, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--run-name", default="ensemble-v1")
    parser.add_argument("--data-path", default="data/features/")
    args = parser.parse_args()
    train(args.data_path, args.experiment_name, args.run_name)
