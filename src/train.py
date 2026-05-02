"""
train.py — Treina 3 modelos e registra no MLflow/DagsHub.
    python src/train.py
"""

import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, f1_score, roc_auc_score, accuracy_score
from dotenv import load_dotenv

load_dotenv()

DAGSHUB_USER  = os.getenv("DAGSHUB_USER")
DAGSHUB_REPO  = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN


def load_splits():
    train = pd.read_parquet("data/processed/train.parquet")
    val   = pd.read_parquet("data/processed/val.parquet")
    X_train = train.drop(columns=["Churn"]).values
    y_train = train["Churn"].values
    X_val   = val.drop(columns=["Churn"]).values
    y_val   = val["Churn"].values
    return X_train, X_val, y_train, y_val


def compute_metrics(model, X, y):
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "recall_churn": float(recall_score(y, y_pred)),
        "f1_weighted":  float(f1_score(y, y_pred, average="weighted")),
        "roc_auc":      float(roc_auc_score(y, y_proba)),
        "accuracy":     float(accuracy_score(y, y_pred)),
    }


def main():
    os.makedirs("models/trained", exist_ok=True)

    X_train, X_val, y_train, y_val = load_splits()

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos = round(n_neg / n_pos, 3)

    experiments = [
        {
            "run_name": "logistic_regression",
            "model": LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced", random_state=42
            ),
            "params": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
        },
        {
            "run_name": "random_forest",
            "model": RandomForestClassifier(
                n_estimators=200, max_depth=10, class_weight="balanced",
                random_state=42, n_jobs=-1
            ),
            "params": {"n_estimators": 200, "max_depth": 10, "class_weight": "balanced"},
        },
        {
            "run_name": "xgboost",
            "model": XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                scale_pos_weight=scale_pos, random_state=42,
                eval_metric="logloss", verbosity=0,
            ),
            "params": {
                "n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
                "scale_pos_weight": scale_pos,
            },
        },
    ]

    mlflow.set_experiment("telecom-churn")

    results = []
    for exp in experiments:
        with mlflow.start_run(run_name=exp["run_name"]):
            mlflow.log_params(exp["params"])
            exp["model"].fit(X_train, y_train)
            metrics = compute_metrics(exp["model"], X_val, y_val)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(exp["model"], artifact_path="model")
            path = f"models/trained/{exp['run_name']}.joblib"
            joblib.dump(exp["model"], path)
            results.append({"model": exp["run_name"], **metrics})
            print(f"✓ {exp['run_name']} saved → {path}")

    df = pd.DataFrame(results).set_index("model")
    print("\n" + df[["recall_churn", "f1_weighted", "roc_auc"]].to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
