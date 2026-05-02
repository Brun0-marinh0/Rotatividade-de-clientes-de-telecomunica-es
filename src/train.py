"""
train.py — Treina 3 modelos, registra no MLflow/DagsHub e promove o melhor ao Model Registry.
    python src/train.py
"""

import os
import tempfile
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    recall_score, f1_score, roc_auc_score, accuracy_score,
    precision_score, classification_report, confusion_matrix,
    RocCurveDisplay, ConfusionMatrixDisplay,
)
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

DAGSHUB_USER  = os.getenv("DAGSHUB_USER")
DAGSHUB_REPO  = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
EXPERIMENT    = "telecom-churn"
REGISTRY_NAME = "TelcoChurnClassifier"

mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN


# ── Dados ────────────────────────────────────────────────────────────────────

def load_splits():
    train = pd.read_parquet("data/processed/train.parquet")
    val   = pd.read_parquet("data/processed/val.parquet")
    X_train = train.drop(columns=["Churn"]).values
    y_train = train["Churn"].values
    X_val   = val.drop(columns=["Churn"]).values
    y_val   = val["Churn"].values
    feat_names = list(train.drop(columns=["Churn"]).columns)
    return X_train, X_val, y_train, y_val, feat_names


# ── Métricas ─────────────────────────────────────────────────────────────────

def compute_metrics(model, X, y):
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "recall_churn":  float(recall_score(y, y_pred)),
        "precision_churn": float(precision_score(y, y_pred)),
        "f1_churn":      float(f1_score(y, y_pred)),
        "f1_weighted":   float(f1_score(y, y_pred, average="weighted")),
        "roc_auc":       float(roc_auc_score(y, y_proba)),
        "accuracy":      float(accuracy_score(y, y_pred)),
    }, y_pred, y_proba


# ── Artefatos ────────────────────────────────────────────────────────────────

def log_plots(model, X, y, y_pred, y_proba, feat_names, run_name):
    with tempfile.TemporaryDirectory() as tmp:

        # ROC Curve
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y, y_proba, ax=ax, name=run_name)
        ax.set_title(f"ROC Curve — {run_name}")
        plt.tight_layout()
        roc_path = os.path.join(tmp, "roc_curve.png")
        fig.savefig(roc_path, dpi=100)
        plt.close(fig)
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y, y_pred, display_labels=["No Churn", "Churn"],
            cmap="Blues", ax=ax,
        )
        ax.set_title(f"Confusion Matrix — {run_name}")
        plt.tight_layout()
        cm_path = os.path.join(tmp, "confusion_matrix.png")
        fig.savefig(cm_path, dpi=100)
        plt.close(fig)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # Feature Importance (RF / XGBoost) ou coeficientes (LR)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return

        fi = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        fi[::-1].plot.barh(ax=ax, color="steelblue")
        ax.set_title(f"Top-15 Feature Importance — {run_name}")
        plt.tight_layout()
        fi_path = os.path.join(tmp, "feature_importance.png")
        fig.savefig(fi_path, dpi=100)
        plt.close(fig)
        mlflow.log_artifact(fi_path, artifact_path="plots")

        # Classification Report (texto)
        report = classification_report(y, y_pred, target_names=["No Churn", "Churn"])
        rpt_path = os.path.join(tmp, "classification_report.txt")
        with open(rpt_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(rpt_path, artifact_path="reports")


# ── Model Registry ────────────────────────────────────────────────────────────

def register_best_model(results: list[dict]):
    best = max(results, key=lambda r: r["roc_auc"])
    print(f"\nMelhor modelo: {best['run_name']}  ROC-AUC={best['roc_auc']:.4f}")

    model_uri = f"runs:/{best['run_id']}/model"
    mv = mlflow.register_model(model_uri, REGISTRY_NAME)

    client = MlflowClient()
    client.update_registered_model(
        name=REGISTRY_NAME,
        description=(
            f"Classificador de churn para telecomunicações. "
            f"Melhor modelo: {best['run_name']} (ROC-AUC={best['roc_auc']:.4f})"
        ),
    )
    client.update_model_version(
        name=REGISTRY_NAME,
        version=mv.version,
        description=f"Treinado em {best['run_name']} — val ROC-AUC={best['roc_auc']:.4f}",
    )

    # Promote to Staging (DagsHub suporta alias em vez de stage nas versões novas)
    try:
        client.set_registered_model_alias(REGISTRY_NAME, "champion", mv.version)
        print(f"✓ alias 'champion' → versão {mv.version}")
    except Exception:
        pass  # alias API pode não estar disponível em todas as versões

    print(f"✓ {REGISTRY_NAME} v{mv.version} registrado no Model Registry")
    return mv


# ── Pipeline principal ────────────────────────────────────────────────────────

def main():
    os.makedirs("models/trained", exist_ok=True)

    X_train, X_val, y_train, y_val, feat_names = load_splits()

    n_neg     = int((y_train == 0).sum())
    n_pos     = int((y_train == 1).sum())
    scale_pos = round(n_neg / n_pos, 3)

    experiments = [
        {
            "run_name":  "logistic_regression",
            "model_type": "LogisticRegression",
            "model": LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced", random_state=42
            ),
            "params": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
        },
        {
            "run_name":  "random_forest",
            "model_type": "RandomForestClassifier",
            "model": RandomForestClassifier(
                n_estimators=200, max_depth=10, class_weight="balanced",
                random_state=42, n_jobs=-1,
            ),
            "params": {"n_estimators": 200, "max_depth": 10, "class_weight": "balanced"},
        },
        {
            "run_name":  "xgboost",
            "model_type": "XGBClassifier",
            "model": XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                scale_pos_weight=scale_pos, random_state=42,
                eval_metric="logloss", verbosity=0,
            ),
            "params": {
                "n_estimators": 200, "max_depth": 6,
                "learning_rate": 0.05, "scale_pos_weight": scale_pos,
            },
        },
    ]

    mlflow.set_experiment(EXPERIMENT)

    results = []
    for exp in experiments:
        with mlflow.start_run(run_name=exp["run_name"]) as run:

            # Tags descritivas
            mlflow.set_tags({
                "model_type":     exp["model_type"],
                "task":           "binary_classification",
                "target":         "churn",
                "dataset":        "telco_churn",
                "feature_count":  str(X_train.shape[1]),
                "train_samples":  str(X_train.shape[0]),
                "val_samples":    str(X_val.shape[0]),
                "class_balance":  f"neg={n_neg} pos={n_pos} ratio={scale_pos}",
                "author":         DAGSHUB_USER,
                "pipeline":       "dvc",
            })

            mlflow.log_params(exp["params"])

            exp["model"].fit(X_train, y_train)

            metrics, y_pred, y_proba = compute_metrics(exp["model"], X_val, y_val)
            mlflow.log_metrics(metrics)

            # Modelo como artefato MLflow
            mlflow.sklearn.log_model(
                exp["model"],
                artifact_path="model",
                input_example=X_val[:3],
            )

            # Plots + relatório como artefatos
            log_plots(exp["model"], X_val, y_val, y_pred, y_proba,
                      feat_names, exp["run_name"])

            # Preprocessor e features.parquet para carregamento em runtime
            mlflow.log_artifact("models/preprocessors/preprocessor.joblib", artifact_path="preprocessor")
            mlflow.log_artifact("data/processed/features.parquet", artifact_path="data")

            # Salva .joblib local (para dvc pipeline)
            local_path = f"models/trained/{exp['run_name']}.joblib"
            joblib.dump(exp["model"], local_path)

            results.append({
                "run_name": exp["run_name"],
                "run_id":   run.info.run_id,
                **metrics,
            })
            print(f"✓ {exp['run_name']}  roc_auc={metrics['roc_auc']:.4f}  recall={metrics['recall_churn']:.4f}")

    # Tabela resumo
    df = pd.DataFrame(results).set_index("run_name")
    cols = ["recall_churn", "precision_churn", "f1_weighted", "roc_auc", "accuracy"]
    print("\n" + df[cols].to_string(float_format="{:.4f}".format))

    # Registra melhor modelo no Model Registry
    register_best_model(results)


if __name__ == "__main__":
    main()
