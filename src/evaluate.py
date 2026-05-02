"""
evaluate.py — Avalia os 3 modelos no conjunto de teste e registra métricas no DagsHub.
    python src/evaluate.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, f1_score, roc_auc_score, accuracy_score
from dotenv import load_dotenv

load_dotenv()

DAGSHUB_USER  = os.getenv("DAGSHUB_USER")
DAGSHUB_REPO  = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN


MODEL_NAMES   = ["logistic_regression", "random_forest", "xgboost"]
REGISTRY_NAME = "TelcoChurnClassifier"


def load_models():
    return {n: joblib.load(f"models/trained/{n}.joblib") for n in MODEL_NAMES}


def load_champion():
    """Carrega o modelo champion diretamente do MLflow Model Registry."""
    uri = f"models:/{REGISTRY_NAME}@champion"
    return mlflow.sklearn.load_model(uri)


def evaluate(model, X, y):
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "Recall(churn)": recall_score(y, y_pred),
        "F1":            f1_score(y, y_pred, average="weighted"),
        "ROC-AUC":       roc_auc_score(y, y_proba),
        "Accuracy":      accuracy_score(y, y_pred),
    }


_KEY_MAP = {
    "Recall(churn)": "recall_churn",
    "F1":            "f1_weighted",
    "ROC-AUC":       "roc_auc",
    "Accuracy":      "accuracy",
}


def log_test_metrics_to_dagshub(model_name, metrics):
    mlflow.set_experiment("telecom-churn")
    with mlflow.start_run(run_name=f"{model_name}_test_eval"):
        mlflow.set_tag("stage", "test_evaluation")
        mlflow.set_tag("model", model_name)
        for display_key, val in metrics.items():
            safe_key = _KEY_MAP.get(display_key, display_key.lower().replace("-", "_").replace("(", "").replace(")", ""))
            mlflow.log_metric(f"test_{safe_key}", float(val))
    print(f"  ✓ run '{model_name}_test_eval' criada no DagsHub")


def main():
    os.makedirs("reports", exist_ok=True)

    test    = pd.read_parquet("data/processed/test.parquet")
    X_test  = test.drop(columns=["Churn"]).values
    y_test  = test["Churn"].values
    feat_names = list(test.drop(columns=["Churn"]).columns)

    models  = load_models()
    results = {n: evaluate(m, X_test, y_test) for n, m in models.items()}
    df      = pd.DataFrame(results).T
    df.index.name = "Modelo"

    print("\n" + df.to_string(float_format="{:.4f}".format))

    print("\nRegistrando métricas de teste no DagsHub…")
    for name, metrics in results.items():
        log_test_metrics_to_dagshub(name, metrics)

    # barplot comparativo
    metrics = list(df.columns)
    x       = np.arange(len(df))
    width   = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, df[m], width, label=m)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df.index, rotation=10)
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.set_title("Model Comparison — Test Set (Holdout)")
    plt.tight_layout()
    plt.savefig("reports/comparison_chart.png", dpi=120)
    plt.close()

    # feature importance do melhor modelo (ROC-AUC)
    best_name  = df["ROC-AUC"].idxmax()
    best_model = models[best_name]

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
    else:
        importances = np.zeros(len(feat_names))

    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    fi[::-1].plot.barh(ax=ax)
    ax.set_title(f"Top-10 Feature Importance — {best_name}")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=120)
    plt.close()

    # model_comparison.md
    best_roc = df.loc[best_name, "ROC-AUC"]
    md = "# Model Comparison — Telecom Churn\n\n"
    md += df.to_markdown() + "\n\n"
    md += f"## Veredito\nMelhor modelo por ROC-AUC: **{best_name}** ({best_roc:.4f})\n\n"
    md += f"Registrado no MLflow Model Registry como `{REGISTRY_NAME}@champion`\n"
    with open("reports/model_comparison.md", "w") as f:
        f.write(md)

    # Verifica que o champion do Registry carrega corretamente
    print("\nVerificando champion no Model Registry…")
    try:
        champion = load_champion()
        y_ch = champion.predict(X_test)
        print(f"  ✓ {REGISTRY_NAME}@champion carregado — accuracy spot-check: {accuracy_score(y_test, y_ch):.4f}")
    except Exception as e:
        print(f"  [warn] {e}")

    print(f"\n→ reports/comparison_chart.png")
    print(f"→ reports/feature_importance.png  (best: {best_name})")
    print(f"→ reports/model_comparison.md")


if __name__ == "__main__":
    main()
