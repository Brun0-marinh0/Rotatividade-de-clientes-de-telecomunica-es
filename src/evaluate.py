"""
evaluate.py — Baixa o melhor modelo registrado no DagsHub/MLflow e salva localmente.

Este script é executado:
  - Manualmente após identificar o melhor run no DagsHub
  - Automaticamente durante o 'docker build' (via Dockerfile)

Uso:
    python src/evaluate.py
"""

import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from dotenv import load_dotenv

from src.prepare_data import load_and_split

# ── Credenciais ──────────────────────────────────────────────────────────────
load_dotenv()

DAGSHUB_USER  = os.getenv("DAGSHUB_USER")
DAGSHUB_REPO  = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# ── Configuração do MLflow ───────────────────────────────────────────────────
mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
)
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN


def main():
    model_uri = "models:/ChurnClassifier/latest"

    print(f"Baixando modelo: {model_uri}")
    print(f"Fonte: https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}")

    model = mlflow.sklearn.load_model(model_uri)
    print("Modelo carregado com sucesso!")

    _, X_test, _, y_test = load_and_split()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 50)
    print("Avaliação do modelo em produção:")
    print(classification_report(y_test, y_pred))
    print(f"Acurácia : {acc:.2%}")
    print(f"F1 Score : {f1:.3f}")
    print(f"ROC AUC  : {roc:.3f}")
    print("=" * 50)

    joblib.dump(model, "model.pkl")
    print("\nModelo salvo em model.pkl — pronto para o Streamlit!")


if __name__ == "__main__":
    main()