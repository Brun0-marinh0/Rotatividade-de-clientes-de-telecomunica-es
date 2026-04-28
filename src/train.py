"""
train.py — Treina o modelo de churn e registra no MLflow/DagsHub.

Execute UMA VEZ por experimento, mudando os parâmetros marcados com # MUDE AQUI:
    python src/train.py

Experimento 1 (baseline)  : MAX_DEPTH=5, N_ESTIMATORS=100
Experimento 2 (mais profundo) : MAX_DEPTH=10, N_ESTIMATORS=200
Experimento 3 (otimizado) : MAX_DEPTH=15, N_ESTIMATORS=300
"""

import os
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from dotenv import load_dotenv

from src.prepare_data import load_and_split

# ── Carrega as credenciais do arquivo .env ───────────────────────────────────
load_dotenv()

DAGSHUB_USER  = os.getenv("DAGSHUB_USER")
DAGSHUB_REPO  = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# ── Aponta o MLflow para o servidor do DagsHub ───────────────────────────────
mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
)

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

# Nome do experimento no MLflow
mlflow.set_experiment("churn-telecom")

# ════════════════════════════════════════════════════════════════════════════
# MUDE ESTES PARÂMETROS A CADA EXPERIMENTO
# ════════════════════════════════════════════════════════════════════════════
MAX_DEPTH     = 10       # Profundidade máxima das árvores
N_ESTIMATORS  = 200      # Número de árvores no Random Forest
MIN_SAMPLES   = 2        # Minimum samples to split a node
RUN_NAME      = "exp-2-profundidade"  # MUDE a cada run!
# ════════════════════════════════════════════════════════════════════════════


def main():
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

    with mlflow.start_run(run_name=RUN_NAME):
        # 1. Loga os parâmetros
        mlflow.log_params({
            "max_depth":       MAX_DEPTH,
            "n_estimators":    N_ESTIMATORS,
            "min_samples_leaf": MIN_SAMPLES,
            "test_size":       0.2,
            "dataset":         "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        })

        # 2. Cria o modelo
        model = RandomForestClassifier(
            max_depth=MAX_DEPTH,
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES,
            random_state=42,
            n_jobs=-1,
        )

        # 3. Treina
        model.fit(X_train, y_train)

        # 4. Avalia
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        roc  = roc_auc_score(y_test, y_proba)

        print("\n" + "=" * 50)
        print(classification_report(y_test, y_pred))
        print("=" * 50)

        # 5. Loga métricas
        mlflow.log_metrics({
            "accuracy":    acc,
            "f1_score":    f1,
            "roc_auc":     roc,
        })

        # 6. Registra o modelo
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="ChurnClassifier",
        )

        print(f"\nRun '{RUN_NAME}' finalizado com sucesso!")
        print(f"Acurácia : {acc:.2%}")
        print(f"F1 Score : {f1:.3f}")
        print(f"ROC AUC  : {roc:.3f}")
        print(f"\nVisualize em: https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}")


if __name__ == "__main__":
    main()