"""
upload_hf.py — Faz upload dos modelos treinados para o Hugging Face Hub.
    python src/upload_hf.py

Requer HF_TOKEN no .env ou variável de ambiente com permissão write.
"""

import os
import joblib
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()

HF_TOKEN   = os.getenv("HF_TOKEN")
HF_USER    = "RafaelPrime"
MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]


def load_test_metrics():
    test = pd.read_parquet("data/processed/test.parquet")
    X    = test.drop(columns=["Churn"]).values
    y    = test["Churn"].values
    from sklearn.metrics import recall_score, f1_score, roc_auc_score, accuracy_score
    metrics = {}
    for name in MODEL_NAMES:
        m      = joblib.load(f"models/trained/{name}.joblib")
        y_pred = m.predict(X)
        y_prob = m.predict_proba(X)[:, 1]
        metrics[name] = {
            "recall_churn": recall_score(y, y_pred),
            "f1_weighted":  f1_score(y, y_pred, average="weighted"),
            "roc_auc":      roc_auc_score(y, y_prob),
            "accuracy":     accuracy_score(y, y_pred),
        }
    return metrics


def make_readme(model_name, metrics):
    m = metrics[model_name]
    return f"""---
language: pt
tags:
  - classification
  - churn-prediction
  - telecom
  - scikit-learn
---

# telecom-churn-{model_name}

Modelo de classificação binária para previsão de churn em clientes de telecomunicações.

## Métricas (conjunto de teste — holdout 15%)

| Métrica       | Valor  |
|---------------|--------|
| Recall(churn) | {m['recall_churn']:.4f} |
| F1 (weighted) | {m['f1_weighted']:.4f} |
| ROC-AUC       | {m['roc_auc']:.4f} |
| Accuracy      | {m['accuracy']:.4f} |

## Uso

```python
import joblib, pandas as pd
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="{HF_USER}/telecom-churn-{model_name}", filename="{model_name}.joblib")
prep_path  = hf_hub_download(repo_id="{HF_USER}/telecom-churn-{model_name}", filename="preprocessor.joblib")

model      = joblib.load(model_path)
preprocessor = joblib.load(prep_path)

# X_raw: DataFrame com as colunas originais do dataset
X_proc = preprocessor.transform(X_raw)
pred   = model.predict(X_proc)
```
"""


def best_model_name(metrics):
    return max(metrics, key=lambda n: metrics[n]["roc_auc"])


def main():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN não encontrado. Adicione ao .env ou variável de ambiente.")

    api     = HfApi(token=HF_TOKEN)
    metrics = load_test_metrics()
    name    = best_model_name(metrics)
    m       = metrics[name]

    print(f"Melhor modelo (ROC-AUC {m['roc_auc']:.4f}): {name}")

    repo_id = f"{HF_USER}/telecom-churn-{name}"
    print(f"→ {repo_id}")

    create_repo(repo_id, token=HF_TOKEN, repo_type="model", exist_ok=True, private=False)

    api.upload_file(
        path_or_fileobj=f"models/trained/{name}.joblib",
        path_in_repo=f"{name}.joblib",
        repo_id=repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj="models/preprocessors/preprocessor.joblib",
        path_in_repo="preprocessor.joblib",
        repo_id=repo_id,
        repo_type="model",
    )
    readme_content = make_readme(name, metrics)
    readme_path    = f"/tmp/README_{name}.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    url = f"https://huggingface.co/{repo_id}"
    os.makedirs("reports", exist_ok=True)
    with open("reports/hf_urls.txt", "w") as f:
        f.write(url + "\n")

    print(f"✓ {url}")
    print("→ reports/hf_urls.txt")


if __name__ == "__main__":
    main()
