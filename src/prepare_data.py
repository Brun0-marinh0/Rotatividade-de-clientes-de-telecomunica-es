"""
prepare_data.py — Carrega, limpa e divide o dataset de churn de telecomunicações.

Uso standalone:
    python src/prepare_data.py

Uso como módulo (importado pelo train.py e evaluate.py):
    from src.prepare_data import load_and_split
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split(
    path: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    test_size: float = 0.2,
    seed: int = 42,
):
    """
    Carrega o CSV de churn, faz limpeza básica e divide em treino/teste.

    Parâmetros:
        path      : caminho para o arquivo CSV
        test_size : proporção do conjunto de teste (padrão: 20%)
        seed      : semente aleatória para reprodutibilidade

    Retorna:
        X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(path)

    df = df.dropna(subset=["Churn"])

    df = df[df["Churn"].isin(["Yes", "No"])]

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)

    categorical_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod"
    ]

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop(columns=["customerID", "Churn"])
    y = df_encoded["Churn"]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()

    print(f"Total de amostras de treino : {len(X_train)}")
    print(f"Total de amostras de teste  : {len(X_test)}")
    print(f"\nDistribuição de classes no treino:")
    print(y_train.value_counts())
