"""
prepare_data.py — Encoding sklearn + split 70/15/15 sobre features do DuckDB.

Uso standalone:
    python src/prepare_data.py

Uso como módulo:
    from src.prepare_data import load_raw_splits
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

FEATURES_PATH = "data/processed/features.parquet"

NUMERIC_FEATS = [
    "tenure", "monthly_charges", "total_charges",
    "avg_monthly_actual", "monthly_delta", "num_addon_services",
]
OHE_FEATS = ["MultipleLines", "InternetService", "Contract", "PaymentMethod"]
BINARY_FEATS = [
    "SeniorCitizen", "gender_male", "has_partner", "has_dependents",
    "has_phone", "paperless_billing", "auto_payment",
    "has_online_security", "has_online_backup", "has_device_protection",
    "has_tech_support", "has_streaming_tv", "has_streaming_movies",
]


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), OHE_FEATS),
        ],
        remainder="passthrough",  # binary features passam sem transformação
    )


def load_raw_splits(path: str = FEATURES_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} não encontrado. Execute: python src/preprocessing.py"
        )
    df = pd.read_parquet(path)
    feature_cols = NUMERIC_FEATS + OHE_FEATS + BINARY_FEATS
    X = df[feature_cols]
    y = df["churn"].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models/preprocessors", exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_raw_splits()

    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p   = preprocessor.transform(X_val)
    X_test_p  = preprocessor.transform(X_test)

    feat_names = list(preprocessor.get_feature_names_out())

    for split, X, y in [
        ("train", X_train_p, y_train),
        ("val",   X_val_p,   y_val),
        ("test",  X_test_p,  y_test),
    ]:
        df_out = pd.DataFrame(X, columns=feat_names)
        df_out["Churn"] = y.values
        df_out.to_parquet(f"data/processed/{split}.parquet", index=False)

    joblib.dump(preprocessor, "models/preprocessors/preprocessor.joblib")

    print(f"Train : {X_train_p.shape}  churn={y_train.mean():.1%}")
    print(f"Val   : {X_val_p.shape}   churn={y_val.mean():.1%}")
    print(f"Test  : {X_test_p.shape}  churn={y_test.mean():.1%}")
    print(f"Features totais: {X_train_p.shape[1]}")
    print("→ data/processed/{train,val,test}.parquet")
    print("→ models/preprocessors/preprocessor.joblib")


if __name__ == "__main__":
    main()
