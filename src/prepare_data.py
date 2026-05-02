"""
prepare_data.py — Pré-processa o dataset de churn e salva splits em data/processed/.

Uso standalone:
    python src/prepare_data.py

Uso como módulo:
    from src.prepare_data import load_raw_splits
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

DATASET_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

BINARY_CATS  = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
MULTI_CATS   = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
NUMERIC_CONT = ["tenure", "MonthlyCharges", "TotalCharges"]
PASSTHROUGH  = ["SeniorCitizen"]


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_CONT),
            ("bin", OrdinalEncoder(), BINARY_CATS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), MULTI_CATS),
        ],
        remainder="passthrough",
    )


def load_raw_splits(path=DATASET_PATH):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["Churn", "TotalCharges"])
    df = df[df["Churn"].isin(["Yes", "No"])]
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.drop(columns=["customerID"])

    feature_cols = NUMERIC_CONT + BINARY_CATS + MULTI_CATS + PASSTHROUGH
    X = df[feature_cols]
    y = df["Churn"].astype(int)

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

    for split, X, y in [("train", X_train_p, y_train), ("val", X_val_p, y_val), ("test", X_test_p, y_test)]:
        df_out = pd.DataFrame(X, columns=feat_names)
        df_out["Churn"] = y.values
        df_out.to_parquet(f"data/processed/{split}.parquet", index=False)

    joblib.dump(preprocessor, "models/preprocessors/preprocessor.joblib")

    print(f"Train : {X_train_p.shape}  churn={y_train.mean():.1%}")
    print(f"Val   : {X_val_p.shape}    churn={y_val.mean():.1%}")
    print(f"Test  : {X_test_p.shape}   churn={y_test.mean():.1%}")
    print("→ data/processed/{train,val,test}.parquet")
    print("→ models/preprocessors/preprocessor.joblib")


if __name__ == "__main__":
    main()
