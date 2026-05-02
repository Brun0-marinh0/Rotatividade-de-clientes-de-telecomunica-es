"""
preprocessing.py — Feature engineering analítico via DuckDB.
    python src/preprocessing.py

Lê data/raw/telco_churn.parquet → transforma via SQL → salva data/processed/features.parquet
"""

import os
import pandas as pd
import duckdb

RAW_PATH      = "data/raw/telco_churn.parquet"
FEATURES_PATH = "data/processed/features.parquet"

# SQL executado pelo DuckDB sobre o DataFrame em memória
TRANSFORM_SQL = """
WITH base AS (
    SELECT *,
           TRY_CAST(TotalCharges AS DOUBLE) AS total_charges_num
    FROM raw_data
    WHERE Churn IN ('Yes', 'No')
      AND TRY_CAST(TotalCharges AS DOUBLE) IS NOT NULL
),
engineered AS (
    SELECT
        -- ── target ─────────────────────────────────────────────────────────
        CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END                     AS churn,

        -- ── numéricas base ─────────────────────────────────────────────────
        tenure,
        CAST(MonthlyCharges AS DOUBLE)                                 AS monthly_charges,
        total_charges_num                                              AS total_charges,

        -- ── features engenheiradas ─────────────────────────────────────────
        -- custo médio histórico (total / meses ativos)
        total_charges_num / NULLIF(tenure, 0)                          AS avg_monthly_actual,
        -- desvio entre cobrança atual e histórica (signal de aumento de preço)
        CAST(MonthlyCharges AS DOUBLE) -
            total_charges_num / NULLIF(tenure, 0)                      AS monthly_delta,
        -- quantidade de serviços adicionais contratados
        (CASE WHEN OnlineSecurity  = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN OnlineBackup    = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN DeviceProtection = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN TechSupport     = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN StreamingTV     = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN StreamingMovies = 'Yes' THEN 1 ELSE 0 END)          AS num_addon_services,

        -- ── binárias (já codificadas como 0/1 pelo SQL) ────────────────────
        SeniorCitizen,
        CASE WHEN gender          = 'Male' THEN 1 ELSE 0 END           AS gender_male,
        CASE WHEN Partner         = 'Yes'  THEN 1 ELSE 0 END           AS has_partner,
        CASE WHEN Dependents      = 'Yes'  THEN 1 ELSE 0 END           AS has_dependents,
        CASE WHEN PhoneService    = 'Yes'  THEN 1 ELSE 0 END           AS has_phone,
        CASE WHEN PaperlessBilling = 'Yes' THEN 1 ELSE 0 END           AS paperless_billing,
        CASE WHEN PaymentMethod LIKE '%automatic%' THEN 1 ELSE 0 END   AS auto_payment,
        CASE WHEN OnlineSecurity   = 'Yes' THEN 1 ELSE 0 END           AS has_online_security,
        CASE WHEN OnlineBackup     = 'Yes' THEN 1 ELSE 0 END           AS has_online_backup,
        CASE WHEN DeviceProtection = 'Yes' THEN 1 ELSE 0 END           AS has_device_protection,
        CASE WHEN TechSupport      = 'Yes' THEN 1 ELSE 0 END           AS has_tech_support,
        CASE WHEN StreamingTV      = 'Yes' THEN 1 ELSE 0 END           AS has_streaming_tv,
        CASE WHEN StreamingMovies  = 'Yes' THEN 1 ELSE 0 END           AS has_streaming_movies,

        -- ── categóricas multi-valor (para OHE downstream) ─────────────────
        MultipleLines,
        InternetService,
        Contract,
        PaymentMethod

    FROM base
)
SELECT * FROM engineered
ORDER BY RANDOM()   -- embaralha para evitar ordenação por classe
"""


def run(raw_path: str = RAW_PATH, out_path: str = FEATURES_PATH) -> pd.DataFrame:
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"{raw_path} não encontrado. Execute: python src/ingestion.py"
        )
    df_raw = pd.read_parquet(raw_path)

    con = duckdb.connect()
    con.register("raw_data", df_raw)
    df = con.execute(TRANSFORM_SQL).df()
    con.close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def main():
    df = run()
    print(f"Shape           : {df.shape}")
    print(f"Churn rate      : {df['churn'].mean():.1%}")
    print(f"Features ({len(df.columns)-1}): {[c for c in df.columns if c != 'churn']}")
    print(f"→ {FEATURES_PATH}")


if __name__ == "__main__":
    main()
