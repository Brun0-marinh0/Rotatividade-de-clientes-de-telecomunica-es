"""
ingestion.py — Busca dados do Supabase e retorna DataFrame pronto para o pipeline.

Uso standalone:
    python src/ingestion.py

Uso como módulo:
    from src.ingestion import fetch_churn_data
    df = fetch_churn_data()
"""

import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME   = "telco_churn"


def get_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY or not SUPABASE_KEY.strip():
        raise EnvironmentError(
            "SUPABASE_KEY não configurada.\n"
            "Acesse: Settings → API → anon public key e adicione ao .env"
        )
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_churn_data(batch_size: int = 1000) -> pd.DataFrame:
    """Baixa todos os registros de telco_churn em batches e retorna DataFrame."""
    client = get_client()
    rows   = []
    offset = 0

    while True:
        response = (
            client.table(TABLE_NAME)
            .select("*")
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        batch = response.data
        if not batch:
            break
        rows.extend(batch)
        offset += batch_size
        if len(batch) < batch_size:
            break

    df = pd.DataFrame(rows)
    return df


def main():
    os.makedirs("data/raw", exist_ok=True)
    print(f"Conectando a {SUPABASE_URL}…")
    df = fetch_churn_data()
    out = "data/raw/telco_churn.parquet"
    df.to_parquet(out, index=False)
    print(f"Shape : {df.shape}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
