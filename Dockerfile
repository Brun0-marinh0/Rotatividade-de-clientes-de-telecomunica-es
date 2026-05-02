FROM python:3.11-slim


# ── Sistema ────────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependências Python (camada cacheada separadamente do código) ──────────────
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# ── Código fonte ───────────────────────────────────────────────────────────────
COPY src/   src/
COPY app/   app/

# ── Artefatos pré-treinados (gerados por `dvc repro` localmente) ───────────────
COPY models/trained/        models/trained/
COPY models/preprocessors/  models/preprocessors/
COPY data/processed/features.parquet  data/processed/features.parquet
COPY reports/               reports/

# ── Porta (8501 padrão Streamlit; PaaS sobrescreve $PORT em runtime) ───────────
EXPOSE 8501

ENV PORT=8501

# Credenciais passadas em runtime via --env-file ou -e (nunca baked na imagem)
# DAGSHUB_USER, DAGSHUB_REPO, DAGSHUB_TOKEN, SUPABASE_URL, SUPABASE_KEY

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/_stcore/health || exit 1

CMD ["sh", "-c", \
     "python3 -m streamlit run app/streamlit_app.py \
      --server.port=${PORT} \
      --server.address=0.0.0.0 \
      --server.headless=true \
      --browser.gatherUsageStats=false"]