FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONPATH=/app

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src/   src/
COPY app/   app/

EXPOSE 8501

ENV PORT=8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/_stcore/health || exit 1

CMD ["sh", "-c", \
     "python3 -m streamlit run app/streamlit_app.py \
      --server.port=${PORT} \
      --server.address=0.0.0.0 \
      --server.headless=true \
      --browser.gatherUsageStats=false"]
