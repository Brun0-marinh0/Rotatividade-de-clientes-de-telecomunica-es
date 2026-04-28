"""
app.py — Interface Streamlit para previsão de churn de clientes.

Execute com:
    streamlit run app.py

Ou via Docker:
    docker build --build-arg DAGSHUB_USER=... -t churn-app .
    docker run -p 8501:8501 churn-app
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Previsão de Churn",
    page_icon="📱",
    layout="centered",
)

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("📱 Previsão de Churn de Clientes")
st.markdown(
    "Classifica clientes de telecomunicação como **Churn (Risco)** ou **Retenção** "
    "usando um modelo Random Forest treinado e rastreado no **DagsHub**."
)

st.divider()

st.subheader("Prever churn de clientes (CSV)")

uploaded_file = st.file_uploader(
    "Selecione o arquivo CSV com os dados dos clientes",
    type=["csv"],
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        st.error(f"O CSV precisa ter as colunas: {', '.join(required_cols)}")
    else:
        X = df.drop(columns=["customerID", "Churn"], errors="ignore")
        
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        
        X = X.select_dtypes(include=["number"])
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)
        
        df["previsao"] = model.predict(X)
        df["prob_churn"] = model.predict_proba(X)[:, 1]

        st.success(f"{len(df)} clientes classificados!")
        st.dataframe(df, use_container_width=True)

        contagem = df["previsao"].value_counts()
        col1, col2 = st.columns(2)
        col1.metric("Churn (Risco)", contagem.get(1, 0))
        col2.metric("Retenção", contagem.get(0, 0))

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Baixar resultado como CSV",
            data=csv_bytes,
            file_name="resultado_churn.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.divider()
st.caption(
    "Modelo: Random Forest · "
    "Rastreamento: MLflow + DagsHub · "
    "Deploy: Docker + Render"
)