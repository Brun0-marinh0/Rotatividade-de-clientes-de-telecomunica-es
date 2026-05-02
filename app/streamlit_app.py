"""
streamlit_app.py — Interface de previsão de churn com exploração de dados e métricas do modelo.
    streamlit run app/streamlit_app.py
"""

import sys
import os
from pathlib import Path

# Garante que o root do projeto esteja no path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

# ── Constantes ────────────────────────────────────────────────────────────────

NUMERIC_FEATS = [
    "tenure", "monthly_charges", "total_charges",
    "avg_monthly_actual", "monthly_delta", "num_addon_services",
]
OHE_FEATS    = ["MultipleLines", "InternetService", "Contract", "PaymentMethod"]
BINARY_FEATS = [
    "SeniorCitizen", "gender_male", "has_partner", "has_dependents",
    "has_phone", "paperless_billing", "auto_payment",
    "has_online_security", "has_online_backup", "has_device_protection",
    "has_tech_support", "has_streaming_tv", "has_streaming_movies",
]
ALL_FEATS = NUMERIC_FEATS + OHE_FEATS + BINARY_FEATS

# ── Carregamento de recursos ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Carregando modelo…")
def load_model():
    """Tenta registry MLflow; fallback para .joblib local."""
    import mlflow
    import mlflow.sklearn
    dagshub_user  = os.getenv("DAGSHUB_USER")
    dagshub_repo  = os.getenv("DAGSHUB_REPO")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    try:
        mlflow.set_tracking_uri(
            f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"
        )
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        model = mlflow.sklearn.load_model("models:/TelcoChurnClassifier@champion")
        return model, "MLflow Registry (champion)"
    except Exception:
        model = joblib.load("models/trained/random_forest.joblib")
        return model, "local (random_forest.joblib)"


@st.cache_resource(show_spinner=False)
def load_preprocessor():
    return joblib.load("models/preprocessors/preprocessor.joblib")


@st.cache_data(show_spinner=False)
def load_features_data():
    return pd.read_parquet("data/processed/features.parquet")


# ── Feature engineering (espelha a SQL do DuckDB) ────────────────────────────

def engineer_features(inp: dict) -> pd.DataFrame:
    tenure  = max(inp["tenure"], 1)
    monthly = inp["monthly_charges"]
    total   = inp["total_charges"]

    addon_services = sum([
        inp["OnlineSecurity"]   == "Yes",
        inp["OnlineBackup"]     == "Yes",
        inp["DeviceProtection"] == "Yes",
        inp["TechSupport"]      == "Yes",
        inp["StreamingTV"]      == "Yes",
        inp["StreamingMovies"]  == "Yes",
    ])

    row = {
        "tenure":               inp["tenure"],
        "monthly_charges":      monthly,
        "total_charges":        total,
        "avg_monthly_actual":   total / tenure,
        "monthly_delta":        monthly - total / tenure,
        "num_addon_services":   addon_services,
        "MultipleLines":        inp["MultipleLines"],
        "InternetService":      inp["InternetService"],
        "Contract":             inp["Contract"],
        "PaymentMethod":        inp["PaymentMethod"],
        "SeniorCitizen":        int(inp["SeniorCitizen"]),
        "gender_male":          1 if inp["gender"] == "Male" else 0,
        "has_partner":          1 if inp["Partner"] == "Yes" else 0,
        "has_dependents":       1 if inp["Dependents"] == "Yes" else 0,
        "has_phone":            1 if inp["PhoneService"] == "Yes" else 0,
        "paperless_billing":    1 if inp["PaperlessBilling"] == "Yes" else 0,
        "auto_payment":         1 if "automatic" in inp["PaymentMethod"].lower() else 0,
        "has_online_security":  1 if inp["OnlineSecurity"]   == "Yes" else 0,
        "has_online_backup":    1 if inp["OnlineBackup"]     == "Yes" else 0,
        "has_device_protection":1 if inp["DeviceProtection"] == "Yes" else 0,
        "has_tech_support":     1 if inp["TechSupport"]      == "Yes" else 0,
        "has_streaming_tv":     1 if inp["StreamingTV"]      == "Yes" else 0,
        "has_streaming_movies": 1 if inp["StreamingMovies"]  == "Yes" else 0,
    }
    return pd.DataFrame([row])[ALL_FEATS]


# ── Gauge de probabilidade ────────────────────────────────────────────────────

def prob_gauge(prob: float) -> go.Figure:
    color = "#e74c3c" if prob >= 0.5 else "#27ae60"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0, 40],  "color": "#d5f5e3"},
                {"range": [40, 60], "color": "#fef9e7"},
                {"range": [60, 100],"color": "#fde8e8"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75, "value": 50,
            },
        },
        title={"text": "Probabilidade de Churn", "font": {"size": 16}},
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
    return fig


# ── Contribuição das features ─────────────────────────────────────────────────

def feature_contribution(model, preprocessor, df_input: pd.DataFrame) -> pd.Series:
    X = preprocessor.transform(df_input)
    feat_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return pd.Series(dtype=float)

    contributions = pd.Series(
        importances * np.abs(X[0]),
        index=feat_names,
    ).sort_values(ascending=False).head(10)

    # Limpa prefixos do ColumnTransformer
    contributions.index = (
        contributions.index
        .str.replace(r"^(num__|bin__|cat__|remainder__)", "", regex=True)
    )
    return contributions


# ─────────────────────────────────────────────────────────────────────────────
# Layout principal
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# Sidebar
with st.sidebar:
    st.title("📡 Churn Predictor")
    st.markdown("**Telecomunicações — Classificação Binária**")
    st.divider()
    model, model_source = load_model()
    preprocessor = load_preprocessor()
    st.success(f"Modelo carregado\n\n`{model_source}`")
    st.divider()
    st.markdown(
        "**Métricas (holdout 15%)**\n\n"
        "| Métrica | Valor |\n|---|---|\n"
        "| ROC-AUC | 0.8173 |\n"
        "| Recall (churn) | 0.7107 |\n"
        "| F1 (weighted) | 0.7678 |\n"
        "| Accuracy | 0.7583 |"
    )
    st.divider()
    st.caption("Pipeline: Supabase → DuckDB → sklearn → MLflow → DVC")

tab_pred, tab_eda, tab_model = st.tabs(["🔮 Previsão", "📊 Exploração", "🧠 Modelo"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREVISÃO
# ─────────────────────────────────────────────────────────────────────────────

with tab_pred:
    st.header("Previsão de Churn")
    st.markdown("Preencha o perfil do cliente e clique em **Prever**.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        # ── Perfil ──────────────────────────────────────────────────────────
        with c1:
            st.subheader("👤 Perfil")
            gender        = st.selectbox("Gênero", ["Male", "Female"])
            senior        = st.checkbox("Cliente sênior (≥ 65 anos)")
            partner       = st.selectbox("Tem parceiro(a)?", ["No", "Yes"])
            dependents    = st.selectbox("Tem dependentes?", ["No", "Yes"])
            tenure        = st.slider("Meses como cliente", 0, 72, 12)

        # ── Serviços ─────────────────────────────────────────────────────────
        with c2:
            st.subheader("📦 Serviços")
            phone_service = st.selectbox("Serviço de telefone", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Múltiplas linhas",
                ["No", "Yes", "No phone service"],
                disabled=(phone_service == "No"),
            )
            internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
            inet_opts = ["Yes", "No", "No internet service"]
            online_sec  = st.selectbox("Segurança online",    inet_opts, disabled=(internet == "No"))
            online_bkp  = st.selectbox("Backup online",       inet_opts, disabled=(internet == "No"))
            dev_prot    = st.selectbox("Proteção dispositivo", inet_opts, disabled=(internet == "No"))
            tech_sup    = st.selectbox("Suporte técnico",     inet_opts, disabled=(internet == "No"))
            stream_tv   = st.selectbox("Streaming TV",        inet_opts, disabled=(internet == "No"))
            stream_mv   = st.selectbox("Streaming filmes",    inet_opts, disabled=(internet == "No"))

        # ── Contrato e Pagamento ──────────────────────────────────────────────
        with c3:
            st.subheader("💳 Contrato")
            contract   = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
            paperless  = st.selectbox("Fatura sem papel", ["Yes", "No"])
            payment    = st.selectbox("Forma de pagamento", [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ])
            st.divider()
            monthly_charges = st.number_input("Cobrança mensal (R$)", 0.0, 200.0, 65.0, 0.5)
            total_charges   = st.number_input(
                "Cobrança total (R$)", 0.0, 10000.0,
                float(monthly_charges * max(tenure, 1)), 1.0,
            )

        submitted = st.form_submit_button("🔮 Prever Churn", type="primary", use_container_width=True)

    # ── Resultado ─────────────────────────────────────────────────────────────
    if submitted:
        # Se internet = No, força valores de serviços para "No internet service"
        if internet == "No":
            online_sec = online_bkp = dev_prot = tech_sup = stream_tv = stream_mv = "No internet service"
        if phone_service == "No":
            multiple_lines = "No phone service"

        raw_input = {
            "gender": gender, "SeniorCitizen": int(senior),
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet, "OnlineSecurity": online_sec,
            "OnlineBackup": online_bkp, "DeviceProtection": dev_prot,
            "TechSupport": tech_sup, "StreamingTV": stream_tv,
            "StreamingMovies": stream_mv, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "monthly_charges": monthly_charges, "total_charges": total_charges,
        }

        df_input   = engineer_features(raw_input)
        X_proc     = preprocessor.transform(df_input)
        pred       = int(model.predict(X_proc)[0])
        prob_churn = float(model.predict_proba(X_proc)[0][1])

        st.divider()
        res_col, gauge_col, contrib_col = st.columns([1.2, 1.5, 1.3])

        with res_col:
            if pred == 1:
                st.error("### ⚠️ ALTO RISCO DE CHURN")
                st.markdown(
                    f"Este cliente tem **{prob_churn:.1%}** de probabilidade "
                    "de cancelar o serviço."
                )
                st.markdown("**Ações sugeridas:**")
                st.markdown("- Oferecer desconto de fidelidade\n- Migrar para contrato anual\n- Contato proativo de retenção")
            else:
                st.success("### ✅ CLIENTE RETIDO")
                st.markdown(
                    f"Este cliente tem **{1-prob_churn:.1%}** de probabilidade "
                    "de permanecer."
                )
                st.markdown("**Perfil:**")
                st.markdown("- Baixo risco de cancelamento\n- Monitorar métricas de satisfação")

            st.metric("Prob. Churn",  f"{prob_churn:.1%}")
            st.metric("Prob. Retenção", f"{1-prob_churn:.1%}")

        with gauge_col:
            st.plotly_chart(prob_gauge(prob_churn), use_container_width=True)

        with contrib_col:
            st.markdown("**Top fatores de influência**")
            contrib = feature_contribution(model, preprocessor, df_input)
            if len(contrib):
                fig = px.bar(
                    contrib.reset_index(),
                    x=contrib.values, y=contrib.index,
                    orientation="h",
                    color=contrib.values,
                    color_continuous_scale=["#27ae60", "#e74c3c"],
                    labels={"x": "Peso", "index": "Feature"},
                )
                fig.update_layout(
                    height=300, showlegend=False,
                    coloraxis_showscale=False,
                    margin=dict(t=10, b=10, l=10, r=10),
                    yaxis={"categoryorder": "total ascending"},
                )
                st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — EXPLORAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

with tab_eda:
    st.header("Exploração dos Dados")

    df = load_features_data()
    df["Churn_label"] = df["churn"].map({1: "Churn", 0: "Retido"})

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total de clientes",  f"{len(df):,}")
    kpi2.metric("Taxa de churn",       f"{df['churn'].mean():.1%}")
    kpi3.metric("Tenure médio",        f"{df['tenure'].mean():.1f} meses")
    kpi4.metric("Cobrança média/mês",  f"R$ {df['monthly_charges'].mean():.2f}")

    st.divider()
    row1_l, row1_r = st.columns(2)

    with row1_l:
        fig = px.pie(
            df["Churn_label"].value_counts().reset_index(),
            names="Churn_label", values="count",
            color="Churn_label",
            color_discrete_map={"Churn": "#e74c3c", "Retido": "#27ae60"},
            title="Distribuição de Churn",
            hole=0.4,
        )
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    with row1_r:
        fig = px.histogram(
            df, x="tenure", color="Churn_label",
            color_discrete_map={"Churn": "#e74c3c", "Retido": "#27ae60"},
            barmode="overlay", opacity=0.7, nbins=36,
            title="Distribuição de Tempo de Contrato (meses)",
            labels={"tenure": "Meses como cliente", "count": "Clientes"},
        )
        fig.update_layout(height=320, legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    row2_l, row2_r = st.columns(2)

    with row2_l:
        fig = px.box(
            df, x="Churn_label", y="monthly_charges",
            color="Churn_label",
            color_discrete_map={"Churn": "#e74c3c", "Retido": "#27ae60"},
            title="Cobrança Mensal por Status",
            labels={"monthly_charges": "R$/mês", "Churn_label": ""},
            points="outliers",
        )
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with row2_r:
        ct = (
            df.groupby(["Contract", "Churn_label"])
            .size().reset_index(name="n")
        )
        ct["pct"] = ct.groupby("Contract")["n"].transform(lambda x: x / x.sum())
        fig = px.bar(
            ct, x="Contract", y="pct", color="Churn_label",
            color_discrete_map={"Churn": "#e74c3c", "Retido": "#27ae60"},
            barmode="stack",
            title="Churn por Tipo de Contrato",
            labels={"pct": "Proporção", "Contract": "Contrato", "Churn_label": ""},
            text_auto=".0%",
        )
        fig.update_layout(height=320, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    # Serviços adicionais vs churn
    st.subheader("Adoção de Serviços por Status")
    service_cols = [
        "has_online_security", "has_online_backup", "has_device_protection",
        "has_tech_support", "has_streaming_tv", "has_streaming_movies",
    ]
    service_labels = [
        "Seg. Online", "Backup", "Prot. Dispositivo",
        "Suporte Téc.", "Stream. TV", "Stream. Filmes",
    ]
    svc_churn  = df[df["churn"] == 1][service_cols].mean()
    svc_retain = df[df["churn"] == 0][service_cols].mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Churn",   x=service_labels, y=svc_churn,  marker_color="#e74c3c"))
    fig.add_trace(go.Bar(name="Retido",  x=service_labels, y=svc_retain, marker_color="#27ae60"))
    fig.update_layout(
        barmode="group", height=320,
        yaxis_tickformat=".0%",
        title="% de clientes com serviço ativo",
        legend_title="",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top correlações numéricas com churn
    st.subheader("Correlação com Churn")
    num_cols = ["tenure", "monthly_charges", "total_charges",
                "avg_monthly_actual", "monthly_delta", "num_addon_services",
                "SeniorCitizen", "has_partner", "has_phone", "paperless_billing",
                "auto_payment", "gender_male"]
    corr = df[num_cols + ["churn"]].corr()["churn"].drop("churn").sort_values()
    fig = px.bar(
        corr.reset_index(), x="churn", y="index",
        orientation="h",
        color="churn",
        color_continuous_scale=["#27ae60", "#e74c3c"],
        labels={"churn": "Correlação de Pearson", "index": "Feature"},
        title="Correlação das features numéricas/binárias com Churn",
    )
    fig.update_layout(height=380, coloraxis_showscale=False,
                      yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODELO
# ─────────────────────────────────────────────────────────────────────────────

with tab_model:
    st.header("Informações do Modelo")

    info_col, feat_col = st.columns([1, 1.5])

    with info_col:
        st.subheader("Arquitetura do Pipeline")
        st.markdown("""
        ```
        Supabase (PostgreSQL)
               ↓ ingestion.py
        data/raw/telco_churn.parquet
               ↓ preprocessing.py (DuckDB)
        data/processed/features.parquet
               ↓ prepare_data.py (sklearn)
        data/processed/{train,val,test}.parquet
               ↓ train.py (MLflow)
        TelcoChurnClassifier@champion
        ```
        """)

        st.subheader("Comparação no Teste (holdout 15%)")
        results = {
            "Modelo":          ["Logistic Regression", "Random Forest ⭐", "XGBoost"],
            "Recall (churn)":  [0.7750,  0.7107,  0.7107],
            "F1 (weighted)":   [0.7469,  0.7678,  0.7434],
            "ROC-AUC":         [0.8215,  0.8173,  0.8111],
            "Accuracy":        [0.7327,  0.7583,  0.7308],
        }
        df_res = pd.DataFrame(results).set_index("Modelo")
        st.dataframe(
            df_res.style.highlight_max(axis=0, color="#d5f5e3")
                        .format("{:.4f}"),
            use_container_width=True,
        )
        st.caption("⭐ = modelo registrado como champion no MLflow Model Registry")

    with feat_col:
        st.subheader("Importância das Features (Top 15)")
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            importances = None

        if importances is not None:
            feat_names = [
                n.replace("num__", "").replace("cat__", "").replace("remainder__", "")
                for n in preprocessor.get_feature_names_out()
            ]
            fi = (
                pd.Series(importances, index=feat_names)
                .sort_values(ascending=False)
                .head(15)
            )
            fig = px.bar(
                fi[::-1].reset_index(),
                x=fi[::-1].values, y=fi[::-1].index,
                orientation="h",
                color=fi[::-1].values,
                color_continuous_scale=["#aed6f1", "#1a5276"],
                labels={"x": "Importância", "index": "Feature"},
                title="Feature Importances — Random Forest (champion)",
            )
            fig.update_layout(height=420, coloraxis_showscale=False,
                              margin=dict(l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Features do Modelo")
    st.markdown("**23 features de entrada** geradas pelo pipeline DuckDB + sklearn:")

    feat_df = pd.DataFrame({
        "Feature": NUMERIC_FEATS + OHE_FEATS + BINARY_FEATS,
        "Tipo": (
            ["Numérico (StandardScaler)"] * len(NUMERIC_FEATS) +
            ["Categórico (OneHotEncoder)"] * len(OHE_FEATS) +
            ["Binário (passthrough)"] * len(BINARY_FEATS)
        ),
        "Origem": (
            ["DuckDB engenheirado"] * 3 +
            ["DuckDB derivado"] * 3 +
            ["Original"] * 4 +
            ["DuckDB engenheirado"] * 2 +
            ["Original"] * 4 +
            ["DuckDB engenheirado"] * 3 +
            ["Original"] * 4
        ),
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
