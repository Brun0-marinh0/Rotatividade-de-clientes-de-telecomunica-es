# Telecom Churn Predictor

Projeto final de infraestrutura em nuvem para ciencia de dados.

## Visao geral

Este projeto resolve o problema de `churn` em telecomunicacoes: prever se um cliente vai cancelar o servico com base no perfil de contrato, uso e cobranca.

O objetivo de negocio e permitir acoes proativas de retencao, priorizando clientes de maior risco antes da perda acontecer.

## Problema escolhido

O dataset modela clientes de uma operadora de telecom. A variavel alvo e binaria:

- `1` = cliente cancelou (`Churn`)
- `0` = cliente retido

O caso e adequado para classificacao binaria com classes desbalanceadas. Por isso, a comparacao dos modelos usa principalmente `ROC-AUC`, complementada por `recall`, `F1` e `accuracy`.

## Arquitetura adotada

```mermaid
flowchart LR
    A[Supabase / dados brutos] --> B[src/ingestion.py]
    B --> C[data/raw/telco_churn.parquet]
    C --> D[src/preprocessing.py / DuckDB]
    D --> E[data/processed/features.parquet]
    E --> F[src/prepare_data.py]
    F --> G[data/processed/train|val|test.parquet]
    F --> H[models/preprocessors/preprocessor.joblib]
    G --> I[src/train.py]
    I --> J[MLflow / DagsHub Model Registry]
    I --> K[models/trained/*.joblib]
    I --> L[reports/*]
    J --> M[Streamlit runtime]
    H --> M
    E --> M
```

### Componentes principais

- `Supabase` e a fonte operacional dos dados.
- `DuckDB` faz a engenharia das features de forma declarativa e reproduzivel.
- `DVC` orquestra o pipeline local e versiona os artefatos gerados.
- `MLflow` no `DagsHub` registra metricas, artefatos e o modelo champion.
- `Streamlit` entrega a interface de previsao e exploracao.

## Pipeline

O pipeline completo esta descrito em `dvc.yaml` e tem 5 etapas:

1. `ingest`
   - Busca os dados na tabela `telco_churn` do Supabase.
   - Gera `data/raw/telco_churn.parquet`.
2. `preprocess`
   - Aplica a engenharia de features com DuckDB.
   - Gera `data/processed/features.parquet`.
3. `prepare`
   - Faz split 70/15/15.
   - Ajusta o `ColumnTransformer`.
   - Gera `train`, `val`, `test` e `preprocessor.joblib`.
4. `train`
   - Treina `Logistic Regression`, `Random Forest` e `XGBoost`.
   - Registra metricas e plots no MLflow.
   - Promove o melhor modelo para o Model Registry com o alias `champion`.
5. `evaluate`
   - Avalia os modelos no conjunto de teste.
   - Gera relatorios comparativos em `reports/`.

## Estrutura do projeto

```text
app/streamlit_app.py        Interface Streamlit
src/ingestion.py            Ingestao a partir do Supabase
src/preprocessing.py        Feature engineering com DuckDB
src/prepare_data.py         Split e preprocessor sklearn
src/train.py                Treino e registro no MLflow/DagsHub
src/evaluate.py             Avaliacao no teste e relatorios
dvc.yaml                    Orquestracao do pipeline
Dockerfile                  Imagem de deploy da app
requirements.txt            Dependencias completas do projeto
requirements-app.txt        Dependencias minimas da app
```

## Como reproduzir o pipeline

### 1. Requisitos

- Python 3.11+
- `git`
- `dvc`
- acesso ao Supabase do projeto
- acesso ao DagsHub / MLflow do projeto

### 2. Criar o ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configurar variaveis de ambiente

Crie um arquivo `.env` com pelo menos:

```env
SUPABASE_URL=...
SUPABASE_KEY=...
DAGSHUB_USER=RafaelRailton
DAGSHUB_REPO=rotatividade-de-clientes-de-telecomunicacoes
DAGSHUB_TOKEN=...
```

### 4. Reproduzir tudo com DVC

```bash
dvc repro
```

Se quiser executar etapa por etapa:

```bash
python3 src/ingestion.py
python3 src/preprocessing.py
python3 src/prepare_data.py
python3 src/train.py
python3 src/evaluate.py
```

### 5. Abrir os artefatos gerados

- `data/raw/telco_churn.parquet`
- `data/processed/features.parquet`
- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/test.parquet`
- `models/preprocessors/preprocessor.joblib`
- `models/trained/*.joblib`
- `reports/*.png`
- `reports/model_comparison.md`

## Como rodar a aplicacao

### Modo local

```bash
pip install -r requirements-app.txt
streamlit run app/streamlit_app.py
```

### Com Docker

```bash
docker build -t telecom-churn-predictor .
docker run --rm -p 8501:8501 telecom-churn-predictor
```

## Decisoes tecnicas

### 1. Modelagem orientada a ROC-AUC

O problema e desbalanceado. `ROC-AUC` foi usado para selecionar o melhor modelo porque mede separacao entre classes sem depender de um limiar fixo.

### 2. Engenharia de features com DuckDB

As transformacoes derivadas do dataset bruto ficam centralizadas em SQL via DuckDB. Isso deixa o fluxo mais legivel, auditavel e facil de reproduzir.

### 3. `ColumnTransformer` para o preprocessamento

O pipeline de features separa:

- numericas com `StandardScaler`
- categoricas com `OneHotEncoder(handle_unknown="ignore")`
- binarias em `passthrough`

Isso evita inconsistencias entre treino e inferencia.

### 4. Registro do modelo no MLflow/DagsHub

Cada treinamento gera um run no MLflow. O melhor modelo e promovido no Registry com o alias `champion`, o que permite deploy desacoplado do treino.

### 5. Deploy sem empacotar artefatos pesados

O `Dockerfile` copia apenas codigo e dependencias. Modelos, dados processados e relatorios nao sao embutidos na imagem:

- reduz o tamanho do container
- evita artefatos obsoletos
- força reproducibilidade via pipeline

### 6. Runtime resiliente na interface

A interface Streamlit carrega o modelo do DagsHub/MLflow em runtime. O preprocessor e reconstituido localmente com o mesmo contrato de features do treino, para evitar depender de binarios pesados no deploy.

### 7. Aba de exploracao pronta para deploy

A aba de exploracao usa um resumo precomputado dos dados para manter a aplicacao responsiva mesmo sem embutir os arquivos processados no container.

## Resultados esperados

Ao final da reproducao, voce deve ter:

- um modelo champion registrado no MLflow Model Registry
- metricas e artefatos visuais no DagsHub
- os dados e splits reconstruidos em `data/processed/`
- a interface Streamlit pronta para previsao e exploracao

## Observacoes

- Os arquivos em `data/processed/`, `models/` e `reports/` sao gerados pelo pipeline e ficam ignorados no Git.
- Se voce for adaptar o projeto para outro workspace, atualize as variaveis `DAGSHUB_*` e `SUPABASE_*`.
- O app depende de acesso de rede para carregar o modelo registrado no DagsHub/MLflow.
