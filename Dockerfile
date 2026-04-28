FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG DAGSHUB_USER
ARG DAGSHUB_REPO
ARG DAGSHUB_TOKEN

ENV DAGSHUB_USER=$DAGSHUB_USER
ENV DAGSHUB_REPO=$DAGSHUB_REPO
ENV DAGSHUB_TOKEN=$DAGSHUB_TOKEN

RUN python src/evaluate.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]