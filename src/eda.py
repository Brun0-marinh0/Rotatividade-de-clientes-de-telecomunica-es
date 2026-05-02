import os
import pandas as pd
import numpy as np

DATASET_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def main():
    df = pd.read_csv(DATASET_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    lines = []

    lines.append(f"Shape: {df.shape}")
    lines.append("")

    lines.append("dtypes (contagem):")
    lines.append(df.dtypes.value_counts().to_string())
    lines.append("")

    null_pct = (df.isnull().sum() / len(df) * 100).round(2)
    null_pct = null_pct[null_pct > 0]
    if len(null_pct) == 0:
        lines.append("% nulos: nenhuma coluna com nulos (após coerce TotalCharges):")
        raw_null = df.isnull().sum()
        raw_null = raw_null[raw_null > 0]
        lines.append(raw_null.to_string() if len(raw_null) else "  nenhum")
    else:
        lines.append("% nulos por coluna:")
        lines.append(null_pct.to_string())
    lines.append("")

    df_c = df[df["Churn"].isin(["Yes", "No"])].copy()
    df_c["Churn_bin"] = df_c["Churn"].map({"Yes": 1, "No": 0})
    counts = df_c["Churn"].value_counts()
    pcts   = df_c["Churn"].value_counts(normalize=True) * 100
    lines.append("Target (Churn):")
    for cls in ["No", "Yes"]:
        lines.append(f"  {cls}: {counts[cls]} ({pcts[cls]:.1f}%)")
    lines.append("")

    num_df = df_c.select_dtypes(include=[np.number])
    if "Churn_bin" not in num_df.columns:
        num_df["Churn_bin"] = df_c["Churn_bin"]
    corr = num_df.corr()["Churn_bin"].drop("Churn_bin").abs().sort_values(ascending=False)
    lines.append("Top-5 correlações com Churn:")
    lines.append(corr.head(5).to_string())

    output = "\n".join(lines)
    print(output)

    os.makedirs("reports", exist_ok=True)
    with open("reports/eda_summary.txt", "w") as f:
        f.write(output)
    print("\n→ reports/eda_summary.txt")


if __name__ == "__main__":
    main()
