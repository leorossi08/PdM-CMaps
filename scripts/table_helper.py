#!/usr/bin/env python3
"""Build publication-quality Scientific Treatise Book in Markdown, LaTeX and PDF."""

import os
import re
import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT_DIR / "results" / "maintenance_guide_fd004" / "figures"
TABLES_DIR = ROOT_DIR / "results" / "maintenance_guide_fd004" / "tables"

BENCHMARK_CSV = TABLES_DIR / "rul_holdout_benchmark.csv"
HYPOTHESIS_CSV = TABLES_DIR / "thesis_hypothesis_tests.csv"
BOOTSTRAP_CSV = TABLES_DIR / "thesis_bootstrap_ci.csv"

def get_markdown_tables():
    bench_df = pd.read_csv(BENCHMARK_CSV)
    hyp_df = pd.read_csv(HYPOTHESIS_CSV)
    boot_df = pd.read_csv(BOOTSTRAP_CSV)

    families = {
        "Baseline (Mean RUL)": "Linha de Base",
        "Ridge": "Linear Regularizado",
        "Support Vector Machine (SVM)": "Kernel / Linear SVR",
        "MiniRocket + RidgeCV": "Convolução Determinística",
        "Extra Trees": "Ensemble Bagging",
        "CatBoost": "GBDT Oblivious",
        "PyTorch Transformer": "Deep Learning Sequencial",
        "PyTorch RNN (LSTM)": "Deep Learning Recorrente",
        "PyTorch CNN": "Deep Learning Convolucional 1D",
        "MLP": "Deep Learning Feedforward",
        "HistGradientBoosting": "GBDT Histograma",
        "Random Forest": "Ensemble Bagging",
        "LightGBM": "GBDT Leaf-wise",
        "XGBoost": "GBDT DMatrix Exact"
    }

    # Benchmark table
    b_rows = ["| Modelo | Família | $R^2$ Hold-out | RMSE (Ciclos) | MAE (Ciclos) | NASA Score | Previsões Atrasadas (%) | Tempo Ajuste (s) |",
              "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for _, row in bench_df.sort_values(by="R2", ascending=False).iterrows():
        m = row["Model"]
        fam = families.get(m, "ML")
        bold = "**" if m == "XGBoost" else ""
        b_rows.append(f"| {bold}{m}{bold} | {fam} | {bold}{row['R2']:.4f}{bold} | {row['RMSE']:.2f} | {row['MAE']:.2f} | {row['NASA_Score']:.1f} | {row['Late_Pred_Pct']:.1f}% | {row['Fit_Time_s']:.2f}s |")
    bench_md = "\n".join(b_rows)

    # Hypothesis table
    h_rows = ["| Hipótese | Fenômeno Investigado | Teste Estatístico | Estatística | p-valor | Decisão ($\alpha=0.05$) | Tamanho do Efeito |",
              "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |"]
    for _, r in hyp_df.iterrows():
        p_str = f"**{r['p_value']}**" if "REJEITADA" in str(r['H0_Rejected']) else f"{r['p_value']}"
        h_rows.append(f"| **{r['ID']}** | {r['Description']} | {r['Test']} | {r['Statistic']:.2f} | {p_str} | {r['H0_Rejected']} | {r['Effect_Size']:.3f} |")
    hyp_md = "\n".join(h_rows)

    # Bootstrap table
    bt_rows = ["| Modelo | $R^2$ Médio [IC 95%] | RMSE Médio [IC 95%] | MAE Médio [IC 95%] |",
               "| :--- | :---: | :---: | :---: |"]
    for _, r in boot_df.iterrows():
        bt_rows.append(f"| **{r['Model']}** | {r['R2_Mean']:.4f} [{r['R2_CI_Lower']:.4f}, {r['R2_CI_Upper']:.4f}] | {r['RMSE_Mean']:.2f} [{r['RMSE_CI_Lower']:.2f}, {r['RMSE_CI_Upper']:.2f}] | {r['MAE_Mean']:.2f} [{r['MAE_CI_Lower']:.2f}, {r['MAE_CI_Upper']:.2f}] |")
    boot_md = "\n".join(bt_rows)

    return bench_md, hyp_md, boot_md

print("Loaded table generator successfully.")
