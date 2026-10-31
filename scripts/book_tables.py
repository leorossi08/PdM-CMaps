#!/usr/bin/env python3
"""Generator for the Complete Scientific Treatise Book.
Outputs:
  - docs/learn.md (Markdown edition)
  - docs/learn.tex (LaTeX edition)
  - docs/learn.pdf (Compiled PDF)
"""

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

def load_tables():
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

    # MD Tables
    bench_md = "| Modelo | Família | $R^2$ Hold-out | RMSE (Ciclos) | MAE (Ciclos) | NASA Score | Previsões Atrasadas (%) | Tempo Ajuste (s) |\n"
    bench_md += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for _, row in bench_df.sort_values(by="R2", ascending=False).iterrows():
        m = row["Model"]
        fam = families.get(m, "ML")
        bold = "**" if m == "XGBoost" else ""
        bench_md += f"| {bold}{m}{bold} | {fam} | {bold}{row['R2']:.4f}{bold} | {row['RMSE']:.2f} | {row['MAE']:.2f} | {row['NASA_Score']:.1f} | {row['Late_Pred_Pct']:.1f}% | {row['Fit_Time_s']:.2f}s |\n"

    hyp_md = "| ID | Fenômeno Investigado | Teste Estatístico | Estatística | p-valor | Decisão ($\alpha=0.05$) | Tamanho do Efeito |\n"
    hyp_md += "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |\n"
    for _, r in hyp_df.iterrows():
        p_str = f"**{r['p_value']}**" if "REJEITADA" in str(r['H0_Rejected']) else f"{r['p_value']}"
        hyp_md += f"| **{r['ID']}** | {r['Description']} | {r['Test']} | {r['Statistic']:.2f} | {p_str} | {r['H0_Rejected']} | {r['Effect_Size']:.3f} |\n"

    boot_md = "| Modelo | $R^2$ Médio [IC 95%] | RMSE Médio [IC 95%] | MAE Médio [IC 95%] |\n"
    boot_md += "| :--- | :---: | :---: | :---: |\n"
    for _, r in boot_df.iterrows():
        boot_md += f"| **{r['Model']}** | {r['R2_Mean']:.4f} [{r['R2_CI_Lower']:.4f}, {r['R2_CI_Upper']:.4f}] | {r['RMSE_Mean']:.2f} [{r['RMSE_CI_Lower']:.2f}, {r['RMSE_CI_Upper']:.2f}] | {r['MAE_Mean']:.2f} [{r['MAE_CI_Lower']:.2f}, {r['MAE_CI_Upper']:.2f}] |\n"

    # LaTeX Tables
    def esc(s):
        return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")

    bench_tex = r"""\begin{table}[H]
\centering
\small
\begin{tabularx}{\textwidth}{l l c c c c c r}
\toprule
\textbf{Modelo} & \textbf{Família} & $\mathbf{R^2}$ & \textbf{RMSE} & \textbf{MAE} & \textbf{NASA Score} & \textbf{Late \%} & \textbf{Tempo} \\
\midrule
"""
    for _, row in bench_df.sort_values(by="R2", ascending=False).iterrows():
        m = row["Model"]
        fam = families.get(m, "ML")
        m_esc = esc(m)
        fam_esc = esc(fam)
        if m == "XGBoost":
            bench_tex += f"\\textbf{{{m_esc}}} & \\textbf{{{fam_esc}}} & \\textbf{{{row['R2']:.4f}}} & \\textbf{{{row['RMSE']:.2f}}} & \\textbf{{{row['MAE']:.2f}}} & {row['NASA_Score']:.0f} & {row['Late_Pred_Pct']:.1f}\\% & {row['Fit_Time_s']:.2f}s \\\\\n"
        else:
            bench_tex += f"{m_esc} & {fam_esc} & {row['R2']:.4f} & {row['RMSE']:.2f} & {row['MAE']:.2f} & {row['NASA_Score']:.0f} & {row['Late_Pred_Pct']:.1f}\\% & {row['Fit_Time_s']:.2f}s \\\\\n"
    bench_tex += r"""\bottomrule
\end{tabularx}
\caption{Benchmark Completo de RUL na Frota Hold-out (62 Motores, 15.890 ciclos avaliados).}
\label{tab:rul_holdout_benchmark}
\end{table}
"""

    hyp_tex = r"""\begin{table}[H]
\centering
\scriptsize
\begin{tabularx}{\textwidth}{l X l r r c}
\toprule
\textbf{ID} & \textbf{Fenômeno Investigado} & \textbf{Teste Estatístico} & \textbf{Estatística} & \textbf{p-valor} & \textbf{Decisão} \\
\midrule
"""
    for _, r in hyp_df.iterrows():
        id_esc = esc(str(r['ID']))
        desc_esc = esc(str(r['Description']))
        test_esc = esc(str(r['Test']))
        p_val = str(r['p_value'])
        decisao = "REJEITADA" if "REJEITADA" in str(r['H0_Rejected']) else "NÃO REJ."
        stat_val = f"{r['Statistic']:.2f}" if abs(r['Statistic']) < 1e5 else f"{r['Statistic']:.1e}"
        hyp_tex += f"\\textbf{{{id_esc}}} & {desc_esc} & {test_esc} & {stat_val} & \\textbf{{{p_val}}} & {decisao} \\\\\n"
    hyp_tex += r"""\bottomrule
\end{tabularx}
\caption{Matriz de Testes de Hipóteses com P-Valores Exatos e Conclusões Científicas.}
\label{tab:thesis_hypothesis_tests}
\end{table}
"""

    boot_tex = r"""\begin{table}[H]
\centering
\small
\begin{tabularx}{\textwidth}{l c c c}
\toprule
\textbf{Modelo} & $\mathbf{R^2}$ \textbf{Médio [IC 95\%]} & \textbf{RMSE Médio [IC 95\%]} & \textbf{MAE Médio [IC 95\%]} \\
\midrule
"""
    for _, r in boot_df.iterrows():
        m_esc = esc(str(r['Model']))
        boot_tex += f"\\textbf{{{m_esc}}} & {r['R2_Mean']:.4f} [{r['R2_CI_Lower']:.4f}, {r['R2_CI_Upper']:.4f}] & {r['RMSE_Mean']:.2f} [{r['RMSE_CI_Lower']:.2f}, {r['RMSE_CI_Upper']:.2f}] & {r['MAE_Mean']:.2f} [{r['MAE_CI_Lower']:.2f}, {r['MAE_CI_Upper']:.2f}] \\\\\n"
    boot_tex += r"""\bottomrule
\end{tabularx}
\caption{Intervalos de Confiança Não-Paramétricos de 95\% via Bootstrap na Frota ($B=500$).}
\label{tab:thesis_bootstrap_ci}
\end{table}
"""

    return {
        "bench_md": bench_md,
        "hyp_md": hyp_md,
        "boot_md": boot_md,
        "bench_tex": bench_tex,
        "hyp_tex": hyp_tex,
        "boot_tex": boot_tex,
    }

print("Loaded table loader.")
