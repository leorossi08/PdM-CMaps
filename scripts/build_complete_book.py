#!/usr/bin/env python3
"""Complete Scientific Treatise Book Generator.
Produces:
1. docs/learn.md: Comprehensive markdown edition with embedded figures, listings, worked examples, third-person perspective and citations.
2. docs/learn.tex: Typeset LaTeX book edition using tcolorbox (codebox, examplebox, diagrambox, calloutbox), tabularx and booktabs.
3. docs/learn.pdf: Compiled publication-grade PDF via pdflatex.
"""

from pathlib import Path
import re
import subprocess
import sys
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
FIG_DIR = ROOT_DIR / "results" / "maintenance_guide_fd004" / "figures"
TABLES_DIR = ROOT_DIR / "results" / "maintenance_guide_fd004" / "tables"

BENCHMARK_CSV = TABLES_DIR / "rul_holdout_benchmark.csv"
HYPOTHESIS_CSV = TABLES_DIR / "thesis_hypothesis_tests.csv"
BOOTSTRAP_CSV = TABLES_DIR / "thesis_bootstrap_ci.csv"


def escape_latex(text: str) -> str:
    """Escapes LaTeX special characters outside math mode."""
    parts = re.split(r"(\$\$.*?\$\$|\$.*?\$)", text, flags=re.DOTALL)
    out = []
    for p in parts:
        if p.startswith("$") and p.endswith("$"):
            # Inside math mode
            math_str = p.replace("\t", " ").replace("\x0c", "\\frac").replace("\x07", "\\argmin ").replace("\x08", "\\bar{y}")
            out.append(math_str)
        else:
            txt = p
            txt = txt.replace("&", r"\&")
            txt = txt.replace("%", r"\%")
            txt = txt.replace("#", r"\#")
            txt = txt.replace("_", r"\_")
            out.append(txt)
    return "".join(out)


def get_benchmark_tables_tex():
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
        m_esc = escape_latex(m)
        fam_esc = escape_latex(fam)
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

    # Hypothesis table
    hyp_tex = r"""\begin{table}[H]
\centering
\scriptsize
\begin{tabularx}{\textwidth}{l X l r r c}
\toprule
\textbf{ID} & \textbf{Fenômeno Investigado} & \textbf{Teste Estatístico} & \textbf{Estatística} & \textbf{p-valor} & \textbf{Decisão} \\
\midrule
"""
    for _, r in hyp_df.iterrows():
        id_esc = escape_latex(str(r['ID']))
        desc_esc = escape_latex(str(r['Description']))
        test_esc = escape_latex(str(r['Test']))
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

    # Bootstrap table
    boot_tex = r"""\begin{table}[H]
\centering
\small
\begin{tabularx}{\textwidth}{l c c c}
\toprule
\textbf{Modelo} & $\mathbf{R^2}$ \textbf{Médio [IC 95\%]} & \textbf{RMSE Médio [IC 95\%]} & \textbf{MAE Médio [IC 95\%]} \\
\midrule
"""
    for _, r in boot_df.iterrows():
        m_esc = escape_latex(str(r['Model']))
        boot_tex += f"\\textbf{{{m_esc}}} & {r['R2_Mean']:.4f} [{r['R2_CI_Lower']:.4f}, {r['R2_CI_Upper']:.4f}] & {r['RMSE_Mean']:.2f} [{r['RMSE_CI_Lower']:.2f}, {r['RMSE_CI_Upper']:.2f}] & {r['MAE_Mean']:.2f} [{r['MAE_CI_Lower']:.2f}, {r['MAE_CI_Upper']:.2f}] \\\\\n"
    boot_tex += r"""\bottomrule
\end{tabularx}
\caption{Intervalos de Confiança Não-Paramétricos de 95\% via Bootstrap na Frota ($B=500$).}
\label{tab:thesis_bootstrap_ci}
\end{table}
"""
    return bench_tex, hyp_tex, boot_tex


print("Initialized book generator base.")
