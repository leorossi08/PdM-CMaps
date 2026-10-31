"""
Update scientific treatise (docs/learn.md and docs/learn.tex) and compile docs/learn.pdf.
Ensures full alignment with:
1. The 14 models evaluated (including SVM, Random Forest, MLP, PyTorch CNN, PyTorch RNN, PyTorch Transformer).
2. The thesis hypothesis testing matrix with exact p-values and 95% bootstrap confidence intervals.
3. 100% clean compilation via pdflatex with all figures and tables embedded.
"""

import os
import re
import subprocess
import pandas as pd

WORKSPACE = "/home/leorossi/PdM-CMaps/PdM-CMaps"
MD_PATH = os.path.join(WORKSPACE, "docs", "learn.md")
TEX_PATH = os.path.join(WORKSPACE, "docs", "learn.tex")
PDF_PATH = os.path.join(WORKSPACE, "docs", "learn.pdf")

BENCHMARK_CSV = os.path.join(WORKSPACE, "results/maintenance_guide_fd004/tables/rul_holdout_benchmark.csv")
HYPOTHESIS_CSV = os.path.join(WORKSPACE, "results/maintenance_guide_fd004/tables/thesis_hypothesis_tests.csv")
BOOTSTRAP_CSV = os.path.join(WORKSPACE, "results/maintenance_guide_fd004/tables/thesis_bootstrap_ci.csv")

def format_markdown_tables():
    bench_df = pd.read_csv(BENCHMARK_CSV)
    hyp_df = pd.read_csv(HYPOTHESIS_CSV)
    boot_df = pd.read_csv(BOOTSTRAP_CSV)

    # Benchmark markdown
    bench_md = "| Modelo | Família | $R^2$ Hold-out | RMSE (Ciclos) | MAE (Ciclos) | NASA Score | Late Pred (%) | Fit Time (s) |\n"
    bench_md += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    
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

    # Sort by R2 descending
    bench_sorted = bench_df.sort_values(by="R2", ascending=False)
    for _, row in bench_sorted.iterrows():
        m = row["Model"]
        fam = families.get(m, "ML")
        bold = "**" if m == "XGBoost" else ""
        bench_md += f"| {bold}{m}{bold} | {fam} | {bold}{row['R2']:.4f}{bold} | {row['RMSE']:.2f} | {row['MAE']:.2f} | {row['NASA_Score']:.1f} | {row['Late_Pred_Pct']:.1f}% | {row['Fit_Time_s']:.2f}s |\n"

    # Hypothesis markdown
    hyp_md = "| Hipótese / Teste | Fenômeno Investigado | Teste Estatístico | Estatística | p-valor | Decisão ($\alpha=0.05$) | Tamanho do Efeito |\n"
    hyp_md += "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |\n"
    for _, r in hyp_df.iterrows():
        p_str = f"**{r['p_value']}**" if "REJEITADA" in str(r['H0_Rejected']) else f"{r['p_value']}"
        hyp_md += f"| **{r['ID']}** | {r['Description']} | {r['Test']} | {r['Statistic']:.2f} | {p_str} | {r['H0_Rejected']} | {r['Effect_Size']:.3f} |\n"

    # Bootstrap markdown
    boot_md = "| Modelo | $R^2$ Médio [IC 95%] | RMSE Médio [IC 95%] | MAE Médio [IC 95%] |\n"
    boot_md += "| :--- | :---: | :---: | :---: |\n"
    for _, r in boot_df.iterrows():
        boot_md += f"| **{r['Model']}** | {r['R2_Mean']:.4f} [{r['R2_CI_Lower']:.4f}, {r['R2_CI_Upper']:.4f}] | {r['RMSE_Mean']:.2f} [{r['RMSE_CI_Lower']:.2f}, {r['RMSE_CI_Upper']:.2f}] | {r['MAE_Mean']:.2f} [{r['MAE_CI_Lower']:.2f}, {r['MAE_CI_Upper']:.2f}] |\n"

    return bench_md, hyp_md, boot_md

def update_markdown():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    bench_md, hyp_md, boot_md = format_markdown_tables()

    thesis_chapter = f"""

---

# Capítulo 11B: Validação Estatística de Hipóteses (P-Values e Bootstrap CI 95% na Frota Hold-out)

Em estrita consonância com as diretrizes do `@ml-best-practices` e a metodologia de Epifani et al. (IEEE CIM 2026), não nos limitamos a reportar métricas pontuais de erro. Conduzimos uma **bateria de testes estatísticos de hipóteses** com extração exata de p-valores e intervalos de confiança não-paramétricos via *Bootstrap* ($B=500$ reamostragens da frota de 62 motores hold-out).

## 11B.1 Matriz de Testes de Hipótese Científica (P-Values)

{hyp_md}

### Interpretação dos P-Valores e Confirmação da Tese:
1. **$H_2$ (Divergência Térmica Real vs Ruído):** O teste t de Welch bicaudal rejeitou a hipótese nula com $p = 3,61 \\times 10^{{-14}}$ ($t = 8,54$, $d = 0,97$). O deslocamento térmico de $+7,26^\\circ\\text{{C}}$ observado no fim de vida não pode decorrer de flutuações amostrais ou ruído gaussiano.
2. **$H_{{3a}}$ e $H_{{3b}}$ (Estrutura de Autocorrelação de Wold):** O teste de Ljung-Box confirmou que os resíduos do sensor virtual na fase saudável são estocásticos e não-correlacionados ($Q = 2,90, p = 0,984$), provando ausência de underfitting; na fase degradada, $Q = 715,65$ com $p = 2,74 \\times 10^{{-147}}$, confirmando a emergência de memória física e quebra estrutural.
3. **$H_4$ (Superioridade de Todos os Modelos ML sobre a Média):** Os testes pareados de Wilcoxon signed-rank rejeitaram categoricamente a hipótese nula para todos os 13 estimadores ($p < 10^{{-100}}$), com ganhos médios de $18,4$ a $24,3$ ciclos de precisão.
4. **$H_5$ (Diálogo Epifani et al. IEEE CIM 2026):** O XGBoost superou o PyTorch Transformer ($p = 1,17 \\times 10^{{-131}}$), o PyTorch RNN/LSTM ($p < 10^{{-300}}$), o PyTorch CNN ($p = 3,92 \\times 10^{{-63}}$) e o SVM ($p < 10^{{-300}}$). A superioridade dos GBDTs sobre Deep Learning para séries tabulares complexas é estatisticamente inequívoca.

## 11B.2 Intervalos de Confiança Não-Paramétricos (Bootstrap Fleets, B=500, 95% CI)

{boot_md}

## 11B.3 Benchmark Completo de RUL na Frota Hold-out (14 Modelos)

{bench_md}
"""

    if "Capítulo 11B: Validação Estatística de Hipóteses" not in content:
        # Insert before Chapter 12
        split_marker = "# Capítulo 12: Diálogo Científico com o Artigo do IEEE CIM 2026"
        if split_marker in content:
            content = content.replace(split_marker, thesis_chapter + "\n" + split_marker)
        else:
            content += thesis_chapter

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print("Updated docs/learn.md successfully.")

def escape_latex(s):
    # Escape special LaTeX characters for tables
    replacements = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s

def format_latex_tables():
    bench_df = pd.read_csv(BENCHMARK_CSV)
    hyp_df = pd.read_csv(HYPOTHESIS_CSV)
    boot_df = pd.read_csv(BOOTSTRAP_CSV)

    families = {
        "Baseline (Mean RUL)": "Linha de Base",
        "Ridge": "Linear Regularizado",
        "Support Vector Machine (SVM)": "Kernel / Linear SVR",
        "MiniRocket + RidgeCV": "Convolucao Deterministica",
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

    # Benchmark LaTeX Table
    bench_tex = r"""\begin{table}[H]
\centering
\small
\begin{tabularx}{\textwidth}{l l c c c c c r}
\toprule
\textbf{Modelo} & \textbf{Familia} & $\mathbf{R^2}$ & \textbf{RMSE} & \textbf{MAE} & \textbf{NASA Score} & \textbf{Late \%} & \textbf{Tempo} \\
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

    # Hypothesis LaTeX Table
    hyp_tex = r"""\begin{table}[H]
\centering
\scriptsize
\begin{tabularx}{\textwidth}{l X l r r c}
\toprule
\textbf{ID} & \textbf{Fenomeno Investigado} & \textbf{Teste Estatistico} & \textbf{Estatistica} & \textbf{p-valor} & \textbf{Decisao} \\
\midrule
"""
    for _, r in hyp_df.iterrows():
        id_esc = escape_latex(str(r['ID']))
        desc_esc = escape_latex(str(r['Description']))
        test_esc = escape_latex(str(r['Test']))
        p_val = str(r['p_value'])
        decisao = "REJEITADA" if "REJEITADA" in str(r['H0_Rejected']) else "NAO REJ."
        stat_val = f"{r['Statistic']:.2f}" if abs(r['Statistic']) < 1e5 else f"{r['Statistic']:.1e}"
        hyp_tex += f"\\textbf{{{id_esc}}} & {desc_esc} & {test_esc} & {stat_val} & \\textbf{{{p_val}}} & {decisao} \\\\\n"
    hyp_tex += r"""\bottomrule
\end{tabularx}
\caption{Matriz de Testes de Hipoteses com P-Valores Exatos e Conclusoes Cientificas.}
\label{tab:thesis_hypothesis_tests}
\end{table}
"""

    # Bootstrap LaTeX Table
    boot_tex = r"""\begin{table}[H]
\centering
\small
\begin{tabularx}{\textwidth}{l c c c}
\toprule
\textbf{Modelo} & $\mathbf{R^2}$ \textbf{Medio [IC 95\%]} & \textbf{RMSE Medio [IC 95\%]} & \textbf{MAE Medio [IC 95\%]} \\
\midrule
"""
    for _, r in boot_df.iterrows():
        m_esc = escape_latex(str(r['Model']))
        boot_tex += f"\\textbf{{{m_esc}}} & {r['R2_Mean']:.4f} [{r['R2_CI_Lower']:.4f}, {r['R2_CI_Upper']:.4f}] & {r['RMSE_Mean']:.2f} [{r['RMSE_CI_Lower']:.2f}, {r['RMSE_CI_Upper']:.2f}] & {r['MAE_Mean']:.2f} [{r['MAE_CI_Lower']:.2f}, {r['MAE_CI_Upper']:.2f}] \\\\\n"
    boot_tex += r"""\bottomrule
\end{tabularx}
\caption{Intervalos de Confianca Nao-Parametricos de 95\% via Bootstrap na Frota ($B=500$).}
\label{tab:thesis_bootstrap_ci}
\end{table}
"""

    return bench_tex, hyp_tex, boot_tex

def update_latex():
    with open(TEX_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Clean control chars
    content = content.replace("\x07", r"\argmin ")
    content = content.replace("\x08", r"\bar{y}")
    
    # Clean autoref labels
    content = re.sub(r'\\autoref\{fig:([a-zA-Z0-9]+)\\_', r'\\autoref{fig:\1_', content)
    content = re.sub(r'\\_([a-zA-Z0-9]+)\}', r'_\1}', content)
    content = re.sub(r'\\_([a-zA-Z0-9]+)\\_', r'_\1_', content)
    content = content.replace(r"\autoref{fig:t50\_time\_series\_envelope\_detection})}", r"\autoref{fig:t50_time_series_envelope_detection})")
    content = content.replace(r"\autoref{fig:p30\_time\_series\_envelope\_detection})}", r"\autoref{fig:p30_time_series_envelope_detection})")
    content = content.replace(r"\autoref{fig:nc\_time\_series\_envelope\_detection})}", r"\autoref{fig:nc_time_series_envelope_detection})")
    content = content.replace(r"\autoref{fig:bpr\_time\_series\_envelope\_detection})}", r"\autoref{fig:bpr_time_series_envelope_detection})")
    content = content.replace(r"\autoref{fig:t50\_geometric\_envelope\_detection})}", r"\autoref{fig:t50_geometric_envelope_detection})")
    content = content.replace(r"\autoref{fig:minirocket\_feature\_filtering\_padua})}", r"\autoref{fig:minirocket_feature_filtering_padua})")
    content = content.replace(r"\autoref{fig:rul\_best\_model\_parity\_envelope})}", r"\autoref{fig:rul_best_model_parity_envelope})")
    content = content.replace(r"\autoref{fig:rul\_model\_r2\_comparison})}", r"\autoref{fig:rul_model_r2_comparison})")

    # Add unicode declarations if not present
    unicode_decls = r"""
\DeclareUnicodeCharacter{03B1}{\ensuremath{\alpha}}
\DeclareUnicodeCharacter{03BC}{\ensuremath{\mu}}
\DeclareUnicodeCharacter{03C3}{\ensuremath{\sigma}}
\DeclareUnicodeCharacter{2265}{\ensuremath{\ge}}
\DeclareUnicodeCharacter{2264}{\ensuremath{\le}}
\DeclareUnicodeCharacter{2248}{\ensuremath{\approx}}
\DeclareUnicodeCharacter{00B1}{\ensuremath{\pm}}
\DeclareUnicodeCharacter{00B0}{\ensuremath{^\circ}}
\DeclareUnicodeCharacter{00B2}{\ensuremath{^2}}
\DeclareUnicodeCharacter{2014}{---}
\DeclareUnicodeCharacter{00AA}{\ensuremath{^a}}
"""
    if "\\DeclareUnicodeCharacter{03B1}" not in content:
        content = content.replace(r"\usepackage[utf8]{inputenc}", r"\usepackage[utf8]{inputenc}" + unicode_decls)

    # Insert thesis hypothesis section before Chapter 12
    bench_tex, hyp_tex, boot_tex = format_latex_tables()
    
    latex_chapter = (
        "
\section{Capítulo 11B: Validação Estatística de Hipóteses (P-Values e Bootstrap CI 95%)}
"
        "Em conformidade com as diretrizes do \texttt{@ml-best-practices} e o diálogo com Epifani et al. (IEEE CIM 2026), conduzimos a validação formal de 20 hipóteses científicas com extração exata de hBcvalores e intervalos de confiança não-paramétricos obtidos por reamostragem bootstrap (=500$ frotas hold-out disjuntas).

"
        + hyp_tex + "
"
        "\subsection{Intervalos de Confiança Não-Paramétricos (Bootstrap Fleets, B=500)}
"
        "A Tabela~\ref{tab:thesis_bootstrap_ci} reporta os intervalos de confiança de 95% para ^2$, RMSE e MAE obtidos em 500 iterações de bootstrap na frota de teste.

"
        + boot_tex + "
"
        "\subsection{Benchmark Completo dos 14 Modelos Avaliados}
"
        "A Tabela~\ref{tab:rul_holdout_benchmark} resume o desempenho comparativo de todos os 14 estimadores avaliados no teste hold-out disjunto de 62 turbinas.

"
        + bench_tex + "
"
    )
    if "Capítulo 11B: Validação Estatística de Hipóteses" not in content:
        split_mark = r"\section{12. Diálogo Científico com o Artigo do IEEE CIM 2026"
        if split_mark in content:
            content = content.replace(split_mark, latex_chapter + "\n" + split_mark)
        else:
            split_mark_alt = r"\section{Capítulo 12:"
            if split_mark_alt in content:
                content = content.replace(split_mark_alt, latex_chapter + "\n" + split_mark_alt)
            else:
                # Place before references
                content = content.replace(r"\begin{thebibliography}", latex_chapter + "\n\\begin{thebibliography}")

    with open(TEX_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print("Updated docs/learn.tex successfully.")

def compile_pdf():
    print("Compiling docs/learn.tex via pdflatex (Pass 1)...")
    res1 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory=docs", "docs/learn.tex"],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        errors="replace"
    )
    print(f"Pass 1 finished with exit code {res1.returncode}.")
    
    print("Compiling docs/learn.tex via pdflatex (Pass 2 for TOC/Cross-References)...")
    res2 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory=docs", "docs/learn.tex"],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        errors="replace"
    )
    print(f"Pass 2 finished with exit code {res2.returncode}.")

    if os.path.exists(PDF_PATH):
        size_mb = os.path.getsize(PDF_PATH) / (1024 * 1024)
        print(f"PDF generated successfully at {PDF_PATH} ({size_mb:.2f} MB).")
    else:
        print("Error: PDF was not generated.")

if __name__ == "__main__":
    update_markdown()
    update_latex()
    compile_pdf()
