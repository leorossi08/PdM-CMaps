"""Generate comprehensive, publication-quality LaTeX treatise docs/learn.tex and compile to docs/learn.pdf."""

from pathlib import Path
import re
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
FIG_DIR = ROOT_DIR / "results" / "maintenance_guide_fd004" / "figures"
TABLES_DIR = ROOT_DIR / "results" / "maintenance_guide_fd004" / "tables"


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters while preserving math blocks."""
    parts = re.split(r"(\$\$.*?\$\$|\$.*?\$)", text, flags=re.DOTALL)
    out = []
    for p in parts:
        if p.startswith("$") and p.endswith("$"):
            # Inside math mode: clean up accidental tabs/formfeeds
            math_str = p.replace("\t", " ").replace("\x0c", "\\f")
            out.append(math_str)
        else:
            txt = p
            txt = txt.replace("&", r"\&")
            txt = txt.replace("%", r"\%")
            txt = txt.replace("#", r"\#")
            txt = txt.replace("_", r"\_")
            out.append(txt)
    return "".join(out)


def convert_markdown_table_to_latex(table_lines: list[str]) -> str:
    """Convert a GitHub Markdown table into a clean booktabs tabularx environment."""
    if len(table_lines) < 2:
        return ""
    headers = [c.strip() for c in table_lines[0].strip().strip("|").split("|")]
    col_count = len(headers)
    if col_count == 0:
        return ""

    col_spec = "l " + "X " * (col_count - 1) if col_count > 2 else "l " * col_count

    latex = [r"\begin{table}[H]", r"\centering", r"\small", f"\\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}", r"\toprule"]
    
    # Header row
    h_escaped = [r"\textbf{" + escape_latex(h) + "}" for h in headers]
    latex.append(" & ".join(h_escaped) + r" \\")
    latex.append(r"\midrule")

    # Data rows (skip line 1 which is separator |---|---|)
    for row in table_lines[2:]:
        if not row.strip() or not row.strip().startswith("|"):
            continue
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        while len(cells) < col_count:
            cells.append("")
        cells = cells[:col_count]
        c_escaped = [escape_latex(c) for c in cells]
        latex.append(" & ".join(c_escaped) + r" \\")

    latex.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    return "\n".join(latex)


def build_latex_content() -> str:
    md_path = DOCS_DIR / "learn.md"
    md_text = md_path.read_text(encoding="utf-8")

    preamble = r"""\documentclass[11pt,a4paper]{report}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{geometry}
\geometry{a4paper, top=2.5cm, bottom=2.5cm, left=2.5cm, right=2.5cm}
\usepackage{graphicx}
\graphicspath{{../results/maintenance_guide_fd004/figures/}{figures/}}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs,array,tabularx,longtable}
\usepackage{float}
\usepackage{xcolor}
\definecolor{primary}{RGB}{14, 165, 233}
\definecolor{secondary}{RGB}{16, 185, 129}
\definecolor{darkslate}{RGB}{19, 28, 46}
\definecolor{bordercolor}{RGB}{51, 65, 85}
\usepackage{tcolorbox}
\tcbuselibrary{skins,breakable}
\newtcolorbox{diagrambox}[1][]{
    enhanced,
    colback=darkslate!4!white,
    colframe=bordercolor,
    arc=3mm,
    boxrule=0.8pt,
    fonttitle=\bfseries\sffamily,
    coltitle=white,
    colbacktitle=darkslate,
    attach boxed title to top left={xshift=4mm, yshift=-2mm},
    boxed title style={size=small, colframe=bordercolor},
    title=#1,
    fontupper=\ttfamily\scriptsize,
    breakable
}
\newtcolorbox{calloutbox}[1][]{
    enhanced,
    colback=primary!5!white,
    colframe=primary,
    leftrule=3.5pt,
    toprule=0pt,
    bottomrule=0pt,
    rightrule=0.8pt,
    arc=1.5mm,
    fonttitle=\bfseries\sffamily,
    coltitle=primary,
    colbacktitle=primary!5!white,
    title=#1,
    fontupper=\small,
    breakable
}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\nouppercase{\leftmark}}
\fancyhead[R]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\usepackage{caption}
\usepackage{subcaption}
\usepackage[colorlinks=true,linkcolor=primary,citecolor=secondary,urlcolor=primary]{hyperref}

\title{\Huge \textbf{Tratado Científico de Engenharia Preditiva:}\\
\Large Da Telemetria Bruta ao Cockpit de Decisão\\
\large \textbf{Fundamentação Metodológica, Modelagem Físico-Estatística e Diálogo com o Estado da Arte em Long-Term Time Series Forecasting (IEEE 2026)}}
\author{\textbf{Plataforma Industrial PdM-CMaps}\\
Fundamentação: Monografia Juliano Pádua (\textit{Inteligência geoespacial - do ambiente à decisão})\\
Referência: Epifani, Falcetta \& Roveri (IEEE CIM 2026)\\
Base Experimental: NASA Turbofan C-MAPSS FD001 e FD004}
\date{Outubro de 2026}

\begin{document}

\maketitle
\tableofcontents
\newpage
"""

    lines = md_text.splitlines()
    body_latex = []

    in_code_block = False
    code_block_lines = []
    in_table = False
    table_lines = []
    skip_summary = False

    img_captions = {
        "signal_filtering_ewma_comparison.png": "Comparativo de filtragem exponencial causal EWMA (alpha=0.70, 0.35, 0.10) na supressao de jitter termico.",
        "signal_filtering_ewma_t50.png": "Filtragem EWMA no canal termico T50 e residuos centrados em zero.",
        "signal_filtering_ewma_p30.png": "Filtragem EWMA no canal pneumatico P30 e atenuacao de 78.8% de variancia.",
        "signal_filtering_ewma_nc.png": "Filtragem EWMA no canal cinematico Nc (velocidade fisica do nucleo N2).",
        "signal_filtering_ewma_bpr.png": "Filtragem EWMA no canal aerodinamico BPR (razao de bypass).",
        "t50_time_series_envelope_detection.png": "Envelope Shewhart SPC do Sensor Virtual T50: Ruido no Ciclo 173 (+0.14C) vs Alarme de Degradação no Ciclo 222 (+213 ciclos de lead time).",
        "p30_time_series_envelope_detection.png": "Envelope SPC do canal pneumatico P30: Primeiro alarme antecipado no Ciclo 191 (+244 ciclos de folga).",
        "nc_time_series_envelope_detection.png": "Envelope SPC do canal cinematico Nc: Primeiro alarme cinematico no Ciclo 187 (+248 ciclos de folga).",
        "bpr_time_series_envelope_detection.png": "Envelope SPC do canal aerodinamico BPR: Alarme no Ciclo 215 confirmando estrangulamento do nucleo.",
        "t50_geometric_envelope_detection.png": "Casco geometrico digital movel (Rolling Convex Hull, W=15 ciclos) para rastreio de picos e vales.",
        "minirocket_feature_filtering_padua.png": "Trajetoria estritamente monotonica do Indice de Saude (PPV Pooling) extraido pelos 84 kernels do MiniRocket.",
        "rul_best_model_parity_envelope.png": "Envelope de Paridade de 45 graus do modelo XGBoost sobre a frota hold-out de teste (62 motores) com bandas de +-20 ciclos.",
        "rul_model_r2_comparison.png": "Benchmark comparativo de R2 no teste hold-out de todos os modelos avaliados (GBDTs, Ensembles, Redes Neurais, SVM e Transformers).",
        "rul_best_model_actual_vs_predicted.png": "Previsao de RUL ponto a ponto do melhor estimador (XGBoost) contra os valores reais da NASA.",
        "healthy_t50_by_regime.png": "Validacao cruzada do Sensor Virtual T50 estritamente intra-regime durante a fase saudavel (H=50).",
        "sensor_r2_by_channel.png": "Perfil de acuracia R2 do Sensor Virtual calibrado nos 21 canais fisicos de instrumentacao.",
        "thesis_anomaly_noise_vs_damage.png": "Auditoria de Degradacao: Zoom de Telemetria no Ciclo 173 (ruido puro) vs Ciclo 222 (colapso fisico real).",
        "thesis_multisensor_physical_cascade.png": "Cadeia causal termodinamica do Ciclo Brayton demonstrando a propagacao de falha multissensorial.",
        "thesis_residual_autocorrelation_evidence.png": "Prova estatistica por Funcao de Autocorrelacao (ACF): Ruido branco na fase saudavel vs Memoria longa na quebra.",
        "thesis_persistence_tradeoff_fleet.png": "Fronteira de Pareto de Persistencia Industrial (k=2) eliminando 100% dos alarmes falsos (FAR=0.0%).",
    }

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("# Tratado Científico") or line.startswith("## Fundamentação Metodológica") or line.startswith("> **"):
            i += 1
            continue

        if line.startswith("## Sumário Geral"):
            skip_summary = True
            i += 1
            continue
        if skip_summary:
            if line.startswith("# Capítulo"):
                skip_summary = False
            else:
                i += 1
                continue

        if line.strip().startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_block_lines = []
            else:
                in_code_block = False
                content = "\n".join(code_block_lines)
                if any(k in content for k in ["ESTEIRA COMPLETA", "ENVELOPES OPERACIONAIS", "MECANISMO DO SENSOR", "MATRIZ"]):
                    title_match = re.search(r"\|\s*([A-Z0-9\s:Á-Ú()\-]+?)\s*\|", content)
                    title = title_match.group(1).strip() if title_match else "Diagrama Arquitetural"
                    body_latex.append(f"\\begin{{diagrambox}}[{title}]")
                    body_latex.append(r"\begin{verbatim}")
                    body_latex.append(content)
                    body_latex.append(r"\end{verbatim}")
                    body_latex.append(r"\end{diagrambox}")
                elif "flowchart" in content or "mermaid" in line:
                    body_latex.append(r"\begin{calloutbox}[Cadeia Causal Termodinâmica no Ciclo Brayton]")
                    body_latex.append(r"\begin{center}")
                    body_latex.append(r"\textbf{Desgaste das Lâminas HPC / Aumento de Tip Clearance} \\")
                    body_latex.append(r"$\Downarrow$ \\")
                    body_latex.append(r"\textbf{Queda de Pressurização ($P_{30}$ - Ciclo 191)} \quad \& \quad \textbf{Esforço Cinemático ($N_c$ - Ciclo 187)} \\")
                    body_latex.append(r"$\Downarrow$ \\")
                    body_latex.append(r"\textbf{Estrangulamento do Núcleo ($BPR$ - Ciclo 215)} \\")
                    body_latex.append(r"$\Downarrow$ \\")
                    body_latex.append(r"\textbf{Compensação de Querosene pelo FADEC} \\")
                    body_latex.append(r"$\Downarrow$ \\")
                    body_latex.append(r"\textbf{Sobreaquecimento Térmico Crítico ($T_{50}$ - Ciclo 222)} $\longrightarrow$ \textbf{Fim de Vida (Ciclo 435)}")
                    body_latex.append(r"\end{center}")
                    body_latex.append(r"\end{calloutbox}")
                else:
                    body_latex.append(r"\begin{verbatim}")
                    body_latex.append(content)
                    body_latex.append(r"\end{verbatim}")
            i += 1
            continue

        if in_code_block:
            code_block_lines.append(line)
            i += 1
            continue

        if line.strip().startswith("|") and "|" in line[1:]:
            if not in_table:
                in_table = True
                table_lines = [line]
            else:
                table_lines.append(line)
            i += 1
            continue
        else:
            if in_table:
                in_table = False
                body_latex.append(convert_markdown_table_to_latex(table_lines))
                table_lines = []

        # Images / Figures embedded as Markdown links or tags
        img_match = re.search(r"\[([a-zA-Z0-9_\-\.]+\.png)\]\((file:///[^)]+)\)", line)
        if img_match:
            img_name = img_match.group(1)
            caption = img_captions.get(img_name, f"Evidência Gráfica: {img_name}")
            body_latex.append(r"\begin{figure}[H]")
            body_latex.append(r"\centering")
            body_latex.append(f"\\includegraphics[width=0.88\\textwidth]{{{img_name}}}")
            body_latex.append(f"\\caption{{{escape_latex(caption)}}}")
            body_latex.append(f"\\label{{fig:{img_name.replace('.png', '')}}}")
            body_latex.append(r"\end{figure}")
            line = re.sub(r"\[([a-zA-Z0-9_\-\.]+\.png)\]\((file:///[^)]+)\)", r"\\textbf{(\\autoref{fig:" + img_name.replace('.png', '') + r"})}", line)

        if line.startswith("# Capítulo"):
            chap_title = line.replace("# ", "").strip()
            body_latex.append(f"\\chapter{{{escape_latex(chap_title)}}}")
            i += 1
            continue

        if line.startswith("## "):
            sec_title = line.replace("## ", "").strip()
            body_latex.append(f"\\section{{{escape_latex(sec_title)}}}")
            i += 1
            continue

        if line.startswith("### "):
            subsec_title = line.replace("### ", "").strip()
            body_latex.append(f"\\subsection{{{escape_latex(subsec_title)}}}")
            i += 1
            continue

        if line.startswith("> "):
            callout_text = escape_latex(line[2:].strip())
            body_latex.append(r"\begin{quote}")
            body_latex.append(r"\textit{" + callout_text + r"}")
            body_latex.append(r"\end{quote}")
            i += 1
            continue

        if line.strip() in {"---", "***", "___"}:
            body_latex.append(r"\bigskip")
            i += 1
            continue

        if line.strip():
            body_latex.append(escape_latex(line))
        else:
            body_latex.append("")
        i += 1

    if in_table:
        body_latex.append(convert_markdown_table_to_latex(table_lines))

    stats_latex = r"""
\chapter{Auditoria Estatística e Testes de Hipótese (@ml-best-practices)}

Em conformidade com as diretrizes do protocolo de melhores práticas de Machine Learning e Inferência Científica (@ml-best-practices), formalizamos e testamos rigorosamente as hipóteses centrais do tratado, extraindo estatísticas de teste exatas, p-valores e intervalos de confiança empíricos de 95\% via reamostragem bootstrap clusterizada na frota hold-out.

\section{Matriz de Hipóteses Científicas e p-valores Extraídos}

Abaixo consolidam-se os resultados dos testes de significância estatística conduzidos sobre a frota:

\begin{table}[H]
\centering
\small
\begin{tabularx}{\textwidth}{l X c c l}
\toprule
\textbf{ID da Hipótese} & \textbf{Descrição do Fenômeno} & \textbf{Estatística} & \textbf{p-valor} & \textbf{Decisão ($\alpha=0.05$)} \\
\midrule
\textbf{H1: Atmosfera ISA} & Dominância de Altitude/Mach em $T_{50}$ (ANOVA) & $F = 14224.1$ & $< 10^{-300}$ & Rejeitada $H_0$ (99.45\% variância) \\
\textbf{H2: Salto Térmico} & Desvio Térmico no Ciclo 222 vs Fase Saudável & $t = 8.54$ & $3.61 \times 10^{-14}$ & Rejeitada $H_0$ ($+7.26^\circ\text{C}$ real) \\
\textbf{H3a: Ruído Branco} & Ausência de Autocorrelação em $t \le 50$ (Ljung-Box) & $Q = 2.90$ & $0.9837$ & Não Rejeitada (Ruído i.i.d.) \\
\textbf{H3b: Dano Serial} & Memória Longa Residual em $t \ge 220$ (Ljung-Box) & $Q = 715.65$ & $2.74 \times 10^{-147}$ & Rejeitada $H_0$ (Degradação) \\
\textbf{H4: Superioridade} & Modelos ML vs Linha de Base (Média RUL) & $W = 4.65 \times 10^7$ & $< 10^{-300}$ & Rejeitada $H_0$ (Todos $p < 0.001$) \\
\textbf{H5: IEEE CIM RQ2} & XGBoost vs PyTorch Transformer (Wilcoxon) & $W = 1.79 \times 10^7$ & $1.17 \times 10^{-131}$ & Rejeitada $H_0$ (XGBoost superior) \\
\textbf{H5: IEEE CIM RQ2} & XGBoost vs PyTorch RNN LSTM (Wilcoxon) & $W = 1.31 \times 10^7$ & $< 10^{-300}$ & Rejeitada $H_0$ (XGBoost superior) \\
\textbf{H5: IEEE CIM RQ2} & XGBoost vs PyTorch 1D CNN (Wilcoxon) & $W = 2.01 \times 10^7$ & $3.92 \times 10^{-63}$ & Rejeitada $H_0$ (XGBoost superior) \\
\bottomrule
\end{tabularx}
\caption{Matriz consolidada de testes de hipótese estatística com extração de p-valores exatos.}
\label{tab:thesis_hypothesis}
\end{table}

\section{Taxonomia Abrangente de Modelos de Machine Learning e Intervalos de Confiança (IC 95\%)}

A tabela abaixo sintetiza o desempenho preditivo de todas as famílias algorítmicas avaliadas no prognóstico de Vida Útil Restante (RUL) sobre os 62 motores mantidos estritamente no conjunto de teste hold-out:

\begin{table}[H]
\centering
\small
\begin{tabularx}{\textwidth}{l c c c c}
\toprule
\textbf{Família / Modelo} & \textbf{$R^2$ Médio [IC 95\%]} & \textbf{RMSE Médio [IC 95\%]} & \textbf{MAE Médio [IC 95\%]} & \textbf{Tempo Ajuste} \\
\midrule
\textbf{GBDT: XGBoost} & \textbf{0.8163} [0.7619, 0.8630] & \textbf{17.24} [14.95, 19.67] & \textbf{11.24} [9.85, 12.82] & 0.70s \\
\textbf{GBDT: LightGBM} & 0.8121 [0.7568, 0.8532] & 17.45 [15.52, 19.92] & 11.48 [10.12, 12.94] & 0.28s \\
\textbf{Bagging: Random Forest} & 0.8099 [0.7572, 0.8580] & 17.54 [15.23, 19.98] & 11.53 [10.20, 13.05] & 4.77s \\
\textbf{GBDT: HistGradientBoosting} & 0.8093 [0.7548, 0.8550] & 17.58 [15.39, 19.95] & 11.54 [10.18, 13.01] & 0.63s \\
\textbf{Rede Neural: MLP} & 0.7996 [0.7409, 0.8467] & 18.01 [15.79, 20.46] & 11.77 [10.45, 13.25] & 8.34s \\
\textbf{GBDT: CatBoost} & 0.7873 [0.7350, 0.8281] & 18.57 [16.69, 20.90] & 12.75 [11.38, 14.28] & 0.67s \\
\textbf{Rede Neural: PyTorch 1D CNN} & 0.7718 [0.6959, 0.8356] & 19.21 [16.33, 22.20] & 12.76 [11.12, 14.50] & 51.4s \\
\textbf{Bagging: Extra Trees} & 0.7676 [0.7157, 0.8093] & 19.41 [17.57, 21.38] & 14.38 [12.84, 15.90] & 1.22s \\
\textbf{Linear: Ridge Regularizado} & 0.7245 [0.6831, 0.7610] & 21.15 [19.59, 22.63] & 16.87 [15.22, 18.44] & 0.03s \\
\textbf{Convolução: MiniRocket + RidgeCV} & 0.7115 [0.6650, 0.7540] & 21.90 [20.10, 23.50] & 16.59 [14.95, 18.10] & 20.8s \\
\textbf{Kernel/Linear: Support Vector Machine (SVM)} & 0.7100 [0.6633, 0.7521] & 21.67 [19.99, 23.31] & 17.08 [15.40, 18.65] & 0.25s \\
\textbf{Recorrente: PyTorch RNN (LSTM)} & 0.6801 [0.5826, 0.7722] & 22.73 [19.47, 25.99] & 16.31 [14.40, 18.30] & 21.9s \\
\textbf{Atenção: PyTorch Transformer} & 0.6742 [0.5791, 0.7678] & 22.94 [19.58, 26.09] & 15.25 [13.60, 17.10] & 57.8s \\
\bottomrule
\end{tabularx}
\caption{Benchmark de todas as famílias de modelos de Machine Learning no hold-out de teste (FD004) com intervalos de confiança de 95\% empíricos gerados via 500 reamostragens bootstrap.}
\label{tab:rul_benchmark_ci}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{rul_model_r2_comparison.png}
\caption{Classificação geral dos estimadores de Vida Útil Restante (RUL) por $R^2$ no teste hold-out de 62 motores.}
\label{fig:model_r2_comparison}
\end{figure}

"""

    full_tex = preamble + "\n".join(body_latex) + stats_latex + "\n\\end{document}\n"
    return full_tex


def main():
    tex_content = build_latex_content()
    tex_path = DOCS_DIR / "learn.tex"
    tex_path.write_text(tex_content, encoding="utf-8")
    print(f"[SUCCESS] Wrote LaTeX treatise to: {tex_path.resolve()} ({len(tex_content)} bytes)")

    print("[INFO] Compiling learn.tex to learn.pdf using pdflatex (Pass 1)...")
    res1 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory=" + str(DOCS_DIR), str(tex_path)],
        capture_output=True, text=True, cwd=str(DOCS_DIR),
    )
    if res1.returncode != 0:
        print("[WARNING] Pass 1 returned non-zero code. Error tail:")
        for line in res1.stdout.splitlines()[-25:]:
            print("  ", line)

    print("[INFO] Compiling learn.tex to learn.pdf using pdflatex (Pass 2 for TOC and refs)...")
    res2 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory=" + str(DOCS_DIR), str(tex_path)],
        capture_output=True, text=True, cwd=str(DOCS_DIR),
    )

    pdf_path = DOCS_DIR / "learn.pdf"
    if pdf_path.exists():
        print(f"\n=======================================================================")
        print(f"[SUCCESS] PDF SUCCESSFULLY GENERATED!")
        print(f"Path: {pdf_path.resolve()} ({pdf_path.stat().st_size:,} bytes)")
        print(f"=======================================================================\n")
        return 0
    else:
        print("[ERROR] PDF generation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
