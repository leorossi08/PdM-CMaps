#!/usr/bin/env python3
"""Converts docs/learn.md into docs/learn.tex with publication-grade formatting and compiles docs/learn.pdf."""

from pathlib import Path
import re
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
MD_PATH = DOCS_DIR / "learn.md"
TEX_PATH = DOCS_DIR / "learn.tex"
PDF_PATH = DOCS_DIR / "learn.pdf"

FIGURE_METADATA = {
    "thesis_multisensor_physical_cascade.png": {
        "caption": "Cadeia causal termodinâmica do Ciclo Brayton demonstrando a propagação de falha multissensorial.",
        "label": "fig:multisensor_cascade"
    },
    "healthy_t50_by_regime.png": {
        "caption": "Validação cruzada do Sensor Virtual T50 estritamente intra-regime durante a fase saudável (H=50 ciclos).",
        "label": "fig:healthy_t50_regime"
    },
    "healthy_p30_by_regime.png": {
        "caption": "Validação cruzada do canal de pressurização estática P30 intra-regime durante a fase saudável.",
        "label": "fig:healthy_p30_regime"
    },
    "sensor_r2_by_channel.png": {
        "caption": "Perfil de acurácia R2 do Sensor Virtual calibrado nos 21 canais físicos de instrumentação.",
        "label": "fig:sensor_r2_channels"
    },
    "t50_parity_envelope.png": {
        "caption": "Envelope de Paridade de 45 graus para o Sensor Virtual T50 na fase de calibração saudável.",
        "label": "fig:t50_parity_env"
    },
    "p30_parity_envelope.png": {
        "caption": "Envelope de Paridade de 45 graus para o Sensor Virtual P30 demonstrando dispersão centrada.",
        "label": "fig:p30_parity_env"
    },
    "signal_filtering_ewma_comparison.png": {
        "caption": "Comparativo de filtragem exponencial causal EWMA (alpha=0.70, 0.35, 0.10) na supressão de jitter térmico.",
        "label": "fig:ewma_comparison"
    },
    "signal_filtering_ewma_t50.png": {
        "caption": "Filtragem EWMA no canal térmico T50 e resíduos centrados em zero.",
        "label": "fig:ewma_t50"
    },
    "signal_filtering_ewma_p30.png": {
        "caption": "Filtragem EWMA no canal pneumático P30 e atenuação de 78.8% de variância.",
        "label": "fig:ewma_p30"
    },
    "signal_filtering_ewma_nc.png": {
        "caption": "Filtragem EWMA no canal cinemático Nc (velocidade física do núcleo N2).",
        "label": "fig:ewma_nc"
    },
    "signal_filtering_ewma_bpr.png": {
        "caption": "Filtragem EWMA no canal aerodinâmico BPR (razão de bypass).",
        "label": "fig:ewma_bpr"
    },
    "t50_time_series_envelope_detection.png": {
        "caption": "Envelope Shewhart SPC do Sensor Virtual T50: Ruído no Ciclo 173 (+0.14C) vs Alarme de Degradação no Ciclo 222 (+213 ciclos de lead time).",
        "label": "fig:t50_shewhart_envelope"
    },
    "p30_time_series_envelope_detection.png": {
        "caption": "Envelope SPC do canal pneumático P30: Primeiro alarme antecipado no Ciclo 191 (+244 ciclos de folga).",
        "label": "fig:p30_shewhart_envelope"
    },
    "nc_time_series_envelope_detection.png": {
        "caption": "Envelope SPC do canal cinemático Nc: Primeiro alarme mecânico no Ciclo 187 (+248 ciclos de folga).",
        "label": "fig:nc_shewhart_envelope"
    },
    "bpr_time_series_envelope_detection.png": {
        "caption": "Envelope SPC do canal aerodinâmico BPR: Alarme no Ciclo 215 confirmando estrangulamento do núcleo.",
        "label": "fig:bpr_shewhart_envelope"
    },
    "t50_geometric_envelope_detection.png": {
        "caption": "Casco geométrico digital móvel (Rolling Convex Hull, W=15 ciclos) para rastreio de picos e vales.",
        "label": "fig:t50_geometric_envelope"
    },
    "thesis_anomaly_noise_vs_damage.png": {
        "caption": "Auditoria de Degradação: Zoom de Telemetria no Ciclo 173 (ruído puro) vs Ciclo 222 (colapso físico real).",
        "label": "fig:anomaly_noise_vs_damage"
    },
    "thesis_residual_autocorrelation_evidence.png": {
        "caption": "Prova estatística por Função de Autocorrelação (ACF): Ruído branco na fase saudável vs Memória longa na quebra.",
        "label": "fig:residual_acf_evidence"
    },
    "thesis_persistence_tradeoff_fleet.png": {
        "caption": "Fronteira de Pareto de Persistência Industrial (k=2) eliminando 100% dos alarmes falsos (FAR=0.0%).",
        "label": "fig:persistence_pareto"
    },
    "minirocket_feature_filtering_padua.png": {
        "caption": "Trajetória estritamente monotônica do Índice de Saúde (PPV Pooling) extraído pelos 84 kernels do MiniRocket.",
        "label": "fig:minirocket_health_index"
    },
    "minirocket_feature_filtering_t50.png": {
        "caption": "Extração convolucional causal MiniRocket aplicada sobre a série térmica T50.",
        "label": "fig:minirocket_t50"
    },
    "rul_model_r2_comparison.png": {
        "caption": "Benchmark comparativo de R2 no teste hold-out de todos os modelos avaliados (GBDTs, Ensembles, Redes Neurais, SVM e Transformers).",
        "label": "fig:rul_r2_comparison"
    },
    "rul_best_model_parity_envelope.png": {
        "caption": "Envelope de Paridade de 45 graus do modelo XGBoost sobre a frota hold-out de teste (62 motores) com bandas de +-20 ciclos.",
        "label": "fig:rul_parity_envelope"
    },
    "rul_best_model_actual_vs_predicted.png": {
        "caption": "Previsão de RUL ponto a ponto do melhor estimador (XGBoost) contra os valores reais da NASA.",
        "label": "fig:rul_actual_vs_pred"
    }
}


def escape_latex(text: str) -> str:
    """Escapes special LaTeX characters outside math blocks."""
    # Convert citation markers [1] -> \cite{padua2023}, [2] -> \cite{epifani2026}, etc.
    cite_map = {
        "[1]": r"\cite{padua2023}",
        "[2]": r"\cite{epifani2026}",
        "[3]": r"\cite{saxena2008}",
        "[4]": r"\cite{montgomery2009}",
        "[5]": r"\cite{dempster2020}",
        "[6]": r"\cite{chen2016}",
        "[7]": r"\cite{ke2017}",
        "[8]": r"\cite{prokhorenkova2018}",
        "[9]": r"\cite{vaswani2017}",
        "[10]": r"\cite{box2015}",
        "[11]": r"\cite{efron1994}",
        "[12]": r"\cite{iso13374}"
    }

    parts = re.split(r"(\$\$.*?\$\$|\$.*?\$)", text, flags=re.DOTALL)
    out = []
    for p in parts:
        if p.startswith("$") and p.endswith("$"):
            # Inside math mode
            math_str = p.replace("\t", " ").replace("\x0c", "\\frac").replace("\x07", "\\argmin ").replace("\x08", "\\bar{y}")
            out.append(math_str)
        else:
            txt = p
            for k, v in cite_map.items():
                txt = txt.replace(k, v)
            txt = txt.replace("&", r"\&")
            txt = txt.replace("%", r"\%")
            txt = txt.replace("#", r"\#")
            txt = txt.replace("_", r"\_")
            out.append(txt)
    return "".join(out)


def convert_table(lines: list[str]) -> str:
    """Converts a Markdown table to a tabularx environment with booktabs."""
    if len(lines) < 2:
        return ""
    headers = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    ncols = len(headers)
    if ncols == 0:
        return ""

    col_spec = "l " + "X " * (ncols - 1) if ncols > 2 else "l " * ncols
    tex = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize" if ncols > 6 else r"\small",
        f"\\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}",
        r"\toprule"
    ]
    h_esc = [r"\textbf{" + escape_latex(h) + "}" for h in headers]
    tex.append(" & ".join(h_esc) + r" \\")
    tex.append(r"\midrule")

    for row in lines[2:]:
        if not row.strip() or not row.strip().startswith("|"):
            continue
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        while len(cells) < ncols:
            cells.append("")
        cells = cells[:ncols]
        c_esc = [escape_latex(c) for c in cells]
        tex.append(" & ".join(c_esc) + r" \\")

    tex.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    return "\n".join(tex)


def build_tex():
    md_text = MD_PATH.read_text(encoding="utf-8")
    lines = md_text.splitlines()

    preamble = r"""\documentclass[11pt,a4paper]{report}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{geometry}
\geometry{a4paper, top=2.5cm, bottom=2.5cm, left=2.5cm, right=2.5cm, headheight=28pt}
\usepackage{graphicx}
\graphicspath{{../results/maintenance_guide_fd004/figures/}{results/maintenance_guide_fd004/figures/}{figures/}}
\usepackage{amsmath,amssymb,amsfonts}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\argmax}{arg\,max}
\usepackage{booktabs,array,tabularx,longtable}
\usepackage{float}
\usepackage{xcolor}
\definecolor{primary}{RGB}{14, 165, 233}
\definecolor{secondary}{RGB}{16, 185, 129}
\definecolor{darkslate}{RGB}{19, 28, 46}
\definecolor{bordercolor}{RGB}{51, 65, 85}

\usepackage{listings}
\lstset{
    inputencoding=utf8,
    extendedchars=true,
    literate=
        {á}{{\'a}}1 {à}{{\`a}}1 {ã}{{\~a}}1 {â}{{\^a}}1
        {é}{{\'e}}1 {ê}{{\^e}}1 {í}{{\'i}}1
        {ó}{{\'o}}1 {ô}{{\^o}}1 {õ}{{\~o}}1 {ú}{{\'u}}1 {ü}{{\"u}}1
        {ç}{{\c{c}}}1
        {Á}{{\'A}}1 {À}{{\`A}}1 {Ã}{{\~A}}1 {Â}{{\^A}}1
        {É}{{\'E}}1 {Ê}{{\^E}}1 {Í}{{\'I}}1
        {Ó}{{\'O}}1 {Ô}{{\^O}}1 {Õ}{{\~O}}1 {Ú}{{\'U}}1
        {Ç}{{\c{C}}}1
        {°}{{$^\circ$}}1 {±}{{$\pm$}}1 {²}{{$^2$}}1 {³}{{$^3$}}1 {µ}{{$\mu$}}1 {λ}{{$\lambda$}}1 {α}{{$\alpha$}}1 {σ}{{$\sigma$}}1
}
\usepackage{tcolorbox}
\tcbuselibrary{skins,breakable,listings}

\newtcolorbox{diagrambox}[1][]{
    enhanced,
    colback=darkslate!4!white,
    colframe=bordercolor,
    arc=2mm,
    boxrule=0.8pt,
    fonttitle=\bfseries\sffamily,
    coltitle=white,
    colbacktitle=darkslate,
    attach boxed title to top left={xshift=4mm, yshift=-2.5mm},
    boxed title style={size=small, colframe=bordercolor},
    title=#1,
    fontupper=\ttfamily\scriptsize,
    breakable
}

\newtcblisting{codebox}[1][]{
    enhanced,
    arc=2mm,
    boxrule=0.8pt,
    colback=gray!5!white,
    colframe=darkslate!80!black,
    fonttitle=\bfseries\sffamily\small,
    coltitle=white,
    colbacktitle=darkslate!80!black,
    attach boxed title to top left={xshift=4mm, yshift=-2.5mm},
    boxed title style={size=small, colframe=darkslate!80!black},
    title=#1,
    listing only,
    listing options={
        language=Python,
        basicstyle=\ttfamily\scriptsize,
        keywordstyle=\color{primary}\bfseries,
        stringstyle=\color{purple},
        commentstyle=\color{gray}\itshape,
        numbers=left,
        numberstyle=\tiny\color{gray},
        stepnumber=1,
        numbersep=5pt,
        breaklines=true,
        showstringspaces=false,
        tabsize=4
    },
    breakable
}

\newtcolorbox{examplebox}[1][]{
    enhanced,
    colback=secondary!4!white,
    colframe=secondary!80!black,
    leftrule=3.5pt,
    toprule=0.8pt,
    bottomrule=0.8pt,
    rightrule=0.8pt,
    arc=1.5mm,
    fonttitle=\bfseries\sffamily,
    coltitle=secondary!80!black,
    colbacktitle=secondary!10!white,
    title=#1,
    fontupper=\small,
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

\title{\Huge \textbf{Tratado Científico de Engenharia Preditiva:}\\
\Large Da Telemetria Bruta ao Cockpit de Decisão\\
\large \textbf{Fundamentação Metodológica, Modelagem Físico-Estatística e Diálogo com o Estado da Arte em Long-Term Time Series Forecasting (IEEE 2026)}}
\author{\textbf{Plataforma Industrial PdM-CMaps}\\
Fundamentação: Monografia Juliano Pádua (\textit{Inteligência geoespacial - do ambiente à decisão})\\
Referência do Estado da Arte: Epifani, Falcetta \& Roveri (IEEE CIM 2026)\\
Base Experimental: NASA Turbofan C-MAPSS FD001 e FD004}
\date{Outubro de 2026}

\begin{document}

\maketitle
\tableofcontents
\newpage
"""

    bib_tex = r"""\begin{thebibliography}{99}

\bibitem{padua2023}
J.~Pádua, \emph{Inteligência geoespacial - do ambiente à decisão}, Monografia de Especialização e Tratado Conceitual de Engenharia de Sistemas Preditivos, Instituto Tecnológico de Aeronáutica / Plataforma de Pós-Graduação, São José dos Campos, SP, 2023.

\bibitem{epifani2026}
A.~Epifani, A.~Falcetta, e M.~Roveri, ``Long-Term Time Series Forecasting: The Good, the Bad, and the Ugly,'' \emph{IEEE Computational Intelligence Magazine}, vol.~21, no.~1, pp.~24--38, Fev. 2026.

\bibitem{saxena2008}
A.~Saxena, K.~Goebel, D.~Simon, e N.~Eklund, ``Damage propagation modeling for aircraft engine run-to-failure simulation,'' em \emph{2008 International Conference on Prognostics and Health Management (PHM)}, Denver, CO, USA, 2008, pp.~1--9.

\bibitem{montgomery2009}
D.~C. Montgomery, \emph{Introduction to Statistical Quality Control}, 6ª~ed., John Wiley \& Sons, New York, NY, 2009.

\bibitem{dempster2020}
A.~Dempster, F.~Petitjean, e G.~I. Webb, ``MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification,'' \emph{Data Mining and Knowledge Discovery}, vol.~35, no.~5, pp.~1762--1781, 2020.

\bibitem{chen2016}
T.~Chen e C.~Guestrin, ``XGBoost: A Scalable Tree Boosting System,'' em \emph{Proc. 22nd ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining}, San Francisco, CA, USA, 2016, pp.~785--794.

\bibitem{ke2017}
G.~Ke, Q.~Meng, T.~Finley, T.~Wang, W.~Chen, W.~Ma, Q.~Ye, e T.-Y. Liu, ``LightGBM: A Highly Efficient Gradient Boosting Decision Tree,'' em \emph{Advances in Neural Information Processing Systems (NeurIPS)}, Long Beach, CA, USA, 2017, vol.~30, pp.~3146--3154.

\bibitem{prokhorenkova2018}
L.~Prokhorenkova, G.~Gusev, A.~Vorobev, A.~V. Dorogush, e A.~Gulin, ``CatBoost: unbiased boosting with categorical features,'' em \emph{Advances in Neural Information Processing Systems (NeurIPS)}, Montréal, Canadá, 2018, vol.~31, pp.~6638--6648.

\bibitem{vaswani2017}
A.~Vaswani, N.~Shazeer, N.~Parmar, J.~Uszkoreit, L.~Jones, A.~N. Gomez, Ł.~Kaiser, e I.~Polosukhin, ``Attention Is All You Need,'' em \emph{Advances in Neural Information Processing Systems (NeurIPS)}, Long Beach, CA, USA, 2017, vol.~30, pp.~5998--6008.

\bibitem{box2015}
G.~E. Box, G.~M. Jenkins, G.~C. Reinsel, e G.~M. Ljung, \emph{Time Series Analysis: Forecasting and Control}, 5ª~ed., John Wiley \& Sons, Hoboken, NJ, 2015.

\bibitem{efron1994}
B.~Efron e R.~J. Tibshirani, \emph{An Introduction to the Bootstrap}, CRC Press, Chapman \& Hall, Boca Raton, FL, 1994.

\bibitem{iso13374}
\emph{ISO 13374-1: Condition monitoring and diagnostics of machines --- Data processing, communication and presentation --- Part 1: General guidelines}, International Organization for Standardization, Genebra, Suíça, 2003.

\end{thebibliography}
"""

    body_tex = []

    in_code = False
    code_lang = "python"
    code_lines = []

    in_table = False
    table_lines = []

    in_example = False
    example_lines = []
    example_title = ""

    skip_summary = False

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

        if line.startswith("# Referências Bibliográficas"):
            break

        # Check for Example header
        if line.startswith("### Exemplo Teórico"):
            example_title = line.replace("### ", "").strip()
            in_example = True
            example_lines = []
            i += 1
            continue

        if in_example:
            if line.startswith("### ") or line.startswith("## ") or line.startswith("# ") or line.strip().startswith("```"):
                in_example = False
                body_tex.append(f"\\begin{{examplebox}}[{escape_latex(example_title)}]")
                body_tex.append(escape_latex("\n".join(example_lines)))
                body_tex.append(r"\end{examplebox}")
                example_lines = []
                # continue processing current line outside example
            else:
                example_lines.append(line)
                i += 1
                continue

        # Code block handler
        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
                code_lang = line.strip().replace("```", "").strip() or "python"
                code_lines = []
            else:
                in_code = False
                code_content = "\n".join(code_lines)
                if code_lang == "python":
                    body_tex.append(r"\begin{codebox}[Listagem de Implementação Python]")
                    body_tex.append(code_content)
                    body_tex.append(r"\end{codebox}")
                elif code_lang == "mermaid":
                    body_tex.append(r"\begin{calloutbox}[Cadeia Causal Termodinâmica no Ciclo Brayton]")
                    body_tex.append(r"\begin{center}")
                    body_tex.append(r"\textbf{Desgaste das Lâminas HPC / Aumento de Tip Clearance} \\")
                    body_tex.append(r"$\Downarrow$ \\")
                    body_tex.append(r"\textbf{Queda de Pressurização ($P_{30}$ - Ciclo 191)} \quad \& \quad \textbf{Esforço Cinemático ($N_c$ - Ciclo 187)} \\")
                    body_tex.append(r"$\Downarrow$ \\")
                    body_tex.append(r"\textbf{Estrangulamento do Núcleo ($BPR$ - Ciclo 215)} \\")
                    body_tex.append(r"$\Downarrow$ \\")
                    body_tex.append(r"\textbf{Compensação de Querosene pelo FADEC} \\")
                    body_tex.append(r"$\Downarrow$ \\")
                    body_tex.append(r"\textbf{Sobreaquecimento Térmico Crítico ($T_{50}$ - Ciclo 222)} $\longrightarrow$ \textbf{Fim de Vida (Ciclo 435)}")
                    body_tex.append(r"\end{center}")
                    body_tex.append(r"\end{calloutbox}")
                else:
                    title_match = re.search(r"\|\s*([A-Z0-9\s:Á-Ú()\-]+?)\s*\|", code_content)
                    title = title_match.group(1).strip() if title_match else "Diagrama Arquitetural"
                    body_tex.append(f"\\begin{{diagrambox}}[{title}]")
                    body_tex.append(r"\begin{verbatim}")
                    body_tex.append(code_content)
                    body_tex.append(r"\end{verbatim}")
                    body_tex.append(r"\end{diagrambox}")
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # Table handler
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
                body_tex.append(convert_table(table_lines))
                table_lines = []

        # Figures embedded as ![caption](file:///...)
        fig_match = re.search(r"!\[(.*?)\]\(file:///[^)]*?/([a-zA-Z0-9_\-]+\.png)\)", line)
        if fig_match:
            caption_text = fig_match.group(1)
            img_file = fig_match.group(2)
            meta = FIGURE_METADATA.get(img_file, {"caption": caption_text, "label": "fig:" + img_file.replace(".png", "")})
            caption = meta["caption"]
            label = meta["label"]
            body_tex.append(r"\begin{figure}[H]")
            body_tex.append(r"\centering")
            body_tex.append(f"\\includegraphics[width=0.88\\textwidth]{{{img_file}}}")
            body_tex.append(f"\\caption{{{escape_latex(caption)}}}")
            body_tex.append(f"\\label{{{label}}}")
            body_tex.append(r"\end{figure}")
            i += 1
            continue

        # Headings
        if line.startswith("# Capítulo"):
            chap_title = line.replace("# ", "").strip()
            body_tex.append(f"\\chapter{{{escape_latex(chap_title)}}}")
            i += 1
            continue

        if line.startswith("## "):
            sec_title = line.replace("## ", "").strip()
            body_tex.append(f"\\section{{{escape_latex(sec_title)}}}")
            i += 1
            continue

        if line.startswith("### "):
            subsec_title = line.replace("### ", "").strip()
            body_tex.append(f"\\subsection{{{escape_latex(subsec_title)}}}")
            i += 1
            continue

        if line.startswith("> "):
            quote_text = escape_latex(line[2:].strip())
            body_tex.append(r"\begin{quote}")
            body_tex.append(r"\textit{" + quote_text + r"}")
            body_tex.append(r"\end{quote}")
            i += 1
            continue

        if line.strip() in {"---", "***", "___"}:
            body_tex.append(r"\bigskip")
            i += 1
            continue

        # Regular text
        if line.strip():
            body_tex.append(escape_latex(line))
        else:
            body_tex.append("")
        i += 1

    if in_example:
        body_tex.append(f"\\begin{{examplebox}}[{escape_latex(example_title)}]")
        body_tex.append(escape_latex("\n".join(example_lines)))
        body_tex.append(r"\end{examplebox}")

    if in_table:
        body_tex.append(convert_table(table_lines))

    full_tex = preamble + "\n".join(body_tex) + "\n" + bib_tex + "\n\\end{document}\n"
    return full_tex


def main():
    print("[INFO] Building docs/learn.tex from docs/learn.md...")
    tex_content = build_tex()
    TEX_PATH.write_text(tex_content, encoding="utf-8")
    print(f"[SUCCESS] Wrote {TEX_PATH} ({len(tex_content):,} chars)")

    print("[INFO] Compiling learn.tex to learn.pdf using pdflatex (Pass 1)...")
    res1 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory=" + str(DOCS_DIR), str(TEX_PATH)],
        capture_output=True, text=True, errors="replace", cwd=str(DOCS_DIR)
    )
    if res1.returncode != 0:
        print("[WARNING] Pass 1 finished with non-zero code. Error tail:")
        for line in res1.stdout.splitlines()[-30:]:
            print("  ", line)

    print("[INFO] Compiling learn.tex to learn.pdf using pdflatex (Pass 2 for TOC, refs and citations)...")
    res2 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory=" + str(DOCS_DIR), str(TEX_PATH)],
        capture_output=True, text=True, errors="replace", cwd=str(DOCS_DIR)
    )
    if res2.returncode != 0:
        print("[WARNING] Pass 2 finished with non-zero code. Error tail:")
        for line in res2.stdout.splitlines()[-30:]:
            print("  ", line)

    if PDF_PATH.exists():
        size_mb = PDF_PATH.stat().st_size / (1024 * 1024)
        print(f"\n=======================================================================")
        print(f"[SUCCESS] TREATISE BOOK COMPILED SUCCESSFULLY!")
        print(f"Path: {PDF_PATH.resolve()} ({size_mb:.2f} MB)")
        print(f"=======================================================================\n")
        return 0
    else:
        print("[ERROR] PDF generation failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
