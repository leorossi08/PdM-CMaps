#!/usr/bin/env python3
"""Build publication-quality scientific treatise in Markdown and LaTeX, and compile to PDF.
All 15+ chapters, code listings, worked examples, third-person perspective, diagrams,
inline figures with analytical captions, and academic bibliography.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT_DIR / "results" / "maintenance_guide_fd004" / "figures"
TABLES_DIR = ROOT_DIR / "results" / "maintenance_guide_fd004" / "tables"

sys.path.insert(0, str(ROOT_DIR / "scripts"))
sys.path.insert(0, str(ROOT_DIR / "scripts" / "chapters"))

from treatise_front_matter import get_front_matter
import ch01_intro
import ch02_raw_data
import ch03_engineering
import ch04_regimes
import ch05_virtual_sensor
import ch06_filtering
import ch07_envelope
import ch08_degradation_proof
import ch09_minirocket
import ch10_benchmarking
import ch11_rul
import ch11b_statistics
import ch12_ieee_dialogue
import ch13_sop_cockpit
import ch14_catalog
import ch15_conclusions
import bibliography
from book_tables import load_tables
from book_config import FIGURE_METADATA

def assemble_markdown():
    chapters = [
        get_front_matter(),
        ch01_intro.get_chapter(),
        ch02_raw_data.get_chapter(),
        ch03_engineering.get_chapter(),
        ch04_regimes.get_chapter(),
        ch05_virtual_sensor.get_chapter(),
        ch06_filtering.get_chapter(),
        ch07_envelope.get_chapter(),
        ch08_degradation_proof.get_chapter(),
        ch09_minirocket.get_chapter(),
        ch10_benchmarking.get_chapter(),
        ch11_rul.get_chapter(),
        ch11b_statistics.get_chapter(),
        ch12_ieee_dialogue.get_chapter(),
        ch13_sop_cockpit.get_chapter(),
        ch14_catalog.get_chapter(),
        ch15_conclusions.get_chapter(),
        bibliography.get_markdown_bibliography(),
    ]
    return "\n\n---\n\n".join(chapters)


def escape_latex(text: str) -> str:
    """Escapes LaTeX special characters outside math mode."""
    parts = re.split(r"(\$\$.*?\$\$|\$.*?\$)", text, flags=re.DOTALL)
    out = []
    for p in parts:
        if p.startswith("$") and p.endswith("$"):
            # Inside math mode: clean up accidental tabs/formfeeds/argmin
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


def convert_table(lines: list[str]) -> str:
    """Convert markdown table lines to LaTeX tabularx."""
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
        r"\small",
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


def assemble_latex():
    tables = load_tables()
    md_full = assemble_markdown()

    preamble = r"""\documentclass[11pt,a4paper]{report}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{geometry}
\geometry{a4paper, top=2.5cm, bottom=2.5cm, left=2.5cm, right=2.5cm, headheight=15pt}
\usepackage{graphicx}
\graphicspath{{../results/maintenance_guide_fd004/figures/}{results/maintenance_guide_fd004/figures/}{figures/}}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs,array,tabularx,longtable}
\usepackage{float}
\usepackage{xcolor}
\definecolor{primary}{RGB}{14, 165, 233}
\definecolor{secondary}{RGB}{16, 185, 129}
\definecolor{darkslate}{RGB}{19, 28, 46}
\definecolor{bordercolor}{RGB}{51, 65, 85}

\usepackage{listings}
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

    lines = md_full.splitlines()
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

        # Skip MD title and summary section
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

        # Bibliography section handled separately
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
            if line.startswith("### ") or line.startswith("## ") or line.startswith("# ") or line.startswith("```"):
                in_example = False
                body_tex.append(f"\\begin{{examplebox}}[{escape_latex(example_title)}]")
                body_tex.append(escape_latex("\n".join(example_lines)))
                body_tex.append(r"\end{examplebox}")
                example_lines = []
                # continue processing line outside example
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
                    body_tex.append(r"\begin{codebox}[Listagem de Implementação]")
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

    bib_tex = bibliography.get_latex_bibliography()
    full_tex = preamble + "\n".join(body_tex) + "\n" + bib_tex + "\n\\end{document}\n"
    return full_tex


def main():
    print("[INFO] Assembling full Markdown treatise...")
    md_content = assemble_markdown()
    md_path = DOCS_DIR / "learn.md"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"[SUCCESS] Wrote {md_path} ({len(md_content):,} chars)")

    print("[INFO] Assembling full LaTeX treatise...")
    tex_content = assemble_latex()
    tex_path = DOCS_DIR / "learn.tex"
    tex_path.write_text(tex_content, encoding="utf-8")
    print(f"[SUCCESS] Wrote {tex_path} ({len(tex_content):,} chars)")

    print("[INFO] Compiling learn.tex to learn.pdf using pdflatex (Pass 1)...")
    res1 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory=" + str(DOCS_DIR), str(tex_path)],
        capture_output=True, text=True, cwd=str(DOCS_DIR)
    )
    if res1.returncode != 0:
        print("[WARNING] Pass 1 finished with warnings/errors. Tail of output:")
        for l in res1.stdout.splitlines()[-25:]:
            print("  ", l)

    print("[INFO] Compiling learn.tex to learn.pdf using pdflatex (Pass 2 for TOC, refs and citations)...")
    res2 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory=" + str(DOCS_DIR), str(tex_path)],
        capture_output=True, text=True, cwd=str(DOCS_DIR)
    )

    pdf_path = DOCS_DIR / "learn.pdf"
    if pdf_path.exists():
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"\n=======================================================================")
        print(f"[SUCCESS] BOOK SUCCESSFULLY COMPILED!")
        print(f"Path: {pdf_path.resolve()} ({size_mb:.2f} MB)")
        print(f"=======================================================================\n")
        return 0
    else:
        print("[ERROR] PDF generation failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
