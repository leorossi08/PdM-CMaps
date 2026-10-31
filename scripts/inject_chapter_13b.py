#!/usr/bin/env python3
"""
Inject Chapter 13B into docs/learn.tex from scripts/chapter_13b.tex
and append the new references to the bibliography.
"""

from pathlib import Path

WORKSPACE = Path("/home/leorossi/PdM-CMaps/PdM-CMaps")
TEX_PATH = WORKSPACE / "docs" / "learn.tex"
CHAPTER_13B_TEX_FILE = WORKSPACE / "scripts" / "chapter_13b.tex"

def main():
    with open(TEX_PATH, "r", encoding="utf-8") as f:
        tex_content = f.read()

    with open(CHAPTER_13B_TEX_FILE, "r", encoding="utf-8") as f:
        chapter_content = f.read()

    if "Capítulo 13B: Perspectiva de Mercado" in tex_content:
        print("Chapter 13B is already present in learn.tex!")
        return

    split_target = r"\chapter{Capítulo 14: Catálogo Técnico e Síntese da Evidência Empírica}"
    if split_target not in tex_content:
        raise ValueError("Could not find Chapter 14 split target in learn.tex")

    tex_content = tex_content.replace(split_target, chapter_content + "\n\n" + split_target)

    bib_entries = r"""\bibitem{oliverwyman2024}
Oliver~Wyman, \emph{Global Fleet and MRO Market Forecast 2024--2034}, Oliver Wyman Aviation Consulting, New York, NY, 2024.

\bibitem{vanhorenbeek2013}
A.~Van~Horenbeek e L.~Pintelon, ``A dynamic predictive maintenance policy for complex multi-component systems,'' \emph{Reliability Engineering \& System Safety}, vol.~120, pp.~39--50, 2013.

\bibitem{oasis_mqtt2019}
OASIS, \emph{MQTT Version 5.0 Specification}, OASIS Standard, Burlington, MA, 2019.

\bibitem{chao2022}
M.~A.~Chao, C.~Kulkarni, K.~Goebel, e O.~Fink, ``Aircraft engine run-to-failure dataset under real flight conditions for prognostics and health management,'' \emph{Data in Brief}, vol.~40, p.~107648, 2022.
"""

    if "oliverwyman2024" not in tex_content:
        tex_content = tex_content.replace(r"\end{thebibliography}", bib_entries + "\n\\end{thebibliography}")

    with open(TEX_PATH, "w", encoding="utf-8") as f:
        f.write(tex_content)

    print("Successfully injected Chapter 13B and bibliography entries into docs/learn.tex.")

if __name__ == "__main__":
    main()
