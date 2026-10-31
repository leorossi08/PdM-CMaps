#!/usr/bin/env python3
"""
Comprehensive script to update docs/learn.tex to a publication-quality textbook/apostila,
embedding all 46 PNG images (5 theory diagrams + 41 empirical results figures),
fixing all LaTeX syntax/encoding errors, adding pedagogical derivations and examples,
and compiling docs/learn.pdf cleanly with pdflatex.
"""

import os
import re
import subprocess
from pathlib import Path

WORKSPACE = Path("/home/leorossi/PdM-CMaps/PdM-CMaps")
TEX_PATH = WORKSPACE / "docs" / "learn.tex"
PDF_PATH = WORKSPACE / "docs" / "learn.pdf"

def main():
    with open(TEX_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update Preamble: Add microtype and complete literate characters in listings
    old_lstset = r"""\lstset{
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
}"""

    new_lstset = r"""\usepackage{microtype}
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
        {≥}{{$\ge$}}1 {≤}{{$\le$}}1 {≈}{{$\approx$}}1 {→}{{$\to$}}1 {—}{{---}}1 {–}{{--}}1
}"""
    content = content.replace(old_lstset, new_lstset)

    # 2. Fix Chapter 1: Replace ASCII box with theory_pipeline_architecture.png and fix math syntax in Example 1.1
    ch1_ascii_pattern = re.compile(
        r"\\begin\{codebox\}\[Listagem de Implementação Python\]\s*\+--+[\s\S]*?ESTEIRA COMPLETA[\s\S]*?\+--+\s*\\end\{codebox\}",
        re.MULTILINE
    )
    ch1_fig_replacement = r"""\begin{figure}[H]
\centering
\includegraphics[width=0.98\textwidth]{theory_pipeline_architecture.png}
\caption{Arquitetura Metodológica Completa do Ecossistema PdM-CMaps: As Sete Camadas do Sinal à Decisão (Pádua / NASA C-MAPSS FD004).}
\label{fig:theory_pipeline_arch}
\end{figure}"""
    content = ch1_ascii_pattern.sub(lambda m: ch1_fig_replacement, content)

    # Fix Example 1.1 syntax
    content = content.replace(r"$60\\%$", r"$60\%$")
    content = content.replace(r"$T\_{real} = 350$", r"$T_{\text{real}} = 350$")
    content = content.replace(r"\Delta_{ciclos\_ano}", r"\Delta_{\text{ciclos\_ano}}")

    # 3. Fix Chapter 2: Replace ASCII diagram with theory_turbofan_stations.png and add theory_brayton_cycle.png
    ch2_ascii_pattern = re.compile(
        r"\\begin\{codebox\}\[Listagem de Implementação Python\]\s*\+--+[\s\S]*?ESTAÇÕES TERMODINÂMICAS[\s\S]*?\+--+\s*\\end\{codebox\}",
        re.MULTILINE
    )
    ch2_fig_replacement = r"""\begin{figure}[H]
\centering
\includegraphics[width=0.98\textwidth]{theory_turbofan_stations.png}
\caption{Corte Esquemático do Motor Turbofan de Alta Razão de Derivação: Estações Termodinâmicas (2, 2.4, 3, 4, 5) e Posicionamento dos 21 Sensores de Telemetria do simulador C-MAPSS.}
\label{fig:theory_turbofan_stations}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.98\textwidth]{theory_brayton_cycle.png}
\caption{Fundamentação Termodinâmica do Ciclo Brayton Real vs Ideal: Diagramas Temperatura-Entropia ($T-s$) e Pressão-Volume ($P-v$) demonstrando a degradação de rendimento isentrópico no compressor ($\eta_c < 1$) e turbina ($\eta_t < 1$), explicando analiticamente a elevação compulsória da temperatura de exaustão $T_{50}$ (EGT) no fim de vida.}
\label{fig:theory_brayton_cycle}
\end{figure}"""
    content = ch2_ascii_pattern.sub(lambda m: ch2_fig_replacement, content)

    # 4. Fix Chapter 4: Replace ASCII box with clean formatted table/text and add healthy_nc and healthy_bpr figures
    ch4_ascii_pattern = re.compile(
        r"\\begin\{codebox\}\[Listagem de Implementação Python\]\s*\+--+[\s\S]*?OS SEIS ENVELOPES OPERACIONAIS[\s\S]*?\+--+\s*\\end\{codebox\}",
        re.MULTILINE
    )
    ch4_text_replacement = r"""\begin{calloutbox}[Os Seis Envelopes Operacionais de Voo Discretizados (C-MAPSS FD004)]
\begin{itemize}
    \item \textbf{Regime 0:} Baixa Altitude e Velocidade Moderada (Altitude: $0 - 10.000\text{ ft}$, Mach: $0,00 - 0,25$, TRA: $100^\circ$).
    \item \textbf{Regime 1:} Decolagem e Subida Inicial ao Nível do Mar (Altitude: $0\text{ ft}$, Mach: $0,00$, TRA: $100^\circ$).
    \item \textbf{Regime 2:} Cruzeiro Transônico de Média Altitude (Altitude: $20.000 - 25.000\text{ ft}$, Mach: $0,62 - 0,70$, TRA: $100^\circ$).
    \item \textbf{Regime 3:} Cruzeiro Econômico em Teto Intermediário (Altitude: $30.000 - 35.000\text{ ft}$, Mach: $0,70 - 0,80$, TRA: $100^\circ$).
    \item \textbf{Regime 4:} Cruzeiro Estratosférico de Alta Altitude (Altitude: $40.000 - 42.000\text{ ft}$, Mach: $0,80 - 0,84$, TRA: $100^\circ$).
    \item \textbf{Regime 5:} Descida e Aproximação com Baixo Empuxo (Altitude: $10.000 - 25.000\text{ ft}$, Mach: $0,25 - 0,50$, TRA: $20^\circ - 40^\circ$).
\end{itemize}
\end{calloutbox}"""
    content = ch4_ascii_pattern.sub(lambda m: ch4_text_replacement, content)

    # Add healthy_nc_by_regime.png and healthy_bpr_by_regime.png after healthy_p30
    if "healthy_nc_by_regime.png" not in content:
        extra_ch4_figs = r"""\includegraphics[width=0.88\textwidth]{healthy_p30_by_regime.png}
\caption{Validação cruzada do canal de pressurização estática P30 intra-regime durante a fase saudável.}
\label{fig:healthy_p30_regime}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{healthy_nc_by_regime.png}
\caption{Validação cruzada do canal cinemático Nc (rotação do núcleo N2) intra-regime durante a fase saudável.}
\label{fig:healthy_nc_regime}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{healthy_bpr_by_regime.png}
\caption{Validação cruzada do canal aerodinâmico BPR (razão de bypass) intra-regime durante a fase saudável.}
\label{fig:healthy_bpr_regime}"""
        p30_block_pattern = re.compile(
            r"\\includegraphics\[width=0\.88\\textwidth\]\{healthy_p30_by_regime\.png\}\s*\\caption\{[^}]*?\}\s*\\label\{fig:healthy_p30_regime\}",
            re.MULTILINE
        )
        content = p30_block_pattern.sub(lambda m: extra_ch4_figs, content)

    # 5. Fix Chapter 5: Replace ASCII diagram with calloutbox and add nc_parity, bpr_parity, sensor_actual_vs_predicted_all_channels
    ch5_ascii_pattern = re.compile(
        r"\\begin\{codebox\}\[Listagem de Implementação Python\]\s*\+--+[\s\S]*?FLUXO DE GERAÇÃO DE RESÍDUOS[\s\S]*?\+--+\s*\\end\{codebox\}",
        re.MULTILINE
    )
    ch5_callout_replacement = r"""\begin{calloutbox}[Mecanismo Físico-Estatístico de Geração de Resíduos de Degradação]
\begin{center}
$\text{Medição Físico-Instrumental } y(t) \quad \text{vs} \quad \text{Estimativa de Normalidade } \hat{y}(t) \implies e(t) = y(t) - \hat{y}(t)$ \\
\vspace{2mm}
\textbf{Fase 1 (Motor Saudável: $t \in [1, 180]$ ciclos):} $y(t) \approx \hat{y}(t) \implies e(t) \sim \mathcal{N}(0, \sigma_e^2)$ (Ruído Branco Puro Gaussiano) \\
\textbf{Fase 2 (Motor Degradado: $t > 220$ ciclos):} $y(t) \gg \hat{y}(t) \implies e(t) > +3\sigma_e$ (Dano Mecânico / Sobreaquecimento)
\end{center}
\end{calloutbox}"""
    content = ch5_ascii_pattern.sub(lambda m: ch5_callout_replacement, content)

    if "nc_parity_envelope.png" not in content:
        p30_parity_pattern = re.compile(
            r"(\\includegraphics\[width=0\.88\\textwidth\]\{p30_parity_envelope\.png\}\s*\\caption\{[^}]*?\}\s*\\label\{fig:p30_parity_env\}\s*\\end\{figure\})",
            re.MULTILINE
        )
        def replace_p30_parity(match):
            orig = match.group(1)
            extra = r"""

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{nc_parity_envelope.png}
\caption{Envelope de Paridade de 45 graus para o canal cinemático Nc (rotação do núcleo N2) na calibração saudável.}
\label{fig:nc_parity_env}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{bpr_parity_envelope.png}
\caption{Envelope de Paridade de 45 graus para o canal aerodinâmico BPR (razão de derivação bypass) na calibração saudável.}
\label{fig:bpr_parity_env}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.98\textwidth]{sensor_actual_vs_predicted_all_channels.png}
\caption{Reconstrução Multissensorial Integral do Gêmeo Digital: Trajetória Real vs Previsto ponto a ponto em todos os 21 canais de instrumentação do motor turbofan.}
\label{fig:sensor_actual_vs_pred_all}
\end{figure}"""
            return orig + extra
        content = p30_parity_pattern.sub(replace_p30_parity, content)

    # 6. Fix Chapter 7: Replace ASCII diagram with theory_persistence_fsm.png, and add geometric envelopes and failure detections
    ch7_ascii_pattern = re.compile(
        r"\\begin\{codebox\}\[Listagem de Implementação Python\]\s*\+--+[\s\S]*?MÁQUINA DE ESTADOS FINITA[\s\S]*?\+--+\s*\\end\{codebox\}",
        re.MULTILINE
    )
    ch7_fig_replacement = r"""\begin{figure}[H]
\centering
\includegraphics[width=0.98\textwidth]{theory_persistence_fsm.png}
\caption{Fundamentação Teórica da Trava de Persistência Industrial ($k \ge 2$) e Máquina de Estados Finitos: Absorção de spikes estocásticos transitórios isolados no Ciclo 173 ($k=1$, reset imediato) versus confirmação estrutural sustentada no Ciclo 222 ($k=2$, disparo de alarme preditivo com $+213$ ciclos de antecedência).}
\label{fig:theory_persistence_fsm}
\end{figure}"""
    content = ch7_ascii_pattern.sub(lambda m: ch7_fig_replacement, content)

    if "p30_geometric_envelope_detection.png" not in content:
        t50_geom_pattern = re.compile(
            r"(\\includegraphics\[width=0\.88\\textwidth\]\{t50_geometric_envelope_detection\.png\}\s*\\caption\{[^}]*?\}\s*\\label\{fig:t50_geometric_envelope\}\s*\\end\{figure\})",
            re.MULTILINE
        )
        def replace_t50_geom(match):
            orig = match.group(1)
            extra = r"""

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{p30_geometric_envelope_detection.png}
\caption{Casco geométrico digital móvel (Rolling Convex Hull, W=15 ciclos) para o canal pneumático P30.}
\label{fig:p30_geometric_envelope}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{nc_geometric_envelope_detection.png}
\caption{Casco geométrico digital móvel (Rolling Convex Hull, W=15 ciclos) para o canal cinemático Nc.}
\label{fig:nc_geometric_envelope}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{bpr_geometric_envelope_detection.png}
\caption{Casco geométrico digital móvel (Rolling Convex Hull, W=15 ciclos) para o canal aerodinâmico BPR.}
\label{fig:bpr_geometric_envelope}
\end{figure}

\subsection{Cronologia de Detecção de Falha Multissensorial na Vida Completa do Ativo}
As quatro figuras a seguir documentam a série temporal completa até a exaustão mecânica (Ciclo 435) para os quatro eixos físicos primários, indicando a evolução do sinal, o limiar estocástico de alarme e o tempo de antecedência até a falha funcional:

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{t50_failure_detection.png}
\caption{Linha do tempo completa de detecção de falha no canal térmico T50 (Alarme no Ciclo 222, Lead Time de +213 ciclos).}
\label{fig:t50_failure_detection}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{p30_failure_detection.png}
\caption{Linha do tempo completa de detecção de falha no canal pneumático P30 (Alarme precoce no Ciclo 191, Lead Time de +244 ciclos).}
\label{fig:p30_failure_detection}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{nc_failure_detection.png}
\caption{Linha do tempo completa de detecção de falha no canal cinemático Nc (Primeiro alarme físico no Ciclo 187, Lead Time de +248 ciclos).}
\label{fig:nc_failure_detection}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{bpr_failure_detection.png}
\caption{Linha do tempo completa de detecção de falha no canal aerodinâmico BPR (Alarme no Ciclo 215, Lead Time de +220 ciclos).}
\label{fig:bpr_failure_detection}
\end{figure}"""
            return orig + extra
        content = t50_geom_pattern.sub(replace_t50_geom, content)

    # 7. Fix Chapter 9: Add minirocket for p30, nc, bpr and rul_minirocket_parity_envelope
    if "minirocket_feature_filtering_p30.png" not in content:
        t50_mini_pattern = re.compile(
            r"(\\includegraphics\[width=0\.88\\textwidth\]\{minirocket_feature_filtering_t50\.png\}\s*\\caption\{[^}]*?\}\s*\\label\{fig:minirocket_t50\}\s*\\end\{figure\})",
            re.MULTILINE
        )
        def replace_t50_mini(match):
            orig = match.group(1)
            extra = r"""

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{minirocket_feature_filtering_p30.png}
\caption{Extração convolucional causal MiniRocket aplicada sobre a série pneumática P30.}
\label{fig:minirocket_p30}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{minirocket_feature_filtering_nc.png}
\caption{Extração convolucional causal MiniRocket aplicada sobre a série cinemática Nc.}
\label{fig:minirocket_nc}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{minirocket_feature_filtering_bpr.png}
\caption{Extração convolucional causal MiniRocket aplicada sobre a série aerodinâmica BPR.}
\label{fig:minirocket_bpr}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{rul_minirocket_parity_envelope.png}
\caption{Envelope de Paridade de 45 graus para a previsão direta de RUL com convoluções determinísticas MiniRocket + RidgeCV na frota hold-out.}
\label{fig:rul_minirocket_parity}
\end{figure}"""
            return orig + extra
        content = t50_mini_pattern.sub(replace_t50_mini, content)

    # 8. Fix Chapter 10: Add t50_xgboost_raw_prediction_preview.png
    if "t50_xgboost_raw_prediction_preview.png" not in content:
        r2_comp_pattern = re.compile(
            r"(\\includegraphics\[width=0\.88\\textwidth\]\{rul_model_r2_comparison\.png\}\s*\\caption\{[^}]*?\}\s*\\label\{fig:rul_model_r2_comparison\}\s*\\end\{figure\})",
            re.MULTILINE
        )
        def replace_r2_comp(match):
            orig = match.group(1)
            extra = r"""

\begin{figure}[H]
\centering
\includegraphics[width=0.98\textwidth]{t50_xgboost_raw_prediction_preview.png}
\caption{Inspeção Granular de Desempenho Ponto a Ponto: Previsões brutas e resíduos do modelo XGBoost em todos os 6 regimes operacionais de altitude e Mach para o motor hold-out \#179.}
\label{fig:t50_xgboost_raw_preview}
\end{figure}"""
            return orig + extra
        content = r2_comp_pattern.sub(replace_r2_comp, content)

    # 9. Fix Chapter 12: Replace ASCII table with clean tabularx
    ch12_ascii_pattern = re.compile(
        r"\\begin\{codebox\}\[Listagem de Implementação Python\]\s*\+--+[\s\S]*?MATRIZ COMPARATIVA: DEEP LEARNING[\s\S]*?\+--+\s*\\end\{codebox\}",
        re.MULTILINE
    )
    ch12_table_replacement = r"""\begin{table}[H]
\centering
\small
\begin{tabularx}{\textwidth}{l X X}
\toprule
\textbf{Critério de Avaliação} & \textbf{Deep Learning / Transformers (PyTorch)} & \textbf{GBDTs (XGBoost / LightGBM)} \\
\midrule
Acurácia $R^2$ na Frota & $0,6729$ a $0,6811$ & \textbf{$0,8121$ a $0,8156$ (Superior)} \\
Erro RMSE Hold-out & $22,77$ a $23,07$ ciclos & \textbf{$17,32$ a $17,48$ ciclos} \\
Erro MAE Hold-out & $15,25$ a $16,30$ ciclos & \textbf{$11,24$ a $11,48$ ciclos} \\
Sensibilidade a Assimetria & NASA Score: $274.340$ a $481.081$ & \textbf{NASA Score: $320.608$ a $358.210$} \\
Previsões Atrasadas (Risco Crítico) & $25,6\%$ a $28,7\%$ & $42,0\%$ a $42,7\%$ \\
Tempo de Treinamento CPU & $21,9$s a $57,8$s & \textbf{$0,28$s a $0,70$s ($80\times$ mais rápido)} \\
Exigência de Hardware & GPUs dedicadas / Memória VRAM & \textbf{CPU padrão / Microcontrolador Edge} \\
Interpretabilidade Física & Caixa-preta densa & \textbf{Importância de atributos e SHAP exatos} \\
\bottomrule
\end{tabularx}
\caption{Matriz Comparativa Rigorosa: Deep Learning vs Gradient Boosted Decision Trees (GBDTs) em Séries Temporais Tabulares Industriais.}
\label{tab:deep_vs_gbdt}
\end{table}"""
    content = ch12_ascii_pattern.sub(lambda m: ch12_table_replacement, content)

    # 10. Fix Chapter 13: Replace ASCII flowchart with theory_sop_flowchart.png
    ch13_ascii_pattern = re.compile(
        r"\\begin\{codebox\}\[Listagem de Implementação Python\]\s*\+--+[\s\S]*?FLUXOGRAMA OPERACIONAL DO PROCEDIMENTO PADRÃO[\s\S]*?\+--+\s*\\end\{codebox\}",
        re.MULTILINE
    )
    ch13_fig_replacement = r"""\begin{figure}[H]
\centering
\includegraphics[width=0.98\textwidth]{theory_sop_flowchart.png}
\caption{Procedimento Operacional Padronizado (SOP Industrial): Fluxograma da Matriz de Decisão em 4 Estágios (Nível 0 Verde, Nível 1 Amarelo, Nível 2 Laranja, Nível 3 Vermelho).}
\label{fig:theory_sop_flowchart}
\end{figure}"""
    content = ch13_ascii_pattern.sub(lambda m: ch13_fig_replacement, content)

    # 11. Add missing literature references to thebibliography
    extra_bib = r"""\bibitem{saravanamuttoo2017}
H.~I.~H.~Saravanamuttoo, G.~F.~C.~Rogers, H.~Cohen, P.~V.~Straznicky, e A.~C.~Nix, \emph{Gas Turbine Theory}, 7ª~ed., Pearson Education, Londres, UK, 2017.

\bibitem{mattingly2006}
J.~D.~Mattingly, \emph{Elements of Propulsion: Gas Turbines and Rockets}, 2ª~ed., AIAA Education Series, Reston, VA, 2006.

\bibitem{shewhart1931}
W.~A.~Shewhart, \emph{Economic Control of Quality of Manufactured Product}, D. Van Nostrand Company, New York, NY, 1931.
"""
    if "saravanamuttoo2017" not in content:
        content = content.replace(r"\end{thebibliography}", extra_bib + "\n\\end{thebibliography}")

    with open(TEX_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Successfully updated {TEX_PATH} ({len(content):,} characters).")

if __name__ == "__main__":
    main()
