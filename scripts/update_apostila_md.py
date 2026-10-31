#!/usr/bin/env python3
"""
Update docs/learn.md to ensure 100% parity with docs/learn.tex:
All 46 figures embedded with standard markdown image links,
all worked examples, theories, code snippets, and bibliography.
"""

from pathlib import Path
import re

WORKSPACE = Path("/home/leorossi/PdM-CMaps/PdM-CMaps")
MD_PATH = WORKSPACE / "docs" / "learn.md"

def update_markdown():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Chapter 1: Add theory_pipeline_architecture.png
    if "theory_pipeline_architecture.png" not in content:
        ch1_pattern = re.compile(r"```\s*\+--+[\s\S]*?ESTEIRA COMPLETA[\s\S]*?\+--+\s*```", re.MULTILINE)
        ch1_repl = (
            "![Figura 1.1: Arquitetura Metodológica Completa do Ecossistema PdM-CMaps: As Sete Camadas do Sinal à Decisão (Pádua / NASA C-MAPSS FD004).]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/theory_pipeline_architecture.png)"
        )
        content = ch1_pattern.sub(lambda m: ch1_repl, content)

    # 2. Chapter 2: Add theory_turbofan_stations.png and theory_brayton_cycle.png
    if "theory_turbofan_stations.png" not in content:
        ch2_pattern = re.compile(r"```\s*\+--+[\s\S]*?ESTAÇÕES TERMODINÂMICAS[\s\S]*?\+--+\s*```", re.MULTILINE)
        ch2_repl = (
            "![Figura 2.1: Corte Esquemático do Motor Turbofan de Alta Razão de Derivação: Estações Termodinâmicas e Posicionamento dos 21 Sensores de Telemetria.]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/theory_turbofan_stations.png)\n\n"
            "![Figura 2.2: Fundamentação Termodinâmica do Ciclo Brayton Real vs Ideal: Diagramas T-s e P-v demonstrando degradação de compressor e turbina.]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/theory_brayton_cycle.png)"
        )
        content = ch2_pattern.sub(lambda m: ch2_repl, content)

    # 3. Chapter 4: Add healthy_nc_by_regime.png and healthy_bpr_by_regime.png
    if "healthy_nc_by_regime.png" not in content:
        p30_reg_str = f"![Figura 4.2: Validação cruzada do canal de pressurização estática P30 intra-regime durante a fase saudável.](file://{WORKSPACE}/results/maintenance_guide_fd004/figures/healthy_p30_by_regime.png)"
        extra_ch4 = (
            p30_reg_str + "\n\n" +
            "![Figura 4.3: Validação cruzada do canal cinemático Nc (rotação do núcleo N2) intra-regime durante a fase saudável.]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/healthy_nc_by_regime.png)\n\n"
            "![Figura 4.4: Validação cruzada do canal aerodinâmico BPR (razão de bypass) intra-regime durante a fase saudável.]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/healthy_bpr_by_regime.png)"
        )
        content = content.replace(p30_reg_str, extra_ch4)

    # 4. Chapter 5: Add nc_parity, bpr_parity, sensor_actual_vs_predicted_all_channels
    if "nc_parity_envelope.png" not in content:
        p30_par_str = f"![Figura 5.3: Envelope de Paridade de 45 graus para o Sensor Virtual P30 demonstrando dispersão centrada.](file://{WORKSPACE}/results/maintenance_guide_fd004/figures/p30_parity_envelope.png)"
        extra_ch5 = (
            p30_par_str + "\n\n" +
            "![Figura 5.4: Envelope de Paridade de 45 graus para o canal cinemático Nc (rotação do núcleo N2) na calibração saudável.]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/nc_parity_envelope.png)\n\n"
            "![Figura 5.5: Envelope de Paridade de 45 graus para o canal aerodinâmico BPR (razão de derivação bypass) na calibração saudável.]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/bpr_parity_envelope.png)\n\n"
            "![Figura 5.6: Reconstrução Multissensorial Integral do Gêmeo Digital: Trajetória Real vs Previsto ponto a ponto em todos os 21 canais de instrumentação.]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/sensor_actual_vs_predicted_all_channels.png)"
        )
        content = content.replace(p30_par_str, extra_ch5)

    # 5. Chapter 7: Add theory_persistence_fsm.png, geometric envelopes and failure detections
    if "theory_persistence_fsm.png" not in content:
        ch7_pattern = re.compile(r"```\s*\+--+[\s\S]*?MÁQUINA DE ESTADOS FINITA[\s\S]*?\+--+\s*```", re.MULTILINE)
        ch7_repl = (
            "![Figura 7.0: Fundamentação Teórica da Trava de Persistência Industrial (k >= 2) e Máquina de Estados Finitos.]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/theory_persistence_fsm.png)"
        )
        content = ch7_pattern.sub(lambda m: ch7_repl, content)

    if "p30_geometric_envelope_detection.png" not in content:
        t50_geom_str = f"![Figura 7.5: Casco geométrico digital móvel (Rolling Convex Hull, W=15 ciclos) para rastreio de picos e vales.](file://{WORKSPACE}/results/maintenance_guide_fd004/figures/t50_geometric_envelope_detection.png)"
        extra_ch7 = (
            t50_geom_str + "\n\n" +
            "![Figura 7.6: Casco geométrico digital móvel (Rolling Convex Hull, W=15 ciclos) para o canal pneumático P30.]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/p30_geometric_envelope_detection.png)\n\n"
            "![Figura 7.7: Casco geométrico digital móvel (Rolling Convex Hull, W=15 ciclos) para o canal cinemático Nc.]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/nc_geometric_envelope_detection.png)\n\n"
            "![Figura 7.8: Casco geométrico digital móvel (Rolling Convex Hull, W=15 ciclos) para o canal aerodinâmico BPR.]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/bpr_geometric_envelope_detection.png)\n\n"
            "### Cronologia de Detecção de Falha Multissensorial na Vida Completa do Ativo\n\n"
            "![Figura 7.9: Linha do tempo completa de detecção de falha no canal térmico T50 (Alarme no Ciclo 222, Lead Time de +213 ciclos).]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/t50_failure_detection.png)\n\n"
            "![Figura 7.10: Linha do tempo completa de detecção de falha no canal pneumático P30 (Alarme precoce no Ciclo 191, Lead Time de +244 ciclos).]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/p30_failure_detection.png)\n\n"
            "![Figura 7.11: Linha do tempo completa de detecção de falha no canal cinemático Nc (Primeiro alarme físico no Ciclo 187, Lead Time de +248 ciclos).]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/nc_failure_detection.png)\n\n"
            "![Figura 7.12: Linha do tempo completa de detecção de falha no canal aerodinâmico BPR (Alarme no Ciclo 215, Lead Time de +220 ciclos).]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/bpr_failure_detection.png)"
        )
        content = content.replace(t50_geom_str, extra_ch7)

    # 6. Chapter 9: Add minirocket for p30, nc, bpr and rul_minirocket_parity_envelope
    if "minirocket_feature_filtering_p30.png" not in content:
        t50_mini_str = f"![Figura 9.2: Extração convolucional causal MiniRocket aplicada sobre a série térmica T50.](file://{WORKSPACE}/results/maintenance_guide_fd004/figures/minirocket_feature_filtering_t50.png)"
        extra_ch9 = (
            t50_mini_str + "\n\n" +
            "![Figura 9.3: Extração convolucional causal MiniRocket aplicada sobre a série pneumática P30.]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/minirocket_feature_filtering_p30.png)\n\n"
            "![Figura 9.4: Extração convolucional causal MiniRocket aplicada sobre a série cinemática Nc.]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/minirocket_feature_filtering_nc.png)\n\n"
            "![Figura 9.5: Extração convolucional causal MiniRocket aplicada sobre a série aerodinâmica BPR.]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/minirocket_feature_filtering_bpr.png)\n\n"
            "![Figura 9.6: Envelope de Paridade de 45 graus para a previsão direta de RUL com convoluções determinísticas MiniRocket + RidgeCV na frota hold-out.]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/rul_minirocket_parity_envelope.png)"
        )
        content = content.replace(t50_mini_str, extra_ch9)

    # 7. Chapter 10: Add t50_xgboost_raw_prediction_preview.png
    if "t50_xgboost_raw_prediction_preview.png" not in content:
        r2_comp_str = f"![Figura 10.1: Benchmark comparativo de R2 no teste hold-out de todos os modelos avaliados (GBDTs, Ensembles, Redes Neurais, SVM e Transformers).](file://{WORKSPACE}/results/maintenance_guide_fd004/figures/rul_model_r2_comparison.png)"
        extra_ch10 = (
            r2_comp_str + "\n\n" +
            "![Figura 10.2: Inspeção Granular de Desempenho Ponto a Ponto: Previsões brutas e resíduos do modelo XGBoost em todos os 6 regimes operacionais de altitude e Mach para o motor hold-out #179.]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/t50_xgboost_raw_prediction_preview.png)"
        )
        content = content.replace(r2_comp_str, extra_ch10)

    # 8. Chapter 13: Add theory_sop_flowchart.png
    if "theory_sop_flowchart.png" not in content:
        ch13_pattern = re.compile(r"```\s*\+--+[\s\S]*?FLUXOGRAMA OPERACIONAL DO PROCEDIMENTO PADRÃO[\s\S]*?\+--+\s*```", re.MULTILINE)
        ch13_repl = (
            "![Figura 13.1: Procedimento Operacional Padronizado (SOP Industrial): Fluxograma da Matriz de Decisão em 4 Estágios.]"
            f"(file://{WORKSPACE}/results/maintenance_guide_fd004/figures/theory_sop_flowchart.png)"
        )
        content = ch13_pattern.sub(lambda m: ch13_repl, content)

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Updated {MD_PATH} successfully ({len(content):,} characters).")

if __name__ == "__main__":
    update_markdown()
