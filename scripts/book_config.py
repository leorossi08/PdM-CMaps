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
from book_tables import load_tables

tables = load_tables()

# Common image captions and labels
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

print("Base configuration loaded.")
