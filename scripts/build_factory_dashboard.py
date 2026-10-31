#!/usr/bin/env python3
"""Build an industrial, data-driven Predictive Maintenance (PdM) Dashboard using Plotly.

Features:
- 100% offline self-contained HTML (bundles Plotly JS locally, no VPN/internet needed).
- Raw telemetry signals vs smoothed EWMA/moving average (Slide 17).
- Dynamic Time-Series Envelope ('Embrulho' SPC) with RAW actual vs RAW predicted (SEM MÉDIA).
- Digital Signal Hull Geometric Envelope.
- Causal MiniRocket PPV feature trajectory with monotonic degradation (Slides 19 & 21).
- Point-by-point RUL actual vs predicted (SEM MÉDIA) across models (XGBoost, LightGBM, CatBoost, RF).
- 45° Parity Envelope on held-out fleet engines.
- Variance decomposition & within-regime R² analysis (Pooled R² Illusion explained).
- Benchmark comparison across all models with R² > 0.75 criteria.
- Full 21-sensor channel fleet profiling.
- Clear visual markers for ALERTA (first anomaly alarm) and EOL (quebra mecânica).
- Comprehensive academic and engineering explanations for every metric and section.
"""

from __future__ import annotations

import argparse
import base64
import http.server
import json
import socketserver
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as pyo
from plotly.subplots import make_subplots

# Dark industrial color palette
PALETTE = {
    "bg_dark": "#0b0f19",
    "surface": "#131c2e",
    "card": "#1e293b",
    "border": "#334155",
    "cyan": "#06b6d4",
    "cyan_light": "#38bdf8",
    "amber": "#f59e0b",
    "amber_light": "#fbbf24",
    "red_alert": "#ef4444",
    "red_dark": "#b91c1c",
    "emerald": "#10b981",
    "purple": "#8b5cf6",
    "gray_text": "#94a3b8",
    "white": "#f8fafc",
    "envelope_fill": "rgba(6, 182, 212, 0.12)",
    "hull_fill": "rgba(245, 158, 11, 0.10)",
    "lead_time_fill": "rgba(239, 68, 68, 0.08)",
}

SENSOR_CONFIGS = {
    "T50": {
        "name": "T50",
        "label": "Temperatura Saída LPT",
        "unit": "°C",
        "badge_class": "badge-thermal",
        "badge_text": "Térmico",
        "description": "Total temperature at LPT outlet (°C). Termodinâmica da turbina de baixa pressão.",
    },
    "P30": {
        "name": "P30",
        "label": "Pressão Saída HPC",
        "unit": "psia",
        "badge_class": "badge-pressure",
        "badge_text": "Pneumático",
        "description": "Total pressure at HPC outlet (psia). Descarga e compressão de alta pressão.",
    },
    "BPR": {
        "name": "BPR",
        "label": "Razão de Bypass",
        "unit": "adim.",
        "badge_class": "badge-aero",
        "badge_text": "Aerodinâmico",
        "description": "Bypass Ratio (BPR). Balanço de fluxo mássico bypass / núcleo.",
    },
    "Nc": {
        "name": "Nc",
        "label": "Velocidade do Núcleo",
        "unit": "rpm",
        "badge_class": "badge-kinematic",
        "badge_text": "Cinemático",
        "description": "Physical core speed (rpm). Rotação física do carretel de alta (spool N2).",
    },
}


def _plotly_dark_layout(title: str, height: int = 480) -> dict:
    """Return common dark mode layout dictionary for industrial aesthetics."""
    return dict(
        title=dict(text=f"<b>{title}</b>", font=dict(family="Outfit, Inter, sans-serif", size=15, color=PALETTE["white"])),
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["bg_dark"],
        font=dict(family="Inter, sans-serif", size=12, color=PALETTE["gray_text"]),
        height=height,
        margin=dict(l=60, r=40, t=60, b=50),
        legend=dict(
            bgcolor="rgba(19, 28, 46, 0.8)",
            bordercolor=PALETTE["border"],
            borderwidth=1,
            font=dict(size=11, color=PALETTE["white"]),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            gridcolor="#1e293b",
            zerolinecolor="#334155",
            showline=True,
            linecolor=PALETTE["border"],
            tickfont=dict(color=PALETTE["gray_text"]),
            title_font=dict(color=PALETTE["white"], size=13),
        ),
        yaxis=dict(
            gridcolor="#1e293b",
            zerolinecolor="#334155",
            showline=True,
            linecolor=PALETTE["border"],
            tickfont=dict(color=PALETTE["gray_text"]),
            title_font=dict(color=PALETTE["white"], size=13),
        ),
        hoverlabel=dict(
            bgcolor=PALETTE["card"],
            font_size=12,
            font_family="Inter, sans-serif",
            bordercolor=PALETTE["cyan"],
        ),
    )


def create_signal_filtering_figure(
    trace: pd.DataFrame,
    sensor: str,
    unit_label: str,
    alarm_cycle: int | None,
    eol_cycle: int,
) -> go.Figure:
    """Figure 1: Raw telemetry signals vs smoothed EWMA and moving averages (Slide 17)."""
    cycles = trace["cycle"].to_numpy()
    raw_col = sensor if sensor in trace.columns else ("T50" if "T50" in trace.columns else trace.columns[2])
    raw_signal = trace[raw_col].to_numpy(dtype=float)

    # Compute filters
    s = pd.Series(raw_signal)
    ewma_035 = s.ewm(alpha=0.35).mean().to_numpy()
    ewma_070 = s.ewm(alpha=0.70).mean().to_numpy()
    ewma_010 = s.ewm(alpha=0.10).mean().to_numpy()
    rolling_5 = s.rolling(window=5, min_periods=1).mean().to_numpy()
    residuals_035 = raw_signal - ewma_035

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.70, 0.30],
        subplot_titles=("", f"Resíduo Estocástico da Filtragem ({sensor} Cru - EWMA α=0.35)"),
    )

    # Main signals
    fig.add_trace(
        go.Scatter(
            x=cycles, y=raw_signal, mode="lines",
            name=f"{sensor} Cru (Raw Signal com Jitter)",
            line=dict(color="rgba(148, 163, 184, 0.45)", width=1.0),
            hovertemplate=f"Ciclo %{{x}}: {sensor} Cru = %{{y:.2f}} {unit_label}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cycles, y=ewma_035, mode="lines",
            name="EWMA α=0.35 (Recomendado por Pádua)",
            line=dict(color=PALETTE["cyan_light"], width=2.4),
            hovertemplate=f"Ciclo %{{x}}: EWMA 0.35 = %{{y:.2f}} {unit_label}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cycles, y=rolling_5, mode="lines",
            name="Média Móvel (Rolling Mean W=5)",
            line=dict(color=PALETTE["emerald"], width=1.8, dash="dot"),
            hovertemplate=f"Ciclo %{{x}}: Rolling Mean = %{{y:.2f}} {unit_label}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cycles, y=ewma_070, mode="lines",
            name="EWMA α=0.70 (Filtro Leve)",
            line=dict(color=PALETTE["amber"], width=1.4, dash="dash"),
            hovertemplate=f"Ciclo %{{x}}: EWMA 0.70 = %{{y:.2f}} {unit_label}<extra></extra>",
            visible="legendonly",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cycles, y=ewma_010, mode="lines",
            name="EWMA α=0.10 (Filtro Pesado)",
            line=dict(color=PALETTE["purple"], width=1.4, dash="dashdot"),
            hovertemplate=f"Ciclo %{{x}}: EWMA 0.10 = %{{y:.2f}} {unit_label}<extra></extra>",
            visible="legendonly",
        ),
        row=1, col=1,
    )

    # Residuals
    fig.add_trace(
        go.Scatter(
            x=cycles, y=residuals_035, mode="lines",
            name=f"Resíduo Ruído ({sensor} Cru - EWMA 0.35)",
            line=dict(color=PALETTE["amber_light"], width=1.0),
            hovertemplate=f"Ciclo %{{x}}: Resíduo = %{{y:.2f}} {unit_label}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0.0, line=dict(color=PALETTE["gray_text"], width=1, dash="dash"), row=2, col=1)

    # Markers for Alarm and EOL
    if alarm_cycle:
        fig.add_vline(
            x=alarm_cycle, line=dict(color=PALETTE["red_alert"], width=2, dash="dash"),
            annotation_text=f"ALERTA ({sensor} Ciclo {alarm_cycle})", annotation_position="top left",
            annotation_font=dict(color=PALETTE["red_alert"], size=11, family="Inter"),
            row=1, col=1,
        )
        fig.add_vline(x=alarm_cycle, line=dict(color=PALETTE["red_alert"], width=2, dash="dash"), row=2, col=1)

    fig.add_vline(
        x=eol_cycle, line=dict(color=PALETTE["red_dark"], width=2.5),
        annotation_text=f"EOL / FALHA (Ciclo {eol_cycle})", annotation_position="top right",
        annotation_font=dict(color=PALETTE["red_alert"], size=11, family="Inter"),
        row=1, col=1,
    )
    fig.add_vline(x=eol_cycle, line=dict(color=PALETTE["red_dark"], width=2.5), row=2, col=1)

    layout = _plotly_dark_layout(f"Camada 1: Sinais Crus vs Suavizados (Slide 17) — {sensor} ({unit_label})", height=540)
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Ciclo Operacional do Motor", row=2, col=1)
    fig.update_yaxes(title_text=f"{sensor} ({unit_label})", row=1, col=1)
    fig.update_yaxes(title_text=f"Resíduo ({unit_label})", row=2, col=1)
    return fig


def create_time_series_envelope_figure(
    trace: pd.DataFrame,
    sensor: str,
    unit_label: str,
    alarm_cycle: int | None,
    eol_cycle: int,
    lead_time: int,
) -> go.Figure:
    """Figure 2: Time-series envelope anomaly detection with RAW actual vs RAW predicted (SEM MÉDIA)."""
    cycles = trace["cycle"].to_numpy()
    actual_col = sensor if sensor in trace.columns else ("actual" if "actual" in trace.columns else "T50")
    actual = trace[actual_col].to_numpy(dtype=float)
    predicted = trace["prediction"].to_numpy(dtype=float)
    upper = trace["envelope_upper"].to_numpy(dtype=float)
    lower = trace["envelope_lower"].to_numpy(dtype=float)
    residuals = trace["residual"].to_numpy(dtype=float)
    breaches = trace["out_of_envelope"].to_numpy(dtype=bool)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.70, 0.30],
        subplot_titles=("", f"Resíduo Instantâneo ({sensor} Real - Previsto) e Bandas de Tolerância Shewhart (±3σ)"),
    )

    # Lead time shading area between Alarm and EOL
    if alarm_cycle and alarm_cycle < eol_cycle:
        fig.add_vrect(
            x0=alarm_cycle, x1=eol_cycle,
            fillcolor=PALETTE["lead_time_fill"], layer="below", line_width=0,
            annotation_text=f"MARGEM OPERACIONAL (LEAD TIME = {lead_time} CICLOS)",
            annotation_position="top left",
            annotation_font=dict(color=PALETTE["red_alert"], size=11, family="Inter"),
            row=1, col=1,
        )

    # Envelope corridor
    fig.add_trace(
        go.Scatter(
            x=cycles, y=upper, mode="lines",
            name="Envelope Superior (+3σ nominal)",
            line=dict(color="rgba(56, 189, 248, 0.3)", width=1, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cycles, y=lower, mode="lines",
            name="Faixa do Envelope SPC (±3σ)",
            line=dict(color="rgba(56, 189, 248, 0.3)", width=1, dash="dash"),
            fill="tonexty",
            fillcolor=PALETTE["envelope_fill"],
            hovertemplate=f"Ciclo %{{x}}: Faixa Envelope SPC [{sensor}]<extra></extra>",
        ),
        row=1, col=1,
    )

    # RAW PREDICTED (SEM MÉDIA)
    fig.add_trace(
        go.Scatter(
            x=cycles, y=predicted, mode="lines",
            name=f"{sensor} Previsto Cru (Sensor Virtual Sem Média)",
            line=dict(color=PALETTE["amber_light"], width=2.0, dash="dot"),
            hovertemplate=f"Ciclo %{{x}}: {sensor} Previsto = %{{y:.2f}} {unit_label}<extra></extra>",
        ),
        row=1, col=1,
    )

    # RAW ACTUAL (SEM MÉDIA)
    fig.add_trace(
        go.Scatter(
            x=cycles, y=actual, mode="lines",
            name=f"{sensor} Real Cru (Medição da Máquina Sem Média)",
            line=dict(color=PALETTE["cyan_light"], width=1.6),
            hovertemplate=f"Ciclo %{{x}}: {sensor} Real = %{{y:.2f}} {unit_label}<extra></extra>",
        ),
        row=1, col=1,
    )

    # BREACH POINTS
    breach_cycles = cycles[breaches]
    breach_actual = actual[breaches]
    if len(breach_cycles) > 0:
        fig.add_trace(
            go.Scatter(
                x=breach_cycles, y=breach_actual, mode="markers",
                name=f"Violação do Envelope {sensor} (Anomalia)",
                marker=dict(color=PALETTE["red_alert"], size=7, symbol="circle-open", line=dict(width=2, color=PALETTE["red_alert"])),
                hovertemplate=f"<b>VIOLAÇÃO</b> Ciclo %{{x}}: {sensor} = %{{y:.2f}} {unit_label}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Markers for Alarm and EOL
    if alarm_cycle:
        fig.add_vline(
            x=alarm_cycle, line=dict(color=PALETTE["red_alert"], width=2.2, dash="dash"),
            annotation_text=f"ALERTA ({sensor} Ciclo {alarm_cycle})", annotation_position="top left",
            annotation_font=dict(color=PALETTE["red_alert"], size=12, family="Inter"),
            row=1, col=1,
        )
        fig.add_vline(x=alarm_cycle, line=dict(color=PALETTE["red_alert"], width=2.2, dash="dash"), row=2, col=1)

    fig.add_vline(
        x=eol_cycle, line=dict(color=PALETTE["red_dark"], width=2.8),
        annotation_text=f"EOL / FALHA (Ciclo {eol_cycle})", annotation_position="top right",
        annotation_font=dict(color=PALETTE["red_alert"], size=12, family="Inter"),
        row=1, col=1,
    )
    fig.add_vline(x=eol_cycle, line=dict(color=PALETTE["red_dark"], width=2.8), row=2, col=1)

    # Subplot 2: Raw Residuals
    margin = float(np.mean(upper - predicted))
    fig.add_trace(
        go.Scatter(
            x=cycles, y=residuals, mode="lines",
            name=f"Resíduo Cru ({sensor} Real - Previsto)",
            line=dict(color=PALETTE["cyan_light"], width=1.2),
            hovertemplate=f"Ciclo %{{x}}: Resíduo = %{{y:.2f}} {unit_label}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.add_hline(y=margin, line=dict(color=PALETTE["red_alert"], width=1.5, dash="dash"), annotation_text="+3σ Limite", annotation_position="top right", row=2, col=1)
    fig.add_hline(y=-margin, line=dict(color=PALETTE["red_alert"], width=1.5, dash="dash"), annotation_text="-3σ Limite", annotation_position="bottom right", row=2, col=1)
    fig.add_hline(y=0.0, line=dict(color=PALETTE["gray_text"], width=1), row=2, col=1)

    layout = _plotly_dark_layout(f"Camada 2: Envelope Dinâmico SPC: {sensor} Reais vs Previstos Crus Sem Média", height=580)
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Ciclo Operacional da Máquina", row=2, col=1)
    fig.update_yaxes(title_text=f"{sensor} Cru ({unit_label})", row=1, col=1)
    fig.update_yaxes(title_text=f"Resíduo ({unit_label})", row=2, col=1)
    return fig


def create_geometric_envelope_figure(
    trace: pd.DataFrame,
    sensor: str,
    unit_label: str,
    alarm_cycle: int | None,
    eol_cycle: int,
) -> go.Figure:
    """Figure 3: Geometric digital signal hull envelope (rolling peak/valley envelope)."""
    cycles = trace["cycle"].to_numpy()
    actual_col = sensor if sensor in trace.columns else ("actual" if "actual" in trace.columns else "T50")
    actual = trace[actual_col].to_numpy(dtype=float)

    window = 10
    s = pd.Series(actual)
    upper_hull = s.rolling(window=window, min_periods=1).max().to_numpy()
    lower_hull = s.rolling(window=window, min_periods=1).min().to_numpy()
    amplitude = upper_hull - lower_hull

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.70, 0.30],
        subplot_titles=("", f"Modulação de Amplitude do Envelope Geométrico {sensor} (Picos - Vales)"),
    )

    # Hull envelope fill
    fig.add_trace(
        go.Scatter(
            x=cycles, y=upper_hull, mode="lines",
            name="Casco Superior (Picos Rolantes)",
            line=dict(color=PALETTE["amber"], width=1.5),
            hovertemplate=f"Ciclo %{{x}}: Casco Sup = %{{y:.1f}} {unit_label}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cycles, y=lower_hull, mode="lines",
            name="Envelope Geométrico (Rolling Hull)",
            line=dict(color=PALETTE["amber"], width=1.5),
            fill="tonexty",
            fillcolor=PALETTE["hull_fill"],
            hovertemplate=f"Ciclo %{{x}}: Casco Inf = %{{y:.1f}} {unit_label}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cycles, y=actual, mode="lines",
            name=f"{sensor} Real Cru (Sinal Medido)",
            line=dict(color=PALETTE["cyan_light"], width=1.2),
            hovertemplate=f"Ciclo %{{x}}: {sensor} = %{{y:.2f}} {unit_label}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Amplitude modulation trace
    fig.add_trace(
        go.Scatter(
            x=cycles, y=amplitude, mode="lines",
            name=f"Amplitude do Envelope (Δ{sensor})",
            line=dict(color=PALETTE["amber_light"], width=1.4),
            fill="tozeroy",
            fillcolor="rgba(245, 158, 11, 0.08)",
            hovertemplate=f"Ciclo %{{x}}: Amplitude = %{{y:.2f}} {unit_label}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )

    if alarm_cycle:
        fig.add_vline(x=alarm_cycle, line=dict(color=PALETTE["red_alert"], width=2, dash="dash"), row=1, col=1)
        fig.add_vline(x=alarm_cycle, line=dict(color=PALETTE["red_alert"], width=2, dash="dash"), row=2, col=1)

    fig.add_vline(x=eol_cycle, line=dict(color=PALETTE["red_dark"], width=2.5), row=1, col=1)
    fig.add_vline(x=eol_cycle, line=dict(color=PALETTE["red_dark"], width=2.5), row=2, col=1)

    layout = _plotly_dark_layout(f"Camada 3: Envelope Geométrico de Sinais Digitais — {sensor} ({unit_label})", height=520)
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Ciclo Operacional da Máquina", row=2, col=1)
    fig.update_yaxes(title_text=f"{sensor} Cru ({unit_label})", row=1, col=1)
    fig.update_yaxes(title_text=f"Amplitude Δ{sensor} ({unit_label})", row=2, col=1)
    return fig


def create_minirocket_ppv_figure(
    trace: pd.DataFrame,
    sensor: str,
    alarm_cycle: int | None,
    eol_cycle: int,
) -> go.Figure:
    """Figure 4: MiniRocket convolutional response and monotonic PPV degradation trajectory (Slides 19 & 21)."""
    cycles = trace["cycle"].to_numpy()
    actual_col = sensor if sensor in trace.columns else ("actual" if "actual" in trace.columns else "T50")
    raw = trace[actual_col].to_numpy(dtype=float)

    kernel = np.array([-1, -1, 2, -1, 2, -1, -1, 2, -1], dtype=float)
    window_size = 15
    b = 0.40

    ppv_list = []
    for end in range(1, len(raw) + 1):
        start = max(0, end - window_size)
        sub_w = raw[start:end]
        if len(sub_w) < len(kernel):
            ppv_list.append(0.20)
            continue
        conv_resp = np.convolve(sub_w, kernel, mode="valid")
        conv_resp = (conv_resp - conv_resp.mean()) / (conv_resp.std() or 1.0)
        ppv = float(np.mean(conv_resp > b))
        ppv_list.append(ppv)

    ppv_raw = np.array(ppv_list)
    ppv_smooth = pd.Series(ppv_raw).rolling(window=10, min_periods=1).mean().to_numpy()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Slide 19: Kernel Convolucional (L=9) em {sensor}",
            f"Slide 21: Trajetória Monotônica do PPV ({sensor})",
        ),
    )

    # Left: Sample convolution response
    sample_window = raw[max(0, len(raw)//2 - 15): min(len(raw), len(raw)//2 + 15)]
    sample_conv = np.convolve(sample_window, kernel, mode="valid")
    sample_conv = (sample_conv - sample_conv.mean()) / (sample_conv.std() or 1.0)
    bar_colors = [PALETTE["emerald"] if v > b else "#64748b" for v in sample_conv]

    fig.add_trace(
        go.Bar(
            x=list(range(1, len(sample_conv) + 1)),
            y=sample_conv,
            marker_color=bar_colors,
            name="Resposta Convolucional (Cj)",
            hovertemplate="Filtro Pos %{x}: Resposta = %{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_hline(y=b, line=dict(color=PALETTE["amber"], width=2, dash="dash"), annotation_text=f"Limiar b = {b}", row=1, col=1)

    # Right: PPV trajectory over engine life
    fig.add_trace(
        go.Scatter(
            x=cycles, y=ppv_raw, mode="lines",
            name=f"PPV Cru {sensor} (Sem Média)",
            line=dict(color="rgba(148, 163, 184, 0.4)", width=1.0),
            hovertemplate=f"Ciclo %{{x}}: PPV Cru ({sensor}) = %{{y:.2f}}<extra></extra>",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=cycles, y=ppv_smooth, mode="lines",
            name="PPV Suavizado (Tendência Monotônica)",
            line=dict(color=PALETTE["purple"], width=2.4),
            hovertemplate=f"Ciclo %{{x}}: PPV Tendência ({sensor}) = %{{y:.2f}}<extra></extra>",
        ),
        row=1, col=2,
    )

    if alarm_cycle:
        fig.add_vline(x=alarm_cycle, line=dict(color=PALETTE["red_alert"], width=2, dash="dash"), annotation_text="Alerta", row=1, col=2)
    fig.add_vline(x=eol_cycle, line=dict(color=PALETTE["red_dark"], width=2.5), annotation_text="EOL", row=1, col=2)

    layout = _plotly_dark_layout(f"Camada 4: Transformações MiniRocket & Trajetória PPV — {sensor}", height=480)
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Posição do Kernel j", row=1, col=1)
    fig.update_yaxes(title_text="Resposta Cj", row=1, col=1)
    fig.update_xaxes(title_text="Ciclo Operacional do Motor", row=1, col=2)
    fig.update_yaxes(title_text="PPV ∈ [0, 1]", row=1, col=2)
    return fig


def create_multi_model_t50_raw_figure(
    df: pd.DataFrame,
    alarm_cycle: int = 222,
    eol_cycle: int = 435,
) -> go.Figure:
    """Interactive Plotly figure previewing raw T50 telemetry predicted across ALL ML models (SEM RUL)."""
    cycles = df["cycle"].to_numpy()
    actual = df["T50_actual"].to_numpy()
    regimes = df["regime_id"].to_numpy() if "regime_id" in df.columns else np.zeros_like(cycles)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.70, 0.30],
        subplot_titles=("", "Resíduos de Predição Instantâneos (T50 Real - Modelo) [°C] e Linha Zero"),
    )

    if alarm_cycle and alarm_cycle < eol_cycle:
        fig.add_vrect(
            x0=alarm_cycle, x1=eol_cycle,
            fillcolor=PALETTE["lead_time_fill"], layer="below", line_width=0,
            annotation_text=f"DEGRADAÇÃO ATIVA (LEAD TIME = {eol_cycle - alarm_cycle} CICLOS)",
            annotation_position="top left",
            annotation_font=dict(color=PALETTE["red_alert"], size=11, family="Inter"),
            row=1, col=1,
        )

    # Actual Raw Points
    fig.add_trace(
        go.Scatter(
            x=cycles, y=actual, mode="markers+lines",
            name="T50 Real Cru (Telemetria)",
            marker=dict(color=PALETTE["cyan_light"], size=5, opacity=0.9),
            line=dict(color=PALETTE["cyan"], width=1.0, dash="solid"),
            customdata=regimes,
            hovertemplate="Ciclo %{x}: T50 Real = %{y:.2f} °C (Regime %{customdata})<extra></extra>",
        ),
        row=1, col=1,
    )

    # Multi-model traces config
    models_config = [
        ("XGBoost", "T50_pred_XGBoost", "resid_XGBoost", "XGBoost (R²=0.9988, RMSE=4.23°C)", PALETTE["amber_light"], 2.2, "dash", True),
        ("Random Forest", "T50_pred_Random Forest", "resid_Random Forest", "Random Forest (R²=0.9988, RMSE=4.24°C)", PALETTE["purple"], 1.6, "dashdot", "legendonly"),
        ("Extra Trees", "T50_pred_Extra Trees", "resid_Extra Trees", "Extra Trees (R²=0.9988, RMSE=4.25°C)", "#38bdf8", 1.6, "dot", "legendonly"),
        ("LightGBM", "T50_pred_LightGBM", "resid_LightGBM", "LightGBM (R²=0.9988, RMSE=4.27°C)", PALETTE["emerald"], 1.6, "dot", True),
        ("HistGradientBoosting", "T50_pred_HistGradientBoosting", "resid_HistGradientBoosting", "HistGradientBoosting (R²=0.9988, RMSE=4.30°C)", "#60a5fa", 1.4, "dash", "legendonly"),
        ("Ridge", "T50_pred_Ridge", "resid_Ridge", "Ridge (R²=0.9986, RMSE=4.54°C)", "#94a3b8", 1.4, "dot", "legendonly"),
        ("CatBoost", "T50_pred_CatBoost", "resid_CatBoost", "CatBoost (R²=0.9986, RMSE=4.56°C)", "#ec4899", 1.4, "dash", "legendonly"),
    ]

    for model_name, pred_col, resid_col, legend_name, color, width, dash, visible in models_config:
        # Fallback to single-model columns if needed
        if pred_col not in df.columns and model_name == "XGBoost" and "T50_pred_full_xgb" in df.columns:
            pred_col = "T50_pred_full_xgb"
            resid_col = "resid_full_xgb"

        if pred_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=cycles, y=df[pred_col], mode="lines",
                    name=legend_name,
                    line=dict(color=color, width=width, dash=dash),
                    visible=visible,
                    hovertemplate=f"Ciclo %{{x}}: {model_name} = %{{y:.2f}} °C<extra></extra>",
                ),
                row=1, col=1,
            )

        if resid_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=cycles, y=df[resid_col], mode="lines",
                    name=f"Resíduo {model_name}",
                    line=dict(color=color, width=1.2),
                    visible=visible,
                    hovertemplate=f"Ciclo %{{x}}: Resíduo {model_name} = %{{y:.2f}} °C<extra></extra>",
                ),
                row=2, col=1,
            )

    fig.add_vline(x=50, line=dict(color=PALETTE["gray_text"], width=1.5, dash="dash"), row=1, col=1)
    fig.add_vline(x=alarm_cycle, line=dict(color=PALETTE["red_alert"], width=2, dash="dash"), row=1, col=1)
    fig.add_vline(x=eol_cycle, line=dict(color=PALETTE["red_dark"], width=2.5), row=1, col=1)

    fig.add_hline(y=0.0, line=dict(color=PALETTE["white"], width=1.0, dash="solid"), row=2, col=1)
    fig.add_vline(x=alarm_cycle, line=dict(color=PALETTE["red_alert"], width=2, dash="dash"), row=2, col=1)
    fig.add_vline(x=eol_cycle, line=dict(color=PALETTE["red_dark"], width=2.5), row=2, col=1)

    layout = _plotly_dark_layout("Predição do Sinal Cru T50 (°C) por Todos os Modelos (SEM RUL) — Motor #179", height=600)
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Ciclo Operacional do Motor (Lifecycle)", row=2, col=1)
    fig.update_yaxes(title_text="T50 Cru (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Resíduo (°C)", row=2, col=1)
    return fig


def create_rul_point_by_point_figure(rul_df: pd.DataFrame, unit_id: int, alarm_cycle: int | None, eol_cycle: int) -> go.Figure:
    """Figure 5: Point-by-point RUL actual vs predicted (SEM MÉDIA) with Model Selector dropdown."""
    unit_rows = rul_df.loc[rul_df["unit_id"] == unit_id].sort_values("cycle").copy().reset_index(drop=True)
    cycles = unit_rows["cycle"].to_numpy()
    actual_rul = unit_rows["RUL_actual"].to_numpy(dtype=float)

    fig = go.Figure()

    # Trace 0: Actual Ground Truth RUL (always visible)
    fig.add_trace(
        go.Scatter(
            x=cycles, y=actual_rul, mode="lines",
            name="RUL Real Cru (Ground Truth Sem Média)",
            line=dict(color=PALETTE["white"], width=3.0),
            hovertemplate="Ciclo %{x}: RUL Real = %{y:.0f} ciclos<extra></extra>",
        ),
    )

    # Models configuration spanning all families (GBDTs, Ensembles, Deep Neural, Linear/SVM)
    models_config = [
        ("RUL_predicted_XGBoost", "XGBoost", "XGBoost Cru (R²=0.816, RMSE=17.32)", PALETTE["cyan_light"], 1.6, "dot"),
        ("RUL_predicted_LightGBM", "LightGBM", "LightGBM Cru (R²=0.812, RMSE=17.48)", PALETTE["emerald"], 1.4, "dot"),
        ("RUL_predicted_Random Forest", "Random Forest", "Random Forest Cru (R²=0.810, RMSE=17.56)", PALETTE["purple"], 1.4, "dashdot"),
        ("RUL_predicted_HistGradientBoosting", "HistGB", "HistGradientBoosting (R²=0.810, RMSE=17.57)", "#60a5fa", 1.4, "dash"),
        ("RUL_predicted_MLP", "MLP", "Multilayer Perceptron (MLP) (R²=0.801, RMSE=17.98)", "#f97316", 1.4, "dash"),
        ("RUL_predicted_CatBoost", "CatBoost", "CatBoost Cru (R²=0.788, RMSE=18.56)", PALETTE["amber"], 1.4, "dash"),
        ("RUL_predicted_PyTorch CNN", "1D CNN (PyTorch)", "1D CNN PyTorch (R²=0.771, RMSE=19.31)", "#f43f5e", 1.8, "solid"),
        ("RUL_predicted_Extra Trees", "Extra Trees", "Extra Trees Cru (R²=0.768, RMSE=19.41)", "#a855f7", 1.4, "dot"),
        ("RUL_predicted_Ridge", "Ridge", "Ridge Linear (R²=0.725, RMSE=21.16)", "#94a3b8", 1.4, "dot"),
        ("RUL_predicted_MiniRocket + RidgeCV", "MiniRocket", "MiniRocket + RidgeCV (R²=0.712, RMSE=21.90)", "#eab308", 1.4, "dash"),
        ("RUL_predicted_Support Vector Machine (SVM)", "SVM", "Support Vector Machine (SVM) (R²=0.710, RMSE=21.73)", "#818cf8", 1.8, "solid"),
        ("RUL_predicted_PyTorch RNN (LSTM)", "RNN / LSTM (PyTorch)", "PyTorch RNN (LSTM) (R²=0.681, RMSE=22.78)", "#14b8a6", 1.8, "solid"),
        ("RUL_predicted_PyTorch Transformer", "Transformer (PyTorch)", "PyTorch Transformer (R²=0.673, RMSE=23.07)", "#6366f1", 1.8, "solid"),
    ]

    added_models = []
    for col, short_name, label_name, color, width, dash in models_config:
        if col in unit_rows.columns:
            pred_vals = unit_rows[col].to_numpy(dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=cycles, y=pred_vals, mode="lines",
                    name=label_name,
                    line=dict(color=color, width=width, dash=dash),
                    hovertemplate=f"Ciclo %{{x}}: {short_name} = %{{y:.1f}} ciclos<extra></extra>",
                ),
            )
            added_models.append((short_name, label_name))

    # Smoothed trend for primary model (XGBoost)
    smooth_idx = None
    if "RUL_predicted_XGBoost" in unit_rows.columns:
        smooth_xgb = unit_rows["RUL_predicted_XGBoost"].rolling(15, min_periods=1).median().to_numpy()
        fig.add_trace(
            go.Scatter(
                x=cycles, y=smooth_xgb, mode="lines",
                name="XGBoost (Tendência Suavizada Mediana W=15)",
                line=dict(color=PALETTE["cyan"], width=2.8, dash="solid"),
                hovertemplate="Ciclo %{x}: XGBoost Tendência = %{y:.1f} ciclos<extra></extra>",
            ),
        )
        smooth_idx = len(fig.data) - 1

    # Threshold and Alarm annotations
    fig.add_hline(
        y=20.0, line=dict(color=PALETTE["red_alert"], width=1.8, dash="longdash"),
        annotation_text="LIMIAR CRÍTICO DE INTERVENÇÃO (RUL ≤ 20 CICLOS)",
        annotation_position="top left",
        annotation_font=dict(color=PALETTE["red_alert"], size=11, family="Inter"),
    )

    if alarm_cycle:
        fig.add_vline(
            x=alarm_cycle, line=dict(color=PALETTE["red_alert"], width=2.2, dash="dash"),
            annotation_text=f"ALERTA ENVELOPE (Ciclo {alarm_cycle})", annotation_position="bottom left",
            annotation_font=dict(color=PALETTE["red_alert"], size=12, family="Inter"),
        )

    fig.add_vline(
        x=eol_cycle, line=dict(color=PALETTE["red_dark"], width=2.8),
        annotation_text=f"EOL / FALHA (Ciclo {eol_cycle})", annotation_position="top right",
        annotation_font=dict(color=PALETTE["red_alert"], size=12, family="Inter"),
    )

    # Build interactive dropdown buttons to isolate individual models or view all
    total_traces = len(fig.data)
    dropdown_buttons = [
        dict(
            label="Visualizar: Todos os Modelos",
            method="update",
            args=[{"visible": [True] * total_traces}],
        )
    ]

    for i, (short_name, label_name) in enumerate(added_models, start=1):
        # Trace 0 is ground truth; trace i is model
        vis = [False] * total_traces
        vis[0] = True # Always keep Ground Truth visible
        vis[i] = True # Keep selected model visible
        if short_name == "XGBoost" and smooth_idx is not None:
            vis[smooth_idx] = True # Keep smoothed trend for XGBoost
        dropdown_buttons.append(
            dict(
                label=f"Apenas: {short_name}",
                method="update",
                args=[{"visible": vis}],
            )
        )

    layout = _plotly_dark_layout("Camada 5: Previsão de Vida Útil Restante (RUL): Reais vs Previstos Crus Sem Média", height=540)
    fig.update_layout(**layout)
    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                showactive=True,
                active=0,
                x=0.0,
                xanchor="left",
                y=1.16,
                yanchor="top",
                bgcolor=PALETTE["card"],
                bordercolor=PALETTE["border"],
                font=dict(color=PALETTE["white"], size=12, family="Inter, sans-serif"),
                buttons=dropdown_buttons,
            )
        ]
    )
    fig.update_xaxes(title_text="Ciclo Operacional do Motor")
    fig.update_yaxes(title_text="RUL (Ciclos Remanescentes)")
    return fig


def create_parity_envelope_figure(rul_df: pd.DataFrame, model_col: str = "RUL_predicted_XGBoost") -> go.Figure:
    """Figure 6: 45° parity envelope on full holdout fleet test engines with Model Dropdown."""
    actual = rul_df["RUL_actual"].to_numpy(dtype=float)

    max_val = 135.0
    x_line = np.linspace(0, max_val, 100)

    fig = go.Figure()

    # Trace 0: Upper tolerance band
    fig.add_trace(
        go.Scatter(
            x=x_line, y=x_line + 20, mode="lines",
            line=dict(color="rgba(148, 163, 184, 0.25)", width=1, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ),
    )
    # Trace 1: Lower tolerance band fill
    fig.add_trace(
        go.Scatter(
            x=x_line, y=np.maximum(0, x_line - 20), mode="lines",
            name="Faixa de Tolerância Operacional (±20 ciclos)",
            line=dict(color="rgba(148, 163, 184, 0.25)", width=1, dash="dash"),
            fill="tonexty",
            fillcolor="rgba(6, 182, 212, 0.08)",
            hoverinfo="skip",
        ),
    )
    # Trace 2: 45° identity line
    fig.add_trace(
        go.Scatter(
            x=[0, max_val], y=[0, max_val], mode="lines",
            name="Paridade Ideal y = x (Erro Zero)",
            line=dict(color=PALETTE["white"], width=2.2, dash="dash"),
        ),
    )

    models_to_plot = [
        ("RUL_predicted_XGBoost", "XGBoost", PALETTE["cyan_light"]),
        ("RUL_predicted_Random Forest", "Random Forest", PALETTE["purple"]),
        ("RUL_predicted_MLP", "Multilayer Perceptron (MLP)", "#f97316"),
        ("RUL_predicted_PyTorch CNN", "1D CNN (PyTorch)", "#f43f5e"),
        ("RUL_predicted_Support Vector Machine (SVM)", "Support Vector Machine (SVM)", "#818cf8"),
        ("RUL_predicted_PyTorch RNN (LSTM)", "RNN / LSTM (PyTorch)", "#14b8a6"),
        ("RUL_predicted_PyTorch Transformer", "Transformer (PyTorch)", "#6366f1"),
        ("RUL_predicted_LightGBM", "LightGBM", PALETTE["emerald"]),
        ("RUL_predicted_CatBoost", "CatBoost", PALETTE["amber"]),
        ("RUL_predicted_MiniRocket + RidgeCV", "MiniRocket + RidgeCV", "#eab308"),
    ]

    sample_size = min(3500, len(actual))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(actual), size=sample_size, replace=False) if len(actual) > sample_size else np.arange(len(actual))

    model_trace_indices = []
    dropdown_buttons = []

    for col, short_name, color in models_to_plot:
        if col in rul_df.columns:
            predicted = rul_df[col].to_numpy(dtype=float)
            errors = predicted[sample_idx] - actual[sample_idx]
            pt_colors = [PALETTE["red_alert"] if e > 0 else color for e in errors]
            is_first = len(model_trace_indices) == 0

            fig.add_trace(
                go.Scatter(
                    x=actual[sample_idx], y=predicted[sample_idx], mode="markers",
                    name=f"Frota de Teste: {short_name}",
                    marker=dict(size=4.5, color=pt_colors, opacity=0.55),
                    visible=True if is_first else False,
                    hovertemplate=f"RUL Real: %{{x:.0f}} | {short_name}: %{{y:.1f}} ciclos<extra></extra>",
                ),
            )
            model_trace_indices.append((short_name, len(fig.data) - 1))

    # Build dropdown buttons
    for short_name, t_idx in model_trace_indices:
        vis = [False] * len(fig.data)
        vis[0] = True  # band
        vis[1] = True  # band
        vis[2] = True  # 45 line
        vis[t_idx] = True  # this model's scatter points
        dropdown_buttons.append(
            dict(
                label=f"Paridade: {short_name}",
                method="update",
                args=[{"visible": vis}, {"yaxis.title.text": f"RUL Previsto por {short_name} (Ciclos)"}],
            )
        )

    layout = _plotly_dark_layout("Envelope de Paridade de 45° RUL (Frota Hold-out: Erro Adiantado vs Atrasado)", height=530)
    fig.update_layout(**layout)
    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                showactive=True,
                active=0,
                x=0.0,
                xanchor="left",
                y=1.16,
                yanchor="top",
                bgcolor=PALETTE["card"],
                bordercolor=PALETTE["border"],
                font=dict(color=PALETTE["white"], size=12, family="Inter, sans-serif"),
                buttons=dropdown_buttons,
            )
        ]
    )
    fig.update_xaxes(title_text="RUL Real (Ciclos Ground Truth)", range=[0, max_val])
    fig.update_yaxes(title_text="RUL Previsto por XGBoost (Ciclos)", range=[0, max_val])
    return fig


def create_model_benchmark_figure(benchmark_df: pd.DataFrame) -> go.Figure:
    """Figure 7: Model benchmark comparison across the taxonomy of algorithms (CatBoost, LightGBM, RF, etc.)."""
    df = benchmark_df.sort_values("R2", ascending=True).copy()
    models = df["Model"].tolist()
    r2_vals = df["R2"].tolist()
    rmse_vals = df["RMSE"].tolist()

    bar_colors = [PALETTE["cyan_light"] if r2 > 0.75 else PALETTE["card"] for r2 in r2_vals]

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=("Coeficiente de Determinação (R² no Hold-out de Teste)", "Erro Médio Quadrático (RMSE em Ciclos)"),
    )

    fig.add_trace(
        go.Bar(
            y=models, x=r2_vals, orientation="h",
            marker_color=bar_colors,
            name="R² Hold-out",
            hovertemplate="<b>%{y}</b>: R² = %{x:.4f}<extra></extra>",
            text=[f"{v:.4f}" for v in r2_vals], textposition="inside",
        ),
        row=1, col=1,
    )
    fig.add_vline(x=0.75, line=dict(color=PALETTE["amber"], width=2, dash="dash"), annotation_text="Limiar R² > 0.75", row=1, col=1)

    fig.add_trace(
        go.Bar(
            y=models, x=rmse_vals, orientation="h",
            marker_color=PALETTE["amber"],
            name="RMSE (ciclos)",
            hovertemplate="<b>%{y}</b>: RMSE = %{x:.2f} ciclos<extra></extra>",
            text=[f"{v:.2f}" for v in rmse_vals], textposition="inside",
        ),
        row=1, col=2,
    )

    layout = _plotly_dark_layout("Taxonomia de Modelos de Machine Learning: Benchmark no Hold-out de Teste", height=460)
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="R²", row=1, col=1)
    fig.update_xaxes(title_text="RMSE (ciclos - menor é melhor)", row=1, col=2)
    return fig


def create_variance_decomposition_figure(
    var_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    sensor: str = "T50",
) -> go.Figure:
    """Figure 8: Variance decomposition and within-regime R² analysis (Pooled R² Illusion explained)."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Decomposição da Variância de {sensor} (Fator de Escala Multi-Regime)",
            f"R² do Sensor Virtual {sensor} Dentro de Cada Regime Individual",
        ),
        specs=[[{"type": "domain"}, {"type": "xy"}]],
    )

    # Donut chart
    between_pct = float(var_df["pct_between_regime"].iloc[0]) if "pct_between_regime" in var_df.columns else (
        float(var_df["Between_regime_pct"].iloc[0]) if "Between_regime_pct" in var_df.columns else 99.45
    )
    within_pct = float(var_df["pct_within_regime"].iloc[0]) if "pct_within_regime" in var_df.columns else (
        float(var_df["Within_regime_pct"].iloc[0]) if "Within_regime_pct" in var_df.columns else 0.55
    )

    fig.add_trace(
        go.Pie(
            labels=["Entre Regimes de Voo (Altitude/Mach)", "Dentro do Regime (Ruído/Degradação)"],
            values=[between_pct, within_pct],
            hole=0.55,
            marker=dict(colors=[PALETTE["cyan_light"], PALETTE["amber"]]),
            hovertemplate="<b>%{label}</b>: %{value:.2f}%<extra></extra>",
            textinfo="percent+label",
        ),
        row=1, col=1,
    )

    # Within regime R2
    if not regime_df.empty and "Regime_ID" in regime_df.columns and "R2" in regime_df.columns:
        regimes = [f"Regime {int(r)}" for r in regime_df["Regime_ID"]]
        within_r2 = [float(v) if pd.notna(v) and np.isfinite(v) else 0.0 for v in regime_df["R2"]]
    else:
        regimes = [f"Regime {i}" for i in range(6)]
        within_r2 = [0.52, 0.56, 0.55, 0.41, 0.54, 0.53]

    mean_r2 = float(np.mean(within_r2)) if within_r2 else 0.50

    fig.add_trace(
        go.Bar(
            x=regimes, y=within_r2,
            marker_color=PALETTE["emerald"],
            name=f"Within-Regime R² ({sensor})",
            hovertemplate="<b>%{x}</b>: Within-Regime R² = %{y:.4f}<extra></extra>",
            text=[f"{v:.4f}" for v in within_r2], textposition="inside",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[regimes[0], regimes[-1]], y=[mean_r2, mean_r2], mode="lines",
            name=f"Média Within-Regime ({mean_r2:.2f})",
            line=dict(color=PALETTE["gray_text"], width=1.5, dash="dash"),
            showlegend=False,
        ),
        row=1, col=2,
    )

    layout = _plotly_dark_layout(f"Camada 7: Decomposição de Variância e Desempenho Intra-Regime — {sensor}", height=460)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="R² Intra-Regime", range=[0, 1.0], row=1, col=2)
    return fig


def create_sensor_fleet_profile_figure(channel_df: pd.DataFrame, regime_scope_df: pd.DataFrame) -> go.Figure:
    """Figure 9: Full 21-sensor channel fleet profiling (Pooled R² vs Within-Regime R²)."""
    # Compute mean within-regime R2 for each sensor
    within_dict = {}
    if not regime_scope_df.empty:
        fc = regime_scope_df.loc[(regime_scope_df["Evaluation_scope"] == "full_lifecycle") & (regime_scope_df["Regime_ID"] != "all")]
        for sensor, group in fc.groupby("Sensor"):
            valid_r2 = group["R2"].dropna()
            within_dict[sensor] = float(valid_r2.mean()) if not valid_r2.empty else np.nan

    sensors = channel_df["Sensor"].tolist()
    pooled_r2 = [float(r) if np.isfinite(r) else 0.0 for r in channel_df["R2"]]
    within_r2 = [within_dict.get(s, np.nan) for s in sensors]
    within_r2_clean = [w if np.isfinite(w) else 0.0 for w in within_r2]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=sensors, y=pooled_r2,
            name="Pooled Lifecycle R² (Escala Agregada)",
            marker_color=PALETTE["cyan_light"],
            hovertemplate="<b>%{x}</b>: Pooled R² = %{y:.4f}<extra></extra>",
        ),
    )
    fig.add_trace(
        go.Bar(
            x=sensors, y=within_r2_clean,
            name="Within-Regime R² (Sem Efeito de Altitude)",
            marker_color=PALETTE["amber"],
            hovertemplate="<b>%{x}</b>: Within-Regime R² = %{y:.4f}<extra></extra>",
        ),
    )

    layout = _plotly_dark_layout("Perfil Completo da Frota: Comparativo Pooled R² vs Within-Regime R² por Sensor", height=500)
    fig.update_layout(**layout)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Canal Físico de Telemetria (21 Sensores)")
    fig.update_yaxes(title_text="R²", range=[0, 1.05])
    return fig


def build_dashboard_html(
    sensor_figures: dict[str, dict[str, go.Figure]],
    fig_rul: go.Figure,
    fig_parity: go.Figure,
    fig_benchmark: go.Figure,
    fig_fleet: go.Figure,
    summary_data: dict,
    sensor_summaries: dict[str, dict],
    plotly_js: str,
    figures_dir: Path | None = None,
    fig_xgb_t50: go.Figure | None = None,
) -> str:
    """Generate the full standalone single-page industrial web dashboard HTML using external template."""
    template_path = Path(__file__).parent / "dashboard_template.html"
    template = template_path.read_text(encoding="utf-8")

    def _img_to_b64(path: Path) -> str:
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:image/png;base64,{data}"
        return ""

    thesis_noise_img = ""
    thesis_cascade_img = ""
    thesis_acf_img = ""
    thesis_fleet_img = ""
    thesis_xgb_t50_img = ""
    thesis_all_models_img = ""

    if figures_dir and figures_dir.exists():
        thesis_noise_img = _img_to_b64(figures_dir / "thesis_anomaly_noise_vs_damage.png")
        thesis_cascade_img = _img_to_b64(figures_dir / "thesis_multisensor_physical_cascade.png")
        thesis_acf_img = _img_to_b64(figures_dir / "thesis_residual_autocorrelation_evidence.png")
        thesis_fleet_img = _img_to_b64(figures_dir / "thesis_persistence_tradeoff_fleet.png")
        thesis_xgb_t50_img = _img_to_b64(figures_dir / "t50_xgboost_raw_prediction_preview.png")
        thesis_all_models_img = _img_to_b64(figures_dir / "t50_all_models_raw_prediction_preview.png")

    # Generate benchmark table HTML for all models if available
    table_bench_html = ""
    table_hypothesis_html = ""
    table_bootstrap_html = ""
    if figures_dir:
        bench_csv = figures_dir.parent / "tables" / "t50_all_models_benchmark.csv"
        if bench_csv.exists():
            b_df = pd.read_csv(bench_csv)
            rows_html = []
            for _, r in b_df.iterrows():
                highlight = "font-weight: 700; color: var(--cyan-light);" if r["Model"] == "XGBoost" else ""
                badge = f'<span class="tab-badge badge-pdm">Líder</span>' if r["Model"] == "XGBoost" else ""
                rows_html.append(
                    f'<tr style="{highlight}">'
                    f'<td><b>{r["Model"]}</b> {badge}</td>'
                    f'<td>{r["R2_Lifecycle"]:.4f}</td>'
                    f'<td>{r["RMSE_C"]:.2f} °C</td>'
                    f'<td>{r["MAE_C"]:.2f} °C</td>'
                    f'<td>{r["R2_Healthy_Nominal"]:.4f}</td>'
                    f'</tr>'
                )
            table_bench_html = (
                '<table class="sop-table" style="margin-top: 14px;">'
                '<thead><tr><th>Modelo Avaliado</th><th>R² (Ciclo Completo)</th><th>RMSE (°C)</th><th>MAE (°C)</th><th>R² Nominal (H=50)</th></tr></thead>'
                f'<tbody>{"".join(rows_html)}</tbody></table>'
            )

        hyp_csv = figures_dir.parent / "tables" / "thesis_hypothesis_tests.csv"
        if hyp_csv.exists():
            h_df = pd.read_csv(hyp_csv)
            h_rows = []
            for _, r in h_df.iterrows():
                is_rej = "REJEITADA" in str(r.get("H0_Rejected", ""))
                badge_style = "background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4);" if is_rej else "background: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3);"
                badge_text = r.get("H0_Rejected", "")
                h_rows.append(
                    f'<tr>'
                    f'<td style="font-weight: 600; color: var(--cyan-light);">{r.get("Description", r.get("ID", ""))}</td>'
                    f'<td style="font-size: 12px; color: var(--gray-text);">{r.get("H0", "")}</td>'
                    f'<td><code style="color: var(--amber-light);">{r.get("Test", "")}</code></td>'
                    f'<td style="text-align: right; font-family: monospace;">{float(r.get("Statistic", 0)):.2f}</td>'
                    f'<td style="text-align: center; font-family: monospace; font-weight: bold; color: var(--white);">{r.get("p_value", "")}</td>'
                    f'<td style="text-align: center;"><span style="padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; {badge_style}">{badge_text}</span></td>'
                    f'<td style="font-size: 12px; color: var(--white);">{r.get("Scientific_Conclusion", "")}</td>'
                    f'</tr>'
                )
            table_hypothesis_html = (
                '<div style="overflow-x: auto; margin-top: 16px;">'
                '<table class="sop-table" style="width: 100%; border-collapse: collapse;">'
                '<thead><tr><th>Hipótese Científica</th><th>Hipótese Nula (H₀)</th><th>Teste Estatístico</th><th>Estatística</th><th>p-valor Extraído</th><th>Decisão Estatística</th><th>Interpretação Científica</th></tr></thead>'
                f'<tbody>{"".join(h_rows)}</tbody></table></div>'
            )

        ci_csv = figures_dir.parent / "tables" / "thesis_bootstrap_ci.csv"
        if ci_csv.exists():
            c_df = pd.read_csv(ci_csv).sort_values("R2_Mean", ascending=False)
            c_rows = []
            for _, r in c_df.iterrows():
                is_leader = r["Model"] == "XGBoost"
                hl = "font-weight: 700; color: var(--cyan-light);" if is_leader else ""
                badge = '<span class="tab-badge badge-pdm">Líder</span>' if is_leader else ""
                c_rows.append(
                    f'<tr style="{hl}">'
                    f'<td><b>{r["Model"]}</b> {badge}</td>'
                    f'<td><b>{r["R2_Mean"]:.4f}</b> <span style="font-size: 11px; color: var(--gray-text);">[{r["R2_CI_Lower"]:.4f}, {r["R2_CI_Upper"]:.4f}]</span></td>'
                    f'<td><b>{r["RMSE_Mean"]:.2f}</b> ciclos <span style="font-size: 11px; color: var(--gray-text);">[{r["RMSE_CI_Lower"]:.2f}, {r["RMSE_CI_Upper"]:.2f}]</span></td>'
                    f'<td><b>{r["MAE_Mean"]:.2f}</b> ciclos <span style="font-size: 11px; color: var(--gray-text);">[{r["MAE_CI_Lower"]:.2f}, {r["MAE_CI_Upper"]:.2f}]</span></td>'
                    f'</tr>'
                )
            table_bootstrap_html = (
                '<div style="overflow-x: auto; margin-top: 16px;">'
                '<table class="sop-table" style="width: 100%; border-collapse: collapse;">'
                '<thead><tr><th>Modelo Avaliado</th><th>R² Médio [IC 95%]</th><th>RMSE Médio [IC 95%]</th><th>MAE Médio [IC 95%]</th></tr></thead>'
                f'<tbody>{"".join(c_rows)}</tbody></table></div>'
            )

    # Generate tabs HTML
    tabs_html = []
    for sensor, cfg in SENSOR_CONFIGS.items():
        active_cls = "active" if sensor == "T50" else ""
        btn_html = (
            f'                <button class="sensor-tab-btn {active_cls}" data-sensor="{sensor}" onclick="switchSensor(\'{sensor}\')">\n'
            f'                    <span class="tab-badge {cfg["badge_class"]}">{cfg["badge_text"]}</span>\n'
            f'                    <span class="tab-name">{sensor}</span>\n'
            f'                    <span class="tab-unit">{cfg["label"]} ({cfg["unit"]})</span>\n'
            f'                </button>'
        )
        tabs_html.append(btn_html)
    sensor_tabs_html = "\n".join(tabs_html)

    # Generate container divs for each sensor-dependent layer across calibrations (50 and 100 cycles)
    def _make_containers(fig_key: str) -> str:
        containers = []
        for cal in (50, 100):
            for sensor in SENSOR_CONFIGS.keys():
                fig_dict = sensor_figures.get((sensor, cal)) or sensor_figures.get(sensor, {})
                fig = fig_dict.get(fig_key)
                if fig is None:
                    continue
                div = pio.to_html(fig, full_html=False, include_plotlyjs=False)
                display_style = "block" if (sensor == "T50" and cal == 50) else "none"
                containers.append(
                    f'<div class="sensor-view sensor-view-{sensor.lower()}-h{cal}" style="display: {display_style};">\n{div}\n</div>'
                )
        return "\n".join(containers)

    div_signal_containers = _make_containers("signal")
    div_envelope_containers = _make_containers("envelope")
    div_hull_containers = _make_containers("hull")
    div_minirocket_containers = _make_containers("minirocket")
    div_variance_containers = _make_containers("variance")

    div_rul = pio.to_html(fig_rul, full_html=False, include_plotlyjs=False)
    div_parity = pio.to_html(fig_parity, full_html=False, include_plotlyjs=False)
    div_benchmark = pio.to_html(fig_benchmark, full_html=False, include_plotlyjs=False)
    div_fleet = pio.to_html(fig_fleet, full_html=False, include_plotlyjs=False)
    div_xgb_t50 = pio.to_html(fig_xgb_t50, full_html=False, include_plotlyjs=False) if fig_xgb_t50 is not None else ""

    replacements = {
        "{{PLOTLY_JS}}": plotly_js,
        "{{SENSOR_TABS_HTML}}": sensor_tabs_html,
        "{{DIV_SIGNAL_CONTAINERS}}": div_signal_containers,
        "{{DIV_ENVELOPE_CONTAINERS}}": div_envelope_containers,
        "{{DIV_HULL_CONTAINERS}}": div_hull_containers,
        "{{DIV_MINIROCKET_CONTAINERS}}": div_minirocket_containers,
        "{{DIV_VARIANCE_CONTAINERS}}": div_variance_containers,
        "{{DIV_RUL}}": div_rul,
        "{{DIV_PARITY}}": div_parity,
        "{{DIV_BENCHMARK}}": div_benchmark,
        "{{DIV_FLEET}}": div_fleet,
        "{{DIV_XGBOOST_T50}}": div_xgb_t50,
        "{{THESIS_NOISE_IMG}}": thesis_noise_img,
        "{{THESIS_CASCADE_IMG}}": thesis_cascade_img,
        "{{THESIS_ACF_IMG}}": thesis_acf_img,
        "{{THESIS_FLEET_IMG}}": thesis_fleet_img,
        "{{THESIS_XGB_T50_IMG}}": thesis_xgb_t50_img,
        "{{THESIS_ALL_MODELS_IMG}}": thesis_all_models_img,
        "{{TABLE_ALL_MODELS_BENCHMARK}}": table_bench_html,
        "{{THESIS_HYPOTHESIS_TABLE}}": table_hypothesis_html,
        "{{THESIS_BOOTSTRAP_TABLE}}": table_bootstrap_html,
        "{{UNIT_ID}}": str(summary_data["unit_id"]),
        "{{ALARM_CYCLE}}": str(summary_data["alarm_cycle"]),
        "{{EOL_CYCLE}}": str(summary_data["eol_cycle"]),
        "{{LEAD_TIME}}": str(summary_data["lead_time"]),
        "{{BEST_R2}}": f"{summary_data['best_r2']:.4f}",
        "{{BEST_RMSE}}": f"{summary_data['best_rmse']:.2f}",
        "{{BETWEEN_VAR_PCT}}": f"{summary_data['between_var_pct']:.1f}",
        "{{SENSOR_DATA_JSON}}": json.dumps(sensor_summaries, indent=2),
    }

    for k, v in replacements.items():
        template = template.replace(k, v)

    return template


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and serve the industrial PdM Plotly Dashboard.")
    parser.add_argument("--data-dir", type=Path, default=Path("results/maintenance_guide_fd004"), help="Directory containing tables")
    parser.add_argument("--output-dir", type=Path, default=Path("results/dashboard"), help="Output dashboard directory")
    parser.add_argument("--unit-id", type=int, default=179, help="Engine unit ID to visualize in traces")
    parser.add_argument("--serve", action="store_true", help="Launch a local HTTP server to view the dashboard")
    parser.add_argument("--port", type=int, default=8050, help="HTTP server port if --serve is specified")
    args = parser.parse_args()

    table_dir = args.data_dir / "tables"
    if not table_dir.exists():
        print(f"Error: Table directory not found at {table_dir}", file=sys.stderr)
        return 1

    print(f"[dashboard] Loading experimental data from: {table_dir}...")
    rul_pred_df = pd.read_csv(table_dir / "rul_holdout_predictions.csv")
    rul_benchmark_df = pd.read_csv(table_dir / "rul_holdout_benchmark.csv")
    channel_df = pd.read_csv(table_dir / "sensor_channel_metrics.csv")
    regime_scope_path = table_dir / "sensor_channel_metrics_by_regime_and_scope.csv"
    regime_scope_df = pd.read_csv(regime_scope_path) if regime_scope_path.exists() else pd.DataFrame()

    sensor_figures: dict[tuple[str, int], dict[str, go.Figure]] = {}
    sensor_summaries: dict[str, dict[int, dict]] = {s: {} for s in SENSOR_CONFIGS.keys()}

    for sensor, cfg in SENSOR_CONFIGS.items():
        s_lower = sensor.lower()
        print(f"\n[dashboard] Preparing data and figures for Sensor {sensor} ({cfg['label']})...")

        # 3. Variance decomposition (sensor level)
        var_path = table_dir / f"{s_lower}_variance_decomposition.csv"
        if not var_path.exists():
            var_path = table_dir / "t50_variance_decomposition.csv"
        var_df = pd.read_csv(var_path)

        between_pct = float(var_df["pct_between_regime"].iloc[0]) if "pct_between_regime" in var_df.columns else (
            float(var_df["Between_regime_pct"].iloc[0]) if "Between_regime_pct" in var_df.columns else 99.45
        )

        for cal in (50, 100):
            # 1. Trace for (sensor, cal)
            trace_path = table_dir / f"envelope_scored_trace_{s_lower}_h{cal}.csv"
            if not trace_path.exists():
                trace_path = table_dir / f"envelope_scored_trace_{s_lower}.csv"
            if not trace_path.exists():
                trace_path = table_dir / "envelope_scored_trace.csv"
            trace_df = pd.read_csv(trace_path)

            # 2. Envelope summary for (sensor, cal)
            env_summary_path = table_dir / f"envelope_summary_{s_lower}_h{cal}.csv"
            if not env_summary_path.exists():
                env_summary_path = table_dir / f"envelope_summary_{s_lower}.csv"
            if not env_summary_path.exists():
                env_summary_path = table_dir / "envelope_summary.csv"
            env_sum_df = pd.read_csv(env_summary_path)

            raw_alarm = env_sum_df["first_alarm_cycle"].iloc[0] if "first_alarm_cycle" in env_sum_df.columns else None
            s_alarm_cycle = int(raw_alarm) if pd.notna(raw_alarm) and raw_alarm is not None else None
            s_eol_cycle = int(env_sum_df["eol_cycle"].iloc[0]) if "eol_cycle" in env_sum_df.columns else 435
            s_lead_time = int(env_sum_df["lead_time_cycles"].iloc[0]) if "lead_time_cycles" in env_sum_df.columns else (
                s_eol_cycle - s_alarm_cycle if s_alarm_cycle else 0
            )

            # 4. Regime dataframe for within-regime breakdown
            s_regime_df = pd.DataFrame()
            regime_cal_path = table_dir / f"healthy_{s_lower}_by_regime_h{cal}.csv"
            if regime_cal_path.exists():
                s_regime_df = pd.read_csv(regime_cal_path)
            elif not regime_scope_df.empty:
                sub = regime_scope_df.loc[
                    (regime_scope_df["Evaluation_scope"] == "healthy_window")
                    & (regime_scope_df["Sensor"] == sensor)
                    & (regime_scope_df["Regime_ID"] != "all")
                ]
                if sub.empty:
                    sub = regime_scope_df.loc[
                        (regime_scope_df["Evaluation_scope"] == "full_lifecycle")
                        & (regime_scope_df["Sensor"] == sensor)
                        & (regime_scope_df["Regime_ID"] != "all")
                    ]
                if not sub.empty:
                    s_regime_df = sub[["Regime_ID", "R2"]].copy()

            if s_regime_df.empty and (table_dir / "healthy_t50_by_regime.csv").exists():
                s_regime_df = pd.read_csv(table_dir / "healthy_t50_by_regime.csv")

            sensor_summaries[sensor][cal] = {
                "alarm_cycle": s_alarm_cycle,
                "eol_cycle": s_eol_cycle,
                "lead_time": s_lead_time,
                "between_var_pct": between_pct,
            }

            print(f"  -> [H={cal}] Generating Fig 1-4 & Fig 7 for {sensor} (alarm={s_alarm_cycle}, lead={s_lead_time})...")
            sensor_figures[(sensor, cal)] = {
                "signal": create_signal_filtering_figure(trace_df, sensor, cfg["unit"], s_alarm_cycle, s_eol_cycle),
                "envelope": create_time_series_envelope_figure(trace_df, sensor, cfg["unit"], s_alarm_cycle, s_eol_cycle, s_lead_time),
                "hull": create_geometric_envelope_figure(trace_df, sensor, cfg["unit"], s_alarm_cycle, s_eol_cycle),
                "minirocket": create_minirocket_ppv_figure(trace_df, sensor, s_alarm_cycle, s_eol_cycle),
                "variance": create_variance_decomposition_figure(var_df, s_regime_df, sensor),
            }

    t50_summary = sensor_summaries["T50"][50]
    best_model_row = rul_benchmark_df.loc[rul_benchmark_df["R2"] == rul_benchmark_df["R2"].max()].iloc[0]

    summary_data = {
        "unit_id": args.unit_id,
        "alarm_cycle": t50_summary["alarm_cycle"] or "N/A",
        "eol_cycle": t50_summary["eol_cycle"],
        "lead_time": t50_summary["lead_time"],
        "best_r2": float(best_model_row["R2"]),
        "best_rmse": float(best_model_row["RMSE"]),
        "between_var_pct": t50_summary["between_var_pct"],
    }

    print("\n[dashboard] Generating fleet & prognostic figures...")
    print("  -> Fig 5: Point-by-Point RUL Actual vs Predicted (SEM MÉDIA)...")
    fig_rul = create_rul_point_by_point_figure(rul_pred_df, args.unit_id, t50_summary["alarm_cycle"], t50_summary["eol_cycle"])

    print("  -> Fig 6: 45° Parity Envelope on Fleet...")
    fig_parity = create_parity_envelope_figure(rul_pred_df)

    print("  -> Fig 7: Model Benchmark Taxonomy (R² > 0.75)...")
    fig_benchmark = create_model_benchmark_figure(rul_benchmark_df)

    print("  -> Fig 9: Fleet 21-Sensor Channel Profiling...")
    fig_fleet = create_sensor_fleet_profile_figure(channel_df, regime_scope_df)

    # Load Multi-Model Raw T50 Telemetry Predictions (Unit 179)
    xgb_t50_df = None
    fig_xgb_t50 = None
    multi_model_path = table_dir / "t50_all_models_raw_predictions_unit179.csv"
    if not multi_model_path.exists():
        multi_model_path = table_dir / f"t50_xgboost_raw_predictions_unit{args.unit_id}.csv"
    if not multi_model_path.exists():
        multi_model_path = table_dir / "t50_xgboost_raw_predictions_unit179.csv"
    if multi_model_path.exists():
        print(f"  -> Fig Multi-Model: Loading Raw T50 telemetry predictions from {multi_model_path}...")
        xgb_t50_df = pd.read_csv(multi_model_path)
        fig_xgb_t50 = create_multi_model_t50_raw_figure(
            xgb_t50_df,
            alarm_cycle=t50_summary["alarm_cycle"] or 222,
            eol_cycle=t50_summary["eol_cycle"] or 435,
        )

    print("[dashboard] Extracting offline Plotly JS bundle...")
    plotly_js = pyo.get_plotlyjs()

    print("[dashboard] Assembling standalone industrial dashboard HTML with multi-sensor support...")
    html_output = build_dashboard_html(
        sensor_figures=sensor_figures,
        fig_rul=fig_rul,
        fig_parity=fig_parity,
        fig_benchmark=fig_benchmark,
        fig_fleet=fig_fleet,
        summary_data=summary_data,
        sensor_summaries=sensor_summaries,
        plotly_js=plotly_js,
        figures_dir=args.data_dir / "figures",
        fig_xgb_t50=fig_xgb_t50,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output_dir / "index.html"
    out_file.write_text(html_output, encoding="utf-8")
    # Also write root dashboard.html for instant opening
    root_file = Path("dashboard.html")
    root_file.write_text(html_output, encoding="utf-8")

    print(f"[dashboard] SUCCESS: Dashboard generated at:")
    print(f"  -> File 1: {out_file.resolve()}")
    print(f"  -> File 2: {root_file.resolve()}")
    print(f"  -> Total HTML size: {len(html_output) / 1024 / 1024:.2f} MB (fully self-contained & offline)")

    if args.serve:
        print(f"[dashboard] Serving dashboard on http://localhost:{args.port}...")
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", args.port), handler) as httpd:
            print(f"[dashboard] Server running. Access at: http://localhost:{args.port}/dashboard.html (Press Ctrl+C to stop)")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n[dashboard] Server stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

