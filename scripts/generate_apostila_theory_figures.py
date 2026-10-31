#!/usr/bin/env python3
"""
Generate dedicated, publication-quality pedagogical diagrams for the PdM Treatise / Textbook.
Outputs saved to results/maintenance_guide_fd004/figures/:
1. theory_pipeline_architecture.png: 7-layer "Do Sinal à Decisão" complete pipeline.
2. theory_turbofan_stations.png: Turbofan cross-section with stations (2, 2.4, 3, 4, 5) and sensor locations.
3. theory_brayton_cycle.png: Thermodynamic Brayton cycle (T-s and P-v) showing healthy vs degraded compressor/turbine.
4. theory_persistence_fsm.png: Shewhart SPC Envelope and Persistence Latch (k>=2) Finite State Machine.
5. theory_sop_flowchart.png: 4-Stage Standard Operating Procedure (SOP) Decision Flowchart.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Styling configuration
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Helvetica", "Arial"]

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
    "emerald": "#10b981",
    "purple": "#8b5cf6",
    "gray_text": "#94a3b8",
    "white": "#f8fafc",
    "blue_light": "#60a5fa",
}

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "results" / "maintenance_guide_fd004" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_pipeline_architecture():
    fig, ax = plt.subplots(figsize=(15, 8.5), facecolor=PALETTE["bg_dark"])
    ax.set_facecolor(PALETTE["bg_dark"])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    ax.text(7.5, 8.0, "ESTEIRA INDUSTRIAL: DO SINAL À DECISÃO (7 CAMADAS)", 
            color=PALETTE["cyan_light"], fontsize=18, fontweight="bold", ha="center")
    ax.text(7.5, 7.6, "Arquitetura Metodológica de Juliano Pádua aplicada ao NASA Turbofan C-MAPSS FD004", 
            color=PALETTE["gray_text"], fontsize=11, ha="center")

    layers = [
        ("Camada 1: Aquisição & Condicionamento", 
         "21 Sensores Físicos + 3 Settings de Voo\nFiltro Causal EWMA (α=0.35) -> Atenuação de 78.8% do Jitter", 
         PALETTE["cyan"], 6.2),
        ("Camada 2: Desacoplamento Físico-Atmosférico", 
         "Clusterização K-Means (6 Regimes de Voo de Altitude e Mach)\nANOVA Unidirecional: 99.45% Inter-Regime vs 0.55% Intra-Regime", 
         PALETTE["blue_light"], 5.0),
        ("Camada 3: Sensoriamento Virtual Multi-Sensor", 
         "Gêmeo Digital de Normalidade Calibrado na Janela Saudável H=50\nCanais T50, P30, Nc, BPR -> Resíduos Puros de Degradação: e(t) = y(t) - ŷ(t)", 
         PALETTE["purple"], 3.8),
        ("Camada 4: Detecção por Envelope (\"Embrulho\") & Geometria", 
         "Corredores Shewhart SPC (μ ± 3σ) + Rolling Convex Hull (Picos e Vales)\nTrava de Persistência Industrial (k ≥ 2) -> ZERO Falsos Alarmes (FAR=0.0%)", 
         PALETTE["amber"], 2.6),
        ("Camada 5: Enriquecimento Convolucional MiniRocket", 
         "84 Kernels Determinísticos de Comprimento 9 com Pesos {-1, 2} (Soma Nula)\nInvariância Rigorosa a Offset DC de Altitude -> Health Index Monotônico", 
         PALETTE["cyan_light"], 1.4),
        ("Camada 6: Modelagem Preditiva & RUL", 
         "GBDTs (XGBoost R²=0.8163, LightGBM, CatBoost) vs Deep Learning\nAlvo Linear por Partes (min(125, EOL - t)) + Suavização por Mediana Móvel (W=15)", 
         PALETTE["emerald"], 0.2),
    ]

    for title, desc, col, y in layers:
        box = patches.FancyBboxPatch((1.0, y), 13.0, 0.95, boxstyle="round,pad=0.12",
                                     fc=PALETTE["surface"], ec=col, lw=2.0)
        ax.add_patch(box)
        
        # Color indicator tab
        tab = patches.FancyBboxPatch((1.0, y), 0.35, 0.95, boxstyle="round,pad=0.0",
                                     fc=col, ec=col, lw=1.0)
        ax.add_patch(tab)

        ax.text(1.6, y + 0.65, title, color=col, fontsize=12, fontweight="bold", va="center")
        ax.text(1.6, y + 0.28, desc, color=PALETTE["white"], fontsize=9.5, va="center")

        if y > 0.5:
            ax.annotate("", xy=(7.5, y), xytext=(7.5, y + 0.05),
                        arrowprops=dict(arrowstyle="->", color=PALETTE["gray_text"], lw=2.0))

    out_path = OUTPUT_DIR / "theory_pipeline_architecture.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=PALETTE["bg_dark"])
    plt.close()
    print(f"Generated {out_path}")


def generate_turbofan_stations():
    fig, ax = plt.subplots(figsize=(15, 8.5), facecolor=PALETTE["bg_dark"])
    ax.set_facecolor(PALETTE["bg_dark"])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    ax.text(7.5, 8.0, "ARQUITETURA DO MOTOR TURBOFAN E POSICIONAMENTO DOS 21 SENSORES", 
            color=PALETTE["cyan_light"], fontsize=17, fontweight="bold", ha="center")
    ax.text(7.5, 7.6, "Estações Termodinâmicas do NASA C-MAPSS (Ciclo Brayton de Alta Razão de Derivação)", 
            color=PALETTE["gray_text"], fontsize=11, ha="center")

    # Draw Nacelle Outer Outline
    nacelle = patches.FancyBboxPatch((1.2, 2.2), 12.6, 4.4, boxstyle="round,pad=0.2",
                                     fc=PALETTE["surface"], ec=PALETTE["border"], lw=2.0)
    ax.add_patch(nacelle)

    # Core engine duct outline
    core = patches.FancyBboxPatch((4.0, 3.2), 8.5, 2.4, boxstyle="round,pad=0.15",
                                 fc=PALETTE["card"], ec=PALETTE["cyan"], lw=2.0)
    ax.add_patch(core)

    # Fan Section
    fan = patches.Rectangle((2.2, 2.5), 1.2, 3.8, fc=PALETTE["cyan"], ec=PALETTE["cyan_light"], lw=1.5, alpha=0.3)
    ax.add_patch(fan)
    ax.text(2.8, 4.4, "FAN\n(N1)", color=PALETTE["white"], fontsize=11, fontweight="bold", ha="center", va="center")

    # LPC Section
    lpc = patches.Rectangle((4.3, 3.4), 1.1, 2.0, fc=PALETTE["blue_light"], ec=PALETTE["cyan"], lw=1.5, alpha=0.3)
    ax.add_patch(lpc)
    ax.text(4.85, 4.4, "LPC", color=PALETTE["white"], fontsize=10, fontweight="bold", ha="center", va="center")

    # HPC Section
    hpc = patches.Rectangle((5.8, 3.5), 1.4, 1.8, fc=PALETTE["amber"], ec=PALETTE["amber_light"], lw=1.5, alpha=0.3)
    ax.add_patch(hpc)
    ax.text(6.5, 4.4, "HPC\n(N2)", color=PALETTE["white"], fontsize=10, fontweight="bold", ha="center", va="center")

    # Combustor Section
    comb = patches.Rectangle((7.6, 3.6), 1.3, 1.6, fc=PALETTE["red_alert"], ec=PALETTE["amber"], lw=1.5, alpha=0.4)
    ax.add_patch(comb)
    ax.text(8.25, 4.4, "QUEIMADOR\n(Combustor)", color=PALETTE["white"], fontsize=9, fontweight="bold", ha="center", va="center")

    # HPT Section
    hpt = patches.Rectangle((9.3, 3.5), 1.1, 1.8, fc=PALETTE["purple"], ec=PALETTE["amber_light"], lw=1.5, alpha=0.3)
    ax.add_patch(hpt)
    ax.text(9.85, 4.4, "HPT", color=PALETTE["white"], fontsize=10, fontweight="bold", ha="center", va="center")

    # LPT Section
    lpt = patches.Rectangle((10.8, 3.4), 1.2, 2.0, fc=PALETTE["emerald"], ec=PALETTE["cyan_light"], lw=1.5, alpha=0.3)
    ax.add_patch(lpt)
    ax.text(11.4, 4.4, "LPT", color=PALETTE["white"], fontsize=10, fontweight="bold", ha="center", va="center")

    # Bypass flow labels
    ax.text(8.0, 6.0, "Fluxo Secundário (Duto de Bypass) ==> Razão BPR (s15)", 
            color=PALETTE["cyan_light"], fontsize=11, fontweight="bold", ha="center")
    ax.annotate("", xy=(13.0, 6.0), xytext=(3.5, 6.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["cyan"], lw=2.0, ls="--"))
    ax.text(8.0, 2.7, "Fluxo Secundário Inferior ==> Pressão P15 (s6)", 
            color=PALETTE["cyan_light"], fontsize=10, ha="center")

    # Sensor callout badges
    callouts = [
        (2.0, 6.9, "Estação 2: Entrada Fan\n• s1: T2 (Temp. Total)\n• s5: P2 (Pressão Total)", PALETTE["cyan"]),
        (4.8, 6.9, "Estação 2.4: Saída LPC\n• s2: T24 (Temp. Total)\n• s8: Nf (Rotação Eixo N1)", PALETTE["blue_light"]),
        (7.0, 1.6, "Estação 3: Descarga HPC\n• s3: T30 (Temp. Descarga)\n• s7: P30 (Pressão Total)\n• s11: Ps30 (Estática)\n• s9: Nc (Rotação Eixo N2)", PALETTE["amber"]),
        (8.5, 6.9, "Câmara Combustão\n• s12: phi (Razão Comb./Ar)\n• s16: farB (Razão Queimador)", PALETTE["red_alert"]),
        (10.2, 1.6, "Sangria & Refrigeração\n• s17: htBleed (Entalpia)\n• s20: W31 (Sangria HPT)\n• s21: W32 (Sangria LPT)", PALETTE["purple"]),
        (12.5, 6.9, "Estação 5: Saída LPT / Bocal\n• s4: T50 (Temp. Exaustão EGT)\n• s10: epr (Razão Pressão P50/P2)", PALETTE["emerald"]),
    ]

    for x, y, text, col in callouts:
        box = patches.FancyBboxPatch((x - 1.0, y - 0.5), 2.2, 1.0, boxstyle="round,pad=0.1",
                                     fc=PALETTE["card"], ec=col, lw=1.5)
        ax.add_patch(box)
        ax.text(x + 0.1, y, text, color=PALETTE["white"], fontsize=8.5, ha="center", va="center")

    out_path = OUTPUT_DIR / "theory_turbofan_stations.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=PALETTE["bg_dark"])
    plt.close()
    print(f"Generated {out_path}")


def generate_brayton_cycle():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5), facecolor=PALETTE["bg_dark"])
    for ax in (ax1, ax2):
        ax.set_facecolor(PALETTE["surface"])
        ax.tick_params(colors=PALETTE["gray_text"])
        for spine in ax.spines.values():
            spine.set_color(PALETTE["border"])

    # Diagram 1: T-s (Temperature vs Entropy)
    ax1.set_title("Diagrama Temperatura-Entropia (T-s): Ciclo Brayton\nDegradação de Compressor (Fouling) e Turbina (Erosão)", 
                  color=PALETTE["white"], fontsize=12, fontweight="bold")
    ax1.set_xlabel("Entropia Específica (s) [kJ / (kg·K)]", color=PALETTE["gray_text"], fontsize=11)
    ax1.set_ylabel("Temperatura Total (T) [K ou °C]", color=PALETTE["gray_text"], fontsize=11)

    s1, T1 = 1.0, 290.0
    s2s, T2s = 1.0, 680.0
    s2_real, T2_real = 1.18, 730.0
    s3, T3 = 2.1, 1550.0
    s4s, T4s = 2.1, 850.0
    s4_real, T4_real = 2.32, 980.0  # Spike in EGT / T50!

    # Nominal healthy cycle
    ax1.plot([s1, s2s], [T1, T2s], color=PALETTE["cyan"], lw=2.5, ls="--", label="Compressão Isentrópica Ideal (ηc=1.0)")
    ax1.plot([s2s, s3], [T2s, T3], color=PALETTE["amber"], lw=2.5, label="Combustão Isobárica (P=cte)")
    ax1.plot([s3, s4s], [T3, T4s], color=PALETTE["emerald"], lw=2.5, ls="--", label="Expansão Isentrópica Ideal (ηt=1.0)")
    ax1.plot([s4s, s1], [T4s, T1], color=PALETTE["gray_text"], lw=1.5, ls=":")

    # Degraded cycle (real)
    ax1.plot([s1, s2_real], [T1, T2_real], color=PALETTE["red_alert"], lw=3.0, label="Compressão Real com Perda de Rendimento (ηc < 1)")
    ax1.plot([s2_real, s3], [T2_real, T3], color=PALETTE["amber_light"], lw=2.0)
    ax1.plot([s3, s4_real], [T3, T4_real], color=PALETTE["purple"], lw=3.0, label="Expansão Real com Desgaste de Turbina (ηt < 1)")

    # Callout highlighting T50 / EGT elevation
    ax1.annotate("Elevação Térmica na Descarga:\nΔT50 = +7.26°C a +15°C no Fim de Vida!",
                 xy=(s4_real, T4_real), xytext=(s4_real - 0.7, T4_real + 160),
                 color=PALETTE["red_alert"], fontsize=10, fontweight="bold",
                 arrowprops=dict(facecolor=PALETTE["red_alert"], edgecolor=PALETTE["red_alert"], arrowstyle="->", lw=2.0))

    ax1.legend(loc="upper left", facecolor=PALETTE["card"], edgecolor=PALETTE["border"], 
               labelcolor=PALETTE["white"], fontsize=9.5)
    ax1.grid(True, color=PALETTE["border"], ls="--", alpha=0.4)

    # Diagram 2: P-v (Pressure vs Volume)
    ax2.set_title("Diagrama Pressão-Volume (P-v)\nPerda de Razão de Pressão (P30) e Redução de Trabalho Útil", 
                  color=PALETTE["white"], fontsize=12, fontweight="bold")
    ax2.set_xlabel("Volume Específico (v) [m³ / kg]", color=PALETTE["gray_text"], fontsize=11)
    ax2.set_ylabel("Pressão Total (P) [kPa ou psia]", color=PALETTE["gray_text"], fontsize=11)

    v = np.linspace(0.1, 1.2, 100)
    # Curves
    p_comp_ideal = 100.0 / (v**1.4)
    p_comp_degraded = 85.0 / (v**1.35)

    ax2.plot(v, np.clip(p_comp_ideal, 100, 3000), color=PALETTE["cyan"], lw=2.5, label="Compressor Saudável (P30 Nominal)")
    ax2.plot(v, np.clip(p_comp_degraded, 90, 2550), color=PALETTE["red_alert"], lw=2.5, ls="--", label="Compressor Degradado (Queda de P30)")

    ax2.annotate("Queda de Pressurização P30:\nPerda de Eficiência Isentrópica",
                 xy=(0.18, 2400), xytext=(0.35, 2600),
                 color=PALETTE["amber_light"], fontsize=10, fontweight="bold",
                 arrowprops=dict(facecolor=PALETTE["amber_light"], edgecolor=PALETTE["amber_light"], arrowstyle="->", lw=2.0))

    ax2.legend(loc="upper right", facecolor=PALETTE["card"], edgecolor=PALETTE["border"], 
               labelcolor=PALETTE["white"], fontsize=9.5)
    ax2.grid(True, color=PALETTE["border"], ls="--", alpha=0.4)

    out_path = OUTPUT_DIR / "theory_brayton_cycle.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=PALETTE["bg_dark"])
    plt.close()
    print(f"Generated {out_path}")


def generate_persistence_fsm():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5), facecolor=PALETTE["bg_dark"])
    for ax in (ax1, ax2):
        ax.set_facecolor(PALETTE["surface"])
        ax.tick_params(colors=PALETTE["gray_text"])
        for spine in ax.spines.values():
            spine.set_color(PALETTE["border"])

    # Left: Waveform of noise vs damage
    ax1.set_title("O Mecanismo da Trava de Persistência (k ≥ 2):\nRuído Espúrio (Ciclo 173) vs Quebra Estrutural (Ciclo 222)", 
                  color=PALETTE["white"], fontsize=12, fontweight="bold")
    ax1.set_xlabel("Ciclo Operacional do Turbofan (Voo)", color=PALETTE["gray_text"], fontsize=11)
    ax1.set_ylabel("Resíduo Térmico e(t) = T50 - T50_hat [°C]", color=PALETTE["gray_text"], fontsize=11)

    t = np.linspace(150, 250, 101)
    # Baseline noise around zero
    np.random.seed(42)
    noise = np.random.normal(0, 1.8, len(t))
    signal = noise.copy()

    # Spike at Cycle 173 (index 23)
    signal[23] = 6.42
    # Persistent degradation starting at Cycle 222 (index 72)
    for i in range(72, len(t)):
        signal[i] += 4.5 + 0.15 * (i - 72)

    ax1.plot(t, signal, color=PALETTE["cyan_light"], lw=1.5, marker="o", markersize=3, label="Resíduo Diário e(t)")
    ax1.axhline(6.28, color=PALETTE["red_alert"], lw=2.0, ls="--", label="Limite Superior Shewhart (+3σ = +6.28°C)")
    ax1.axhline(0.0, color=PALETTE["gray_text"], lw=1.0, ls=":")

    # Highlight Cycle 173
    ax1.plot(173, 6.42, marker="o", markersize=10, color=PALETTE["amber"], label="Ciclo 173: Spike Único (k=1) -> Rejeitado!")
    ax1.annotate("k=1: Ruído Espúrio\nNenhum Alarme Falso!",
                 xy=(173, 6.42), xytext=(155, 9.0),
                 color=PALETTE["amber"], fontsize=10, fontweight="bold",
                 arrowprops=dict(facecolor=PALETTE["amber"], edgecolor=PALETTE["amber"], arrowstyle="->", lw=2.0))

    # Highlight Cycle 222 & 223
    ax1.plot(222, signal[72], marker="s", markersize=10, color=PALETTE["red_alert"])
    ax1.plot(223, signal[73], marker="s", markersize=10, color=PALETTE["red_alert"], label="Ciclo 222-223: Persistência (k≥2) -> Alarme Real!")
    ax1.annotate("k=2: Alarme Preditivo Confirmado!\nLead Time de +213 ciclos!",
                 xy=(223, signal[73]), xytext=(185, 12.0),
                 color=PALETTE["red_alert"], fontsize=10, fontweight="bold",
                 arrowprops=dict(facecolor=PALETTE["red_alert"], edgecolor=PALETTE["red_alert"], arrowstyle="->", lw=2.0))

    ax1.legend(loc="upper left", facecolor=PALETTE["card"], edgecolor=PALETTE["border"], 
               labelcolor=PALETTE["white"], fontsize=9.0)
    ax1.grid(True, color=PALETTE["border"], ls="--", alpha=0.4)

    # Right: FSM Diagram
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("Máquina de Estados Finitos da Trava Industrial", 
                  color=PALETTE["white"], fontsize=12, fontweight="bold")

    # State 0
    s0 = patches.Circle((2.5, 7.5), 1.5, fc=PALETTE["emerald"], ec=PALETTE["white"], lw=2.0, alpha=0.3)
    ax2.add_patch(s0)
    ax2.text(2.5, 7.7, "ESTADO 0\nNORMAL", color=PALETTE["white"], fontsize=11, fontweight="bold", ha="center")
    ax2.text(2.5, 6.7, "k = 0", color=PALETTE["cyan_light"], fontsize=10, fontweight="bold", ha="center")

    # State 1
    s1 = patches.Circle((7.5, 7.5), 1.5, fc=PALETTE["amber"], ec=PALETTE["white"], lw=2.0, alpha=0.3)
    ax2.add_patch(s1)
    ax2.text(7.5, 7.7, "ESTADO 1\nSUSPEITA", color=PALETTE["white"], fontsize=11, fontweight="bold", ha="center")
    ax2.text(7.5, 6.7, "k = 1", color=PALETTE["amber_light"], fontsize=10, fontweight="bold", ha="center")

    # State 2
    s2 = patches.Circle((5.0, 2.5), 1.6, fc=PALETTE["red_alert"], ec=PALETTE["white"], lw=2.0, alpha=0.4)
    ax2.add_patch(s2)
    ax2.text(5.0, 2.8, "ESTADO 2\nALARME REAL", color=PALETTE["white"], fontsize=12, fontweight="bold", ha="center")
    ax2.text(5.0, 1.8, "k ≥ 2 (Disparo)", color=PALETTE["white"], fontsize=10, fontweight="bold", ha="center")

    # Transitions
    # S0 -> S1
    ax2.annotate("", xy=(6.0, 7.8), xytext=(4.0, 7.8),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["amber"], lw=2.5))
    ax2.text(5.0, 8.2, "|e(t)| > 3σ", color=PALETTE["amber"], fontsize=10, fontweight="bold", ha="center")

    # S1 -> S0
    ax2.annotate("", xy=(4.0, 7.2), xytext=(6.0, 7.2),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["emerald"], lw=2.5))
    ax2.text(5.0, 6.5, "|e(t)| ≤ 3σ (Reset)", color=PALETTE["emerald"], fontsize=10, fontweight="bold", ha="center")

    # S1 -> S2
    ax2.annotate("", xy=(5.8, 3.8), xytext=(7.0, 6.2),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["red_alert"], lw=2.5))
    ax2.text(7.1, 4.8, "|e(t+1)| > 3σ\n(Confirmado!)", color=PALETTE["red_alert"], fontsize=10, fontweight="bold")

    out_path = OUTPUT_DIR / "theory_persistence_fsm.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=PALETTE["bg_dark"])
    plt.close()
    print(f"Generated {out_path}")


def generate_sop_flowchart():
    fig, ax = plt.subplots(figsize=(15, 8.5), facecolor=PALETTE["bg_dark"])
    ax.set_facecolor(PALETTE["bg_dark"])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    ax.text(7.5, 8.0, "PROCEDIMENTO OPERACIONAL PADRONIZADO (SOP): MATRIZ EM 4 NÍVEIS", 
            color=PALETTE["cyan_light"], fontsize=17, fontweight="bold", ha="center")
    ax.text(7.5, 7.6, "Diretriz de Manutenção Aeronáutica e Transição Decisória de Chão de Fábrica", 
            color=PALETTE["gray_text"], fontsize=11, ha="center")

    levels = [
        ("NÍVEL 0: NORMALIDADE OPERACIONAL", 
         "• Condição: Nenhum alarme ativo e RUL > 80 ciclos\n• Diretriz: Liberação Irrestrita de Voo Comercial\n• Rotina: Inspeção padrão pré-voo e manutenção preventiva de rotina", 
         PALETTE["emerald"], 5.8),
        ("NÍVEL 1: ALERTA PREDITIVO INICIAL", 
         "• Condição: Primeiro sensor rompe persistência (k ≥ 2) ou 50 < RUL ≤ 80 ciclos\n• Diretriz: Alocação da aeronave em rotas de baixa severidade climática\n• Rotina: Monitoramento diário no Cockpit e amostragem espectrométrica de óleo", 
         PALETTE["amber"], 4.1),
        ("NÍVEL 2: AGENDAMENTO PREVENTIVO PRIORITÁRIO", 
         "• Condição: Múltiplos sensores rompem o envelope ou 20 ≤ RUL ≤ 50 ciclos\n• Diretriz: Reserva compulsória de peças sobressalentes no almoxarifado central\n• Rotina: Bloqueio de janela de hangar em até 10 dias para troca programada", 
         PALETTE["amber_light"], 2.4),
        ("NÍVEL 3: PARADA MANDATÓRIA IMEDIATA", 
         "• Condição: RUL < 20 ciclos ou violação térmica crítica (ΔT50 > +15°C)\n• Diretriz: Interdição Compulsória de Voo\n• Rotina: Desmontagem do motor da asa para inspeção boroscópica e revisão completa", 
         PALETTE["red_alert"], 0.7),
    ]

    for title, desc, col, y in levels:
        box = patches.FancyBboxPatch((1.2, y), 12.6, 1.35, boxstyle="round,pad=0.15",
                                     fc=PALETTE["surface"], ec=col, lw=2.5)
        ax.add_patch(box)
        
        # Color badge tab
        badge = patches.FancyBboxPatch((1.2, y), 0.45, 1.35, boxstyle="round,pad=0.0",
                                       fc=col, ec=col, lw=1.0)
        ax.add_patch(badge)

        ax.text(1.9, y + 0.98, title, color=col, fontsize=12.5, fontweight="bold", va="center")
        ax.text(1.9, y + 0.45, desc, color=PALETTE["white"], fontsize=10, va="center")

    out_path = OUTPUT_DIR / "theory_sop_flowchart.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=PALETTE["bg_dark"])
    plt.close()
    print(f"Generated {out_path}")


def generate_mqtt_eda_inventory():
    fig, ax = plt.subplots(figsize=(15, 8.5), facecolor=PALETTE["bg_dark"])
    ax.set_facecolor(PALETTE["bg_dark"])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    ax.text(7.5, 8.0, "ARQUITETURA ORIENTADA A EVENTOS (EDA) COM MQTT PARA GESTÃO DE ESTOQUE PREDITIVO", 
            color=PALETTE["cyan_light"], fontsize=16, fontweight="bold", ha="center")
    ax.text(7.5, 7.6, "Do Rompimento de Envelope em Borda à Requisição Just-in-Time no ERP/WMS Aeronáutico", 
            color=PALETTE["gray_text"], fontsize=11, ha="center")

    steps = [
        ("1. Borda da Aeronave (Edge FADEC)", 
         "• Turbina hold-out #179 em voo\n• Filtro Causal EWMA + K-Means + Envelope SPC\n• Detecção de quebra persistente (k ≥ 2, Ciclo 222)\n• RUL estimado: 213 ciclos (~35 dias)", 
         PALETTE["cyan"], 1.0, 5.0, 2.5, 2.1),
        ("2. Ingestão & MQTT Broker", 
         "• Conexão Satelital ACARS / 5G Gate Link\n• Protocolo MQTT v5.0 (QoS 1/2, Payload JSON)\n• Tópico: aviation/fleet/u179/alerts/t50\n• Broker Clusterizado EMQX / Eclipse Mosquitto", 
         PALETTE["blue_light"], 3.8, 5.0, 2.6, 2.1),
        ("3. Event Mesh & Streaming", 
         "• Apache Kafka / RabbitMQ Event Backbone\n• Tópico de Eventos: engine.anomaly.confirmed\n• Schema Registry (Apache Avro / Protobuf)\n• Roteamento Desacoplado para Microsserviços", 
         PALETTE["purple"], 6.7, 5.0, 2.6, 2.1),
        ("4. Otimizador de Estoque JIT", 
         "• Cruzamento: Lead Time Fornecedor (14d) vs RUL (35d)\n• Algoritmo de Ponto de Reposição Estocástico (s, S)\n• Dimensionamento Dinâmico de Estoque de Segurança\n• Eliminação de Risco de AOG (Aircraft On Ground)", 
         PALETTE["amber"], 9.6, 5.0, 2.6, 2.1),
        ("5. Execução Física & ERP/WMS", 
         "• Conector API REST / RFC SAP S/4HANA MRO\n• Emissão automática de Ordem de Compra de Palhetas\n• WMS: Reserva de kit de HPT no almoxarifado central\n• Alocação de janela de hangar para pernoite programado", 
         PALETTE["emerald"], 12.5, 5.0, 2.4, 2.1),
    ]

    for title, desc, col, x, y, w, h in steps:
        box = patches.FancyBboxPatch((x - 0.9, y - 1.0), w, h, boxstyle="round,pad=0.12",
                                     fc=PALETTE["surface"], ec=col, lw=2.2)
        ax.add_patch(box)
        
        tab = patches.FancyBboxPatch((x - 0.9, y + h - 1.35), w, 0.35, boxstyle="round,pad=0.0",
                                     fc=col, ec=col, lw=1.0)
        ax.add_patch(tab)

        ax.text(x + w/2 - 0.9, y + h - 1.18, title, color=PALETTE["bg_dark"], fontsize=9.5, fontweight="bold", ha="center", va="center")
        ax.text(x - 0.75, y + 0.1, desc, color=PALETTE["white"], fontsize=8.2, va="center")

        # Arrow to next step
        if x < 12.0:
            ax.annotate("", xy=(x + w - 0.85, y), xytext=(x + w - 0.65, y),
                        arrowprops=dict(arrowstyle="->", color=PALETTE["cyan_light"], lw=2.2))

    # Bottom comparison card: Financial & Operational impact
    bot_card = patches.FancyBboxPatch((1.2, 0.5), 12.6, 1.8, boxstyle="round,pad=0.15",
                                     fc=PALETTE["card"], ec=PALETTE["border"], lw=1.8)
    ax.add_patch(bot_card)
    
    ax.text(1.5, 1.95, "IMPACTO TÉCNICO-ECONÔMICO: MANUTENÇÃO REATIVA (AOG) VS LOGÍSTICA PREDITIVA (JIT)", 
            color=PALETTE["amber_light"], fontsize=11, fontweight="bold")
    
    comp_text = (
        "• Cenário Reativo (AOG): Falha não antecipada -> Aeronave parada no solo custando US$ 150.000/dia + Frete aéreo de emergência (US$ 45.000) = Risco Crítico.\n"
        "• Cenário Preditivo (EDA/MQTT): Alerta precoce (+213 ciclos / ~35 dias) -> Pedido disparado automaticamente via MQTT 3 semanas antes do desgaste crítico.\n"
        "• Resultado de Suprimentos: Lead time de 14 dias absorvido com 21 dias de folga -> Custo de AOG reduzido a ZERO e redução de 42% no capital de giro imobilizado."
    )
    ax.text(1.5, 1.15, comp_text, color=PALETTE["white"], fontsize=9.2, va="center")

    out_path = OUTPUT_DIR / "theory_mqtt_eda_inventory.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=PALETTE["bg_dark"])
    plt.close()
    print(f"Generated {out_path}")


if __name__ == "__main__":
    generate_pipeline_architecture()
    generate_turbofan_stations()
    generate_brayton_cycle()
    generate_persistence_fsm()
    generate_sop_flowchart()
    generate_mqtt_eda_inventory()
    print("All pedagogical theory figures generated successfully!")
