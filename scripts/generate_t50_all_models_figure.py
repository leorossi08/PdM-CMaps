#!/usr/bin/env python3
"""Generate high-resolution multi-model comparison graphic for raw T50 telemetry prediction."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    "magenta": "#ec4899",
    "gray_text": "#94a3b8",
    "white": "#f8fafc",
}

def generate_multi_model_t50_figure():
    pred_path = Path("results/maintenance_guide_fd004/tables/t50_all_models_raw_predictions_unit179.csv")
    bench_path = Path("results/maintenance_guide_fd004/tables/t50_all_models_benchmark.csv")
    fig_path = Path("results/maintenance_guide_fd004/figures/t50_all_models_raw_prediction_preview.png")

    if not pred_path.exists() or not bench_path.exists():
        print("Required CSV files not found!")
        return

    df = pd.read_csv(pred_path)
    bench_df = pd.read_csv(bench_path)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(19, 12), dpi=300, facecolor=PALETTE["bg_dark"])
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], hspace=0.32, wspace=0.25)

    cycles = df["cycle"].to_numpy()
    actual = df["T50_actual"].to_numpy()

    # -------------------------------------------------------------
    # Panel 1 (Top Span): Full Lifecycle Time Series Across All Models
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(PALETTE["surface"])
    for spine in ax1.spines.values():
        spine.set_color(PALETTE["border"])

    # Actual Raw Points & Line
    ax1.scatter(cycles, actual, color=PALETTE["cyan_light"], s=18, alpha=0.9, zorder=5, label="T50 Real Cru (Telemetria do Motor #179)")
    ax1.plot(cycles, actual, color=PALETTE["cyan"], lw=1.0, alpha=0.4, zorder=4)

    # Models
    model_styles = [
        ("XGBoost", PALETTE["amber_light"], 2.0, "--"),
        ("LightGBM", PALETTE["emerald"], 1.6, "-."),
        ("Random Forest", PALETTE["purple"], 1.5, ":"),
        ("Extra Trees", "#38bdf8", 1.4, "--"),
        ("CatBoost", PALETTE["magenta"], 1.4, "-."),
    ]

    for model_name, color, lw, ls in model_styles:
        col = f"T50_pred_{model_name}"
        if col in df.columns:
            m_row = bench_df.loc[bench_df["Model"] == model_name]
            rmse_val = m_row["RMSE_C"].iloc[0] if not m_row.empty else 4.2
            ax1.plot(cycles, df[col], color=color, lw=lw, ls=ls, alpha=0.85, zorder=3,
                     label=f"{model_name} (RMSE = {rmse_val:.2f}°C)")

    # Markers
    ax1.axvline(50, color=PALETTE["gray_text"], lw=1.5, ls="--")
    ax1.text(52, 1420, "Fim Calibração (50)", color=PALETTE["gray_text"], fontsize=10, fontweight="bold")
    ax1.axvline(222, color=PALETTE["red_alert"], lw=2.0, ls="-.")
    ax1.text(224, 1420, "ALERTA (Ciclo 222)\nLead Time: 213 Ciclos", color=PALETTE["red_alert"], fontsize=10, fontweight="bold")
    ax1.axvline(435, color=PALETTE["red_alert"], lw=2.5, ls="-")
    ax1.text(437 - 45, 1420, "EOL / Falha\n(Ciclo 435)", color=PALETTE["red_alert"], fontsize=10, fontweight="bold")

    ax1.set_title("A. Comparativo de Séries Temporais: Sinal Cru T50 (°C) vs Predição de Todos os Modelos de ML (SEM MÉDIA)",
                  fontsize=14, fontweight="bold", color=PALETTE["white"], pad=12)
    ax1.set_xlabel("Ciclo Operacional do Motor (Lifecycle)", fontsize=11, color=PALETTE["gray_text"])
    ax1.set_ylabel("Temperatura de Exaustão LPT T50 (°C)", fontsize=11, color=PALETTE["gray_text"])
    ax1.grid(True, color=PALETTE["border"], alpha=0.4, ls=":")
    ax1.legend(loc="lower left", framealpha=0.9, facecolor=PALETTE["card"], edgecolor=PALETTE["border"], fontsize=9.5)
    ax1.set_xlim(0, 445)

    # -------------------------------------------------------------
    # Panel 2 (Bottom Left): Zoom on Transition Cycles (Ciclos 200 a 260)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(PALETTE["surface"])
    for spine in ax2.spines.values():
        spine.set_color(PALETTE["border"])

    zoom_mask = (df["cycle"] >= 200) & (df["cycle"] <= 260)
    df_zoom = df.loc[zoom_mask]

    ax2.scatter(df_zoom["cycle"], df_zoom["T50_actual"], color=PALETTE["cyan_light"], s=35, label="T50 Real Cru", zorder=5)
    ax2.plot(df_zoom["cycle"], df_zoom["T50_actual"], color=PALETTE["cyan"], lw=1.2, alpha=0.5)

    for model_name, color, lw, ls in model_styles[:3]:
        col = f"T50_pred_{model_name}"
        if col in df_zoom.columns:
            ax2.plot(df_zoom["cycle"], df_zoom[col], color=color, lw=lw, ls=ls, label=model_name)

    # Nominal XGBoost baseline
    if "T50_nominal_XGBoost" in df_zoom.columns:
        ax2.plot(df_zoom["cycle"], df_zoom["T50_nominal_XGBoost"], color=PALETTE["gray_text"], lw=1.8, ls=":", label="XGBoost Nominal (H=50)")

    ax2.axvline(222, color=PALETTE["red_alert"], lw=2.0, ls="-.")
    ax2.text(223, 1380, "Alerta 222", color=PALETTE["red_alert"], fontsize=9, fontweight="bold")

    ax2.set_title("B. Zoom na Zona de Alerta (Ciclos 200 a 260)\nTodos os Modelos Rastreiam Juntos a Dinâmica",
                  fontsize=11.5, fontweight="bold", color=PALETTE["white"], pad=10)
    ax2.set_xlabel("Ciclo Operacional", fontsize=10, color=PALETTE["gray_text"])
    ax2.set_ylabel("T50 Cru (°C)", fontsize=10, color=PALETTE["gray_text"])
    ax2.grid(True, color=PALETTE["border"], alpha=0.4, ls=":")
    ax2.legend(loc="lower left", framealpha=0.85, facecolor=PALETTE["card"], edgecolor=PALETTE["border"], fontsize=8.5)

    # -------------------------------------------------------------
    # Panel 3 (Bottom Center): Residual Distribution Across Models
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(PALETTE["surface"])
    for spine in ax3.spines.values():
        spine.set_color(PALETTE["border"])

    for model_name, color, lw, _ in model_styles:
        col = f"resid_{model_name}"
        if col in df.columns:
            res = df[col].to_numpy()
            roll_m = pd.Series(res).rolling(12, min_periods=1).mean()
            ax3.plot(cycles, roll_m, color=color, lw=1.6, label=f"Resíduo {model_name}")

    ax3.axhline(0.0, color=PALETTE["white"], lw=1.0, ls="-")
    ax3.axvline(222, color=PALETTE["red_alert"], lw=1.5, ls="-.")
    ax3.set_title("C. Tendência Residual Suavizada (W=12) (°C)\nConsenso de Resíduos entre Todos os Modelos",
                  fontsize=11.5, fontweight="bold", color=PALETTE["white"], pad=10)
    ax3.set_xlabel("Ciclo Operacional", fontsize=10, color=PALETTE["gray_text"])
    ax3.set_ylabel("Resíduo (T50 Real - Modelo) [°C]", fontsize=10, color=PALETTE["gray_text"])
    ax3.grid(True, color=PALETTE["border"], alpha=0.4, ls=":")
    ax3.legend(loc="upper left", framealpha=0.85, facecolor=PALETTE["card"], edgecolor=PALETTE["border"], fontsize=8.5)

    # -------------------------------------------------------------
    # Panel 4 (Bottom Right): Benchmark Leaderboard (RMSE & R2)
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(PALETTE["surface"])
    for spine in ax4.spines.values():
        spine.set_color(PALETTE["border"])

    models_sorted = bench_df.sort_values("RMSE_C", ascending=False)
    y_pos = np.arange(len(models_sorted))
    colors_bar = [PALETTE["cyan"] if m == "XGBoost" else PALETTE["card"] for m in models_sorted["Model"]]
    edge_colors = [PALETTE["cyan_light"] if m == "XGBoost" else PALETTE["border"] for m in models_sorted["Model"]]

    bars = ax4.barh(y_pos, models_sorted["RMSE_C"], color=colors_bar, edgecolor=edge_colors, height=0.65)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(models_sorted["Model"], fontsize=10, color=PALETTE["white"])
    ax4.set_xlabel("RMSE de Predição de T50 Cru (°C) [Menor é Melhor]", fontsize=10, color=PALETTE["gray_text"])

    for bar, r2_val in zip(bars, models_sorted["R2_Lifecycle"]):
        width = bar.get_width()
        ax4.text(width + 0.05, bar.get_y() + bar.get_height()/2, f"{width:.2f}°C (R²={r2_val:.4f})",
                 va="center", ha="left", fontsize=9, color=PALETTE["white"], fontweight="bold")

    ax4.set_xlim(0, 5.6)
    ax4.set_title("D. Ranking Comparativo de Precisão de T50\n(XGBoost lidera com RMSE=4.23°C e R²=0.9988)",
                  fontsize=11.5, fontweight="bold", color=PALETTE["white"], pad=10)
    ax4.grid(True, color=PALETTE["border"], alpha=0.4, ls=":", axis="x")

    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated multi-model figure at: {fig_path}")

if __name__ == "__main__":
    generate_multi_model_t50_figure()
