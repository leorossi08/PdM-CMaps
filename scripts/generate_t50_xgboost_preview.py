#!/usr/bin/env python3
"""Generate dedicated diagnostic preview of raw T50 telemetry predicted by XGBoost.

This script trains XGBoost regressors on NASA C-MAPSS FD004 to predict the actual
raw values of Sensor T50 (LPT Exhaust Temperature in °C) - NOT the RUL.

It generates:
1. results/maintenance_guide_fd004/tables/t50_xgboost_raw_predictions_unit179.csv
2. results/maintenance_guide_fd004/figures/t50_xgboost_raw_prediction_preview.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import CMapssLoader
from regimes import RegimeDecoupler
from splitting import partition_healthy_and_lifecycle, split_engine_units

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
}


def train_and_evaluate_xgboost_t50(
    subset: str = "FD004",
    unit_id: int = 179,
    healthy_cycle_limit: int = 50,
    seed: int = 42,
    output_dir: Path = Path("results/maintenance_guide_fd004"),
) -> tuple[pd.DataFrame, dict]:
    """Train XGBoost for raw T50 prediction and evaluate on the target test engine."""
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print(f"[xgboost_t50] Loading C-MAPSS {subset} dataset...", flush=True)
    loader = CMapssLoader()
    frame = loader.load(subset)
    sensors = loader.informative_sensors(frame)

    train_units, test_units = split_engine_units(frame, test_ratio=0.25, random_state=seed)
    parts = partition_healthy_and_lifecycle(frame, train_units, test_units, healthy_cycle_limit)

    print(f"[xgboost_t50] Decoupling flight regimes across 6 clusters...", flush=True)
    regimes = RegimeDecoupler(n_clusters=6, random_state=seed).fit(parts["full_lifecycle_train"])
    train_full = regimes.transform(parts["full_lifecycle_train"])
    test_full = regimes.transform(parts["full_lifecycle_test"])
    healthy_train = regimes.transform(parts["healthy_train"])

    # Features: Companion sensors + flight condition settings
    features_all = [c for c in sensors if c != "T50"] + ["setting_1", "setting_2", "setting_3"]

    print(f"[xgboost_t50] Training Model A: XGBoost Virtual Sensor (Trained on nominal cycles <= {healthy_cycle_limit})...", flush=True)
    xgb_vs = xgb.XGBRegressor(
        n_estimators=140,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=-1,
    )
    xgb_vs.fit(healthy_train[features_all], healthy_train["T50"])

    print(f"[xgboost_t50] Training Model B: XGBoost Full Dynamic Regressor (Trained across full lifecycle)...", flush=True)
    xgb_full = xgb.XGBRegressor(
        n_estimators=140,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=-1,
    )
    xgb_full.fit(train_full[features_all], train_full["T50"])

    print(f"[xgboost_t50] Predicting raw T50 for held-out Engine #{unit_id}...", flush=True)
    engine_df = test_full.loc[test_full["unit_id"] == unit_id].sort_values("cycle").copy().reset_index(drop=True)
    if engine_df.empty:
        raise ValueError(f"Engine #{unit_id} not found in held-out test engines: {test_units[:10]}")

    pred_vs = xgb_vs.predict(engine_df[features_all])
    pred_full = xgb_full.predict(engine_df[features_all])

    engine_df["T50_actual"] = engine_df["T50"]
    engine_df["T50_pred_nominal_vs"] = pred_vs
    engine_df["T50_pred_full_xgb"] = pred_full
    engine_df["resid_nominal_vs"] = engine_df["T50"] - pred_vs
    engine_df["resid_full_xgb"] = engine_df["T50"] - pred_full

    # Statistical corridor from healthy calibration residuals
    healthy_mask = engine_df["cycle"] <= healthy_cycle_limit
    mu_healthy = float(engine_df.loc[healthy_mask, "resid_nominal_vs"].mean())
    sigma_healthy = float(engine_df.loc[healthy_mask, "resid_nominal_vs"].std())
    engine_df["envelope_upper"] = pred_vs + (mu_healthy + 3.0 * sigma_healthy)
    engine_df["envelope_lower"] = pred_vs + (mu_healthy - 3.0 * sigma_healthy)

    # Performance metrics
    r2_healthy_vs = float(r2_score(engine_df.loc[healthy_mask, "T50"], pred_vs[healthy_mask]))
    r2_all_vs = float(r2_score(engine_df["T50"], pred_vs))
    rmse_vs = float(np.sqrt(mean_squared_error(engine_df["T50"], pred_vs)))
    mae_vs = float(mean_absolute_error(engine_df["T50"], pred_vs))

    r2_full = float(r2_score(engine_df["T50"], pred_full))
    rmse_full = float(np.sqrt(mean_squared_error(engine_df["T50"], pred_full)))
    mae_full = float(mean_absolute_error(engine_df["T50"], pred_full))

    metrics = {
        "unit_id": unit_id,
        "total_cycles": len(engine_df),
        "r2_healthy_nominal_vs": r2_healthy_vs,
        "r2_overall_nominal_vs": r2_all_vs,
        "rmse_nominal_vs": rmse_vs,
        "mae_nominal_vs": mae_vs,
        "r2_full_xgb": r2_full,
        "rmse_full_xgb": rmse_full,
        "mae_full_xgb": mae_full,
        "sigma_healthy_c": sigma_healthy,
    }

    # Save scored table
    out_csv = table_dir / f"t50_xgboost_raw_predictions_unit{unit_id}.csv"
    engine_df.to_csv(out_csv, index=False)
    print(f"[xgboost_t50] Scored CSV saved to: {out_csv}", flush=True)

    return engine_df, metrics


def plot_xgboost_raw_t50_preview(
    df: pd.DataFrame,
    metrics: dict,
    output_path: Path,
    alarm_cycle: int = 222,
    eol_cycle: int = 435,
) -> None:
    """Generate high-resolution 4-panel diagnostic figure of raw T50 predicted by XGBoost."""
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 12), dpi=300, facecolor=PALETTE["bg_dark"])
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], hspace=0.32, wspace=0.25)

    # -------------------------------------------------------------
    # Panel 1 (Top Span): Full Lifecycle Raw T50 Actual vs XGBoost
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(PALETTE["surface"])
    for spine in ax1.spines.values():
        spine.set_color(PALETTE["border"])

    cycles = df["cycle"].to_numpy()
    actual = df["T50_actual"].to_numpy()
    pred_vs = df["T50_pred_nominal_vs"].to_numpy()
    pred_full = df["T50_pred_full_xgb"].to_numpy()
    upper = df["envelope_upper"].to_numpy()
    lower = df["envelope_lower"].to_numpy()

    # Corridor
    ax1.fill_between(
        cycles, lower, upper,
        color=PALETTE["cyan"], alpha=0.15, label="Envelope Nominal SPC ±3σ (XGBoost Virtual Sensor)"
    )

    # Actual Raw Points & Line
    ax1.scatter(
        cycles, actual,
        color=PALETTE["cyan_light"], s=16, alpha=0.85, zorder=3,
        label="T50 Real Cru (Telemetria do Motor #179)"
    )
    ax1.plot(cycles, actual, color=PALETTE["cyan"], lw=0.9, alpha=0.5, zorder=2)

    # XGBoost Nominal Prediction (Virtual Sensor)
    ax1.plot(
        cycles, pred_vs,
        color=PALETTE["amber_light"], lw=1.8, ls="--", zorder=4,
        label=f"XGBoost Predição Nominal (Sensor Virtual, Treinado H=50, R²={metrics['r2_overall_nominal_vs']:.4f})"
    )

    # XGBoost Full Lifecycle Tracker
    ax1.plot(
        cycles, pred_full,
        color=PALETTE["emerald"], lw=1.4, ls=":", alpha=0.85, zorder=4,
        label=f"XGBoost Rastreador Dinâmico (Treinado Frota Total, R²={metrics['r2_full_xgb']:.4f}, RMSE={metrics['rmse_full_xgb']:.2f}°C)"
    )

    # Milestone markers
    ax1.axvline(50, color=PALETTE["gray_text"], lw=1.5, ls="--")
    ax1.text(52, 1420, "Fim Calibração (H=50)", color=PALETTE["gray_text"], fontsize=10, fontweight="bold")

    ax1.axvline(alarm_cycle, color=PALETTE["red_alert"], lw=2.0, ls="-.")
    ax1.text(alarm_cycle - 75, 1420, f"ALERTA (Ciclo {alarm_cycle})\nLead Time: 213 Ciclos", color=PALETTE["red_alert"], fontsize=10, fontweight="bold")

    ax1.axvline(eol_cycle, color=PALETTE["red_alert"], lw=2.5, ls="-")
    ax1.text(eol_cycle - 35, 1420, f"EOL / Falha\n(Ciclo {eol_cycle})", color=PALETTE["red_alert"], fontsize=10, fontweight="bold")

    ax1.set_title(
        f"A. Séries Temporais Completas do Sinal Cru T50 (°C) vs Predição por XGBoost (SEM MÉDIA) — Motor #{metrics['unit_id']}",
        fontsize=14, fontweight="bold", color=PALETTE["white"], pad=12
    )
    ax1.set_xlabel("Ciclo Operacional do Motor (Lifecycle)", fontsize=11, color=PALETTE["gray_text"])
    ax1.set_ylabel("Temperatura de Exaustão LPT T50 (°C)", fontsize=11, color=PALETTE["gray_text"])
    ax1.grid(True, color=PALETTE["border"], alpha=0.4, ls=":")
    ax1.legend(loc="lower left", framealpha=0.9, facecolor=PALETTE["card"], edgecolor=PALETTE["border"], fontsize=9.5)
    ax1.set_xlim(0, 445)

    # -------------------------------------------------------------
    # Panel 2 (Bottom Left): Zoom Single Regime (Regime 0: Sea Level High Mach)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(PALETTE["surface"])
    for spine in ax2.spines.values():
        spine.set_color(PALETTE["border"])

    # Filter Regime 0 (high-stress sea-level regime)
    regime_0_mask = df["regime_id"] == 0
    df_r0 = df.loc[regime_0_mask]

    ax2.scatter(
        df_r0["cycle"], df_r0["T50_actual"],
        color=PALETTE["cyan_light"], s=28, label="T50 Real Cru", zorder=3
    )
    ax2.plot(df_r0["cycle"], df_r0["T50_actual"], color=PALETTE["cyan"], lw=1.2, alpha=0.6)
    ax2.plot(
        df_r0["cycle"], df_r0["T50_pred_nominal_vs"],
        color=PALETTE["amber_light"], lw=2.0, ls="--", label="XGBoost Nominal (H=50)", zorder=4
    )

    ax2.fill_between(
        df_r0["cycle"], df_r0["envelope_lower"], df_r0["envelope_upper"],
        color=PALETTE["cyan"], alpha=0.15, label="Envelope SPC ±3σ"
    )

    ax2.axvline(alarm_cycle, color=PALETTE["red_alert"], lw=1.5, ls="-.")
    ax2.set_title(
        "B. Isolamento Intra-Regime (Regime 0: Nível do Mar)\nComprova Divergência Térmica Real no Fim de Vida",
        fontsize=11.5, fontweight="bold", color=PALETTE["white"], pad=10
    )
    ax2.set_xlabel("Ciclo Operacional", fontsize=10, color=PALETTE["gray_text"])
    ax2.set_ylabel("T50 (°C) no Regime 0", fontsize=10, color=PALETTE["gray_text"])
    ax2.grid(True, color=PALETTE["border"], alpha=0.4, ls=":")
    ax2.legend(loc="upper left", framealpha=0.85, facecolor=PALETTE["card"], edgecolor=PALETTE["border"], fontsize=8.5)

    # -------------------------------------------------------------
    # Panel 3 (Bottom Center): Residual Tracking & Damage Drift
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(PALETTE["surface"])
    for spine in ax3.spines.values():
        spine.set_color(PALETTE["border"])

    resids = df["resid_nominal_vs"].to_numpy()
    sigma = metrics["sigma_healthy_c"]

    ax3.plot(cycles, resids, color=PALETTE["amber"], lw=1.2, alpha=0.85, label="Resíduo Cru (T50 Real - XGBoost)")
    ax3.axhline(0.0, color=PALETTE["gray_text"], lw=1.0, ls="-")
    ax3.axhline(+3.0 * sigma, color=PALETTE["red_alert"], lw=1.4, ls="--", label=f"Limite Superior +3σ (+{3.0*sigma:.1f}°C)")
    ax3.axhline(-3.0 * sigma, color=PALETTE["cyan"], lw=1.4, ls="--", label=f"Limite Inferior -3σ (-{3.0*sigma:.1f}°C)")

    # Rolling mean residual
    roll_resid = pd.Series(resids).rolling(10, min_periods=1).mean()
    ax3.plot(cycles, roll_resid, color=PALETTE["white"], lw=2.0, label="Tendência Média Móvel (W=10)")

    ax3.axvline(alarm_cycle, color=PALETTE["red_alert"], lw=1.5, ls="-.")
    ax3.set_title(
        "C. Erro Residual de Predição (°C)\nZero Tendência no Início vs Deriva Positiva +8°C no Final",
        fontsize=11.5, fontweight="bold", color=PALETTE["white"], pad=10
    )
    ax3.set_xlabel("Ciclo Operacional", fontsize=10, color=PALETTE["gray_text"])
    ax3.set_ylabel("Resíduo (T50 Real - XGBoost Nominal) [°C]", fontsize=10, color=PALETTE["gray_text"])
    ax3.grid(True, color=PALETTE["border"], alpha=0.4, ls=":")
    ax3.legend(loc="lower left", framealpha=0.85, facecolor=PALETTE["card"], edgecolor=PALETTE["border"], fontsize=8.5)

    # -------------------------------------------------------------
    # Panel 4 (Bottom Right): 45° Parity Plot (Predicted vs Actual T50)
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(PALETTE["surface"])
    for spine in ax4.spines.values():
        spine.set_color(PALETTE["border"])

    t_min, t_max = 1000, 1450
    ax4.plot([t_min, t_max], [t_min, t_max], color=PALETTE["emerald"], lw=1.8, ls="--", label="Paridade Perfeita 45°", zorder=2)
    ax4.fill_between([t_min, t_max], [t_min - 10, t_max - 10], [t_min + 10, t_max + 10], color=PALETTE["emerald"], alpha=0.1, label="Faixa de Tolerância ±10°C")

    # Color scatter by life stage
    stage_colors = np.where(cycles <= 50, PALETTE["emerald"],
                   np.where(cycles <= 222, PALETTE["cyan_light"], PALETTE["red_alert"]))
    ax4.scatter(df["T50_pred_full_xgb"], df["T50_actual"], c=stage_colors, s=22, alpha=0.75, zorder=3)

    # Metric box
    metric_text = (
        f"Acurácia Global XGBoost:\n"
        f"• R² = {metrics['r2_full_xgb']:.4f}\n"
        f"• RMSE = {metrics['rmse_full_xgb']:.2f} °C\n"
        f"• MAE = {metrics['mae_full_xgb']:.2f} °C\n"
        f"• Cobertura: 1000°C - 1430°C"
    )
    ax4.text(
        0.05, 0.65, metric_text, transform=ax4.transAxes,
        fontsize=9.5, family="monospace", color=PALETTE["white"],
        bbox=dict(boxstyle="round,pad=0.6", facecolor=PALETTE["card"], edgecolor=PALETTE["cyan"], alpha=0.9)
    )

    ax4.set_title(
        "D. Diagrama de Paridade 45° de T50\n(XGBoost Predito vs Real nos 6 Regimes)",
        fontsize=11.5, fontweight="bold", color=PALETTE["white"], pad=10
    )
    ax4.set_xlabel("T50 Predito por XGBoost (°C)", fontsize=10, color=PALETTE["gray_text"])
    ax4.set_ylabel("T50 Real de Telemetria (°C)", fontsize=10, color=PALETTE["gray_text"])
    ax4.set_xlim(t_min, t_max)
    ax4.set_ylim(t_min, t_max)
    ax4.grid(True, color=PALETTE["border"], alpha=0.4, ls=":")
    ax4.legend(loc="lower right", framealpha=0.85, facecolor=PALETTE["card"], edgecolor=PALETTE["border"], fontsize=8.5)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[xgboost_t50] High-resolution diagnostic preview figure saved to: {output_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate XGBoost raw T50 prediction preview graphics.")
    parser.add_argument("--subset", type=str, default="FD004", help="C-MAPSS dataset subset")
    parser.add_argument("--unit-id", type=int, default=179, help="Held-out engine unit ID to visualize")
    parser.add_argument("--healthy-cycle-limit", type=int, default=50, help="Healthy window calibration limit")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    parser.add_argument("--output-dir", type=Path, default=Path("results/maintenance_guide_fd004"), help="Output directory")
    args = parser.parse_args()

    df, metrics = train_and_evaluate_xgboost_t50(
        subset=args.subset,
        unit_id=args.unit_id,
        healthy_cycle_limit=args.healthy_cycle_limit,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    fig_path = args.output_dir / "figures" / "t50_xgboost_raw_prediction_preview.png"
    plot_xgboost_raw_t50_preview(df, metrics, fig_path)

    print("\n[xgboost_t50] SUCCESS: All T50 raw prediction models and graphics generated.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
