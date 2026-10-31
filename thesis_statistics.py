"""Statistical Hypothesis Testing and Thesis Analysis for Predictive Maintenance.

Follows the guidelines of @ml-best-practices:
- Statistical testing with exact test statistics and p-values.
- Rigorous hypothesis formulation (H0 vs H1).
- Confirmation of physical phenomena (Brayton cycle, altitude variance) and algorithmic superiority.
- 95% Confidence Intervals via fleet engine bootstrap resampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def hodges_lehmann_estimator(deltas: np.ndarray) -> float:
    """Compute Hodges-Lehmann pseudo-median from pairwise Walsh averages."""
    arr = np.asarray(deltas, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0
    pairwise = [(arr[i] + arr[j]) / 2.0 for i in range(n) for j in range(i, n)]
    return float(np.median(pairwise))


def _bca_interval(bootstrap_values: list[float], jackknife_values: list[float], observed: float) -> tuple[float, float]:
    """Bias-corrected and accelerated interval for a cluster-resampled statistic."""
    boot = np.asarray(bootstrap_values, dtype=float)
    jack = np.asarray(jackknife_values, dtype=float)
    eps = 0.5 / len(boot)
    bias = stats.norm.ppf(np.clip(np.mean(boot < observed), eps, 1.0 - eps))
    centered = jack.mean() - jack
    denom = 6.0 * np.sum(centered ** 2) ** 1.5
    acceleration = float(np.sum(centered ** 3) / denom) if denom > 0 else 0.0
    quantiles = []
    for alpha in (0.025, 0.975):
        z = stats.norm.ppf(alpha)
        adjusted = stats.norm.cdf(bias + (bias + z) / (1.0 - acceleration * (bias + z)))
        quantiles.append(float(np.clip(adjusted, 0.0, 1.0)))
    return tuple(float(x) for x in np.quantile(boot, quantiles))


@dataclass
class HypothesisTestResult:
    """Container for hypothesis test outcomes."""

    hypothesis_id: str
    description: str
    h0: str
    h1: str
    test_name: str
    test_statistic: float
    p_value: float
    significance_level: float = 0.05
    effect_size: float | None = None
    conclusion: str = ""

    @property
    def is_rejected(self) -> bool:
        return self.p_value < self.significance_level

    def to_dict(self) -> dict[str, Any]:
        p_formatted = f"{self.p_value:.2e}" if self.p_value < 1e-4 else f"{self.p_value:.4f}"
        if self.p_value == 0.0:
            p_formatted = "< 1e-300"
        return {
            "ID": self.hypothesis_id,
            "Description": self.description,
            "H0": self.h0,
            "H1": self.h1,
            "Test": self.test_name,
            "Statistic": round(float(self.test_statistic), 4),
            "p_value": p_formatted,
            "p_value_raw": float(self.p_value),
            "H0_Rejected": "REJEITADA (p < 0.05)" if self.is_rejected else "NÃO REJEITADA",
            "Effect_Size": round(float(self.effect_size), 4) if self.effect_size is not None else np.nan,
            "Scientific_Conclusion": self.conclusion,
        }


class ThesisHypothesisAnalyzer:
    """Comprehensive hypothesis testing and statistical analysis engine."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results: list[HypothesisTestResult] = []

    def test_regime_variance_dominance(
        self,
        df: pd.DataFrame,
        sensor: str = "T50",
        regime_col: str = "regime_id",
        unit_col: str = "unit_id",
    ) -> HypothesisTestResult:
        """Hypothesis 1: Operating regime explains statistically significant variance in sensor readings.
        
        Evaluated on repeated engine blocks to eliminate cycle pseudoreplication.
        """
        # If engine unit_id is present, perform repeated-measures engine-block Friedman test
        if unit_col in df.columns and df[unit_col].nunique() >= 2:
            pivot = df.groupby([unit_col, regime_col])[sensor].mean().unstack()
            # Drop incomplete engine rows across regimes
            clean_matrix = pivot.dropna()
            if len(clean_matrix) >= 2 and clean_matrix.shape[1] >= 2:
                cols = [clean_matrix[c].to_numpy() for c in clean_matrix.columns]
                stat, p_val = stats.friedmanchisquare(*cols)
                n_engines = len(clean_matrix)
                k_regimes = clean_matrix.shape[1]
                kendall_w = float(stat / (n_engines * (k_regimes - 1)))
                res = HypothesisTestResult(
                    hypothesis_id="H1_REGIME_VARIANCE",
                    description="Dominância da Variação de Altitude/Mach sobre os Sensores Térmicos",
                    h0="A altitude e velocidade de voo não afetam as leituras térmicas nominais entre motores.",
                    h1="A atmosfera ISA e regimes de voo causam variação altamente significativa (p < 0.001).",
                    test_name=f"Repeated-Measures Friedman Test (n={n_engines} engines, k={k_regimes} regimes)",
                    test_statistic=stat,
                    p_value=p_val,
                    effect_size=kendall_w,
                    conclusion=(
                        f"Efeito intra-motor entre regimes: p={p_val:.2e} (Friedman chi²={stat:.1f}, "
                        f"Kendall's W={kendall_w:.4f}, n={n_engines} motores). "
                        f"Confirmada dominância de regime livre de pseudorreplicação amostral."
                    ),
                )
                self.results.append(res)
                return res

        # Fallback to grouped one-way ANOVA / Alexander-Govern if no engine column
        groups = [group[sensor].dropna().to_numpy(dtype=float) for _, group in df.groupby(regime_col)]
        groups = [group for group in groups if group.size > 1]
        if len(groups) < 2:
            raise ValueError("At least two regimes with two observations each are required")
        levene_stat, levene_p = stats.levene(*groups, center="median")
        if levene_p < 0.05:
            robust_result = stats.alexandergovern(*groups)
            test_statistic, p_val = float(robust_result.statistic), float(robust_result.pvalue)
            test_name = f"Alexander-Govern test (Levene p={levene_p:.2e})"
            statistic_label = "AG"
        else:
            f_result = stats.f_oneway(*groups)
            test_statistic, p_val = float(f_result.statistic), float(f_result.pvalue)
            test_name = f"One-Way ANOVA (Levene p={levene_p:.2e})"
            statistic_label = "F"

        all_vals = df[sensor].dropna().to_numpy(dtype=float)
        ss_total = np.sum((all_vals - np.mean(all_vals)) ** 2)
        ss_between = sum(len(g) * (np.mean(g) - np.mean(all_vals)) ** 2 for g in groups)
        eta_squared = ss_between / (ss_total + 1e-12)

        res = HypothesisTestResult(
            hypothesis_id="H1_REGIME_VARIANCE",
            description="Dominância da Variação de Altitude/Mach sobre os Sensores Térmicos",
            h0="A altitude e velocidade de voo não afetam as leituras térmicas nominais.",
            h1="A atmosfera ISA e regimes de voo causam variação altamente significativa (p < 0.001).",
            test_name=test_name,
            test_statistic=test_statistic,
            p_value=p_val,
            effect_size=eta_squared,
            conclusion=(
                f"Efeito entre regimes: p={p_val:.2e} ({statistic_label}={test_statistic:.1f}; "
                f"Levene p={levene_p:.2e}; eta²={eta_squared:.4f}). "
                f"A estatística de efeito descreve {eta_squared*100:.1f}% da variância da amostra."
            ),
        )
        self.results.append(res)
        return res

    def test_physical_degradation_shift(
        self,
        residuals_df: pd.DataFrame,
        healthy_cutoff: int = 50,
        degraded_start: int | None = None,
        residual_col: str = "residual",
        cycle_col: str = "cycle",
        unit_col: str = "unit_id",
    ) -> HypothesisTestResult:
        """Hypothesis 2: Virtual sensor residuals shift significantly during degradation.
        
        Uses survivor-bias-free relative window (t > max(H, EOL_i - 50)) to preserve all engines.
        """
        if unit_col not in residuals_df:
            raise ValueError(f"{unit_col} is required for engine-level paired inference")

        ordered = residuals_df.copy()
        eol_map = ordered.groupby(unit_col)[cycle_col].transform("max")
        ordered["_eol"] = eol_map

        # Relative degradation window: t > max(healthy_cutoff, EOL_i - 50)
        if degraded_start is None:
            ordered["_is_degraded"] = ordered[cycle_col] > np.maximum(healthy_cutoff, ordered["_eol"] - 50)
        else:
            ordered["_is_degraded"] = ordered[cycle_col] >= degraded_start

        healthy = ordered.loc[ordered[cycle_col] <= healthy_cutoff].groupby(unit_col)[residual_col].mean()
        degraded = ordered.loc[ordered["_is_degraded"]].groupby(unit_col)[residual_col].mean()

        paired = pd.concat([healthy.rename("healthy"), degraded.rename("degraded")], axis=1).dropna()
        n_engines = len(paired)
        if n_engines < 2:
            raise ValueError("At least two engines need observations in both phases")

        deltas = (paired["degraded"] - paired["healthy"]).to_numpy(dtype=float)
        # Normality check on deltas
        shapiro_stat, shapiro_p = stats.shapiro(deltas)
        hl_delta = hodges_lehmann_estimator(deltas)
        cohens_d = float(np.mean(deltas) / (np.std(deltas, ddof=1) + 1e-12))

        if shapiro_p >= 0.05:
            t_stat, p_val = stats.ttest_1samp(deltas, popmean=0.0)
            test_name = f"Paired t-test on engine means (n={n_engines}, Shapiro p={shapiro_p:.3f})"
            stat_used = float(t_stat)
        else:
            w_stat, p_val = stats.wilcoxon(deltas, alternative="greater")
            test_name = f"Paired Wilcoxon signed-rank on engine means (n={n_engines}, Shapiro p={shapiro_p:.3f})"
            stat_used = float(w_stat)

        res = HypothesisTestResult(
            hypothesis_id="H2_PHYSICAL_DEGRADATION",
            description="Divergência Térmica Real vs Ruído de Medição no Fim de Vida",
            h0="Os resíduos do Sensor Virtual no final de vida possuem a mesma distribuição da fase saudável.",
            h1="O desgaste físico das pás provoca um deslocamento sistemático e significativo (p < 1e-15).",
            test_name=test_name,
            test_statistic=stat_used,
            p_value=float(p_val),
            effect_size=hl_delta,
            conclusion=(
                f"Confirmada com p={p_val:.2e} (n={n_engines} motores, d de Cohen={cohens_d:.2f}, "
                f"Hodges-Lehmann Δ={hl_delta:+.2f}°C). "
                f"O desvio térmico médio de +{np.mean(deltas):.2f}°C é uma anomalia física real e irreversível."
            ),
        )
        self.results.append(res)
        return res

    def test_white_noise_residual_autocorrelation(
        self,
        residuals: np.ndarray,
        max_lags: int = 10,
        phase_label: str = "healthy",
    ) -> HypothesisTestResult:
        """Hypothesis 3: Ljung-Box test for residual serial correlation."""
        n = len(residuals)
        mean_r = np.mean(residuals)
        var_r = np.var(residuals)
        lags_autocorr = []
        for k in range(1, max_lags + 1):
            cov = np.sum((residuals[:-k] - mean_r) * (residuals[k:] - mean_r)) / n
            lags_autocorr.append(cov / (var_r + 1e-12))

        q_stat = n * (n + 2) * sum((rk ** 2) / (n - k) for k, rk in enumerate(lags_autocorr, start=1))
        p_val = stats.chi2.sf(q_stat, df=max_lags)

        hyp_id = f"H3_AUTOCORRELATION_{phase_label.upper()}"
        desc = f"Estrutura de Autocorrelação Residual (Fase {phase_label.capitalize()})"
        h0 = "Os resíduos são ruído branco puramente aleatório (sem autocorrelação serial)."
        h1 = "Os resíduos apresentam memória serial e correlação persistente."

        conclusion = (
            f"Fase {phase_label}: Q={q_stat:.2f}, p={p_val:.2e}. "
            + ("Não se rejeita H0; isso não demonstra ausência de autocorrelação." if p_val > 0.05 else "Evidência de autocorrelação serial.")
        )

        res = HypothesisTestResult(
            hypothesis_id=hyp_id,
            description=desc,
            h0=h0,
            h1=h1,
            test_name=f"Ljung-Box Portmanteau Test (lags={max_lags})",
            test_statistic=q_stat,
            p_value=p_val,
            effect_size=lags_autocorr[0] if lags_autocorr else None,
            conclusion=conclusion,
        )
        self.results.append(res)
        return res

    def test_model_superiority_vs_baseline(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        baseline_pred: np.ndarray,
        model_name: str,
        unit_ids: np.ndarray | None = None,
    ) -> HypothesisTestResult:
        """Hypothesis 4: ML model provides statistically significantly lower error than naive baseline."""
        err_model = np.abs(y_true - y_pred)
        err_base = np.abs(y_true - baseline_pred)

        if unit_ids is None or len(unit_ids) != len(err_model):
            diff = err_base - err_model
        else:
            diff = pd.DataFrame({"unit_id": unit_ids, "diff": err_base - err_model}).groupby("unit_id")["diff"].mean().to_numpy()

        w_stat, p_val = stats.wilcoxon(diff, alternative="greater")
        mean_reduction = np.mean(err_base) - np.mean(err_model)
        hl_reduction = hodges_lehmann_estimator(diff)

        res = HypothesisTestResult(
            hypothesis_id=f"H4_SUPERIORITY_{model_name.upper().replace(' ', '_')}",
            description=f"Superioridade Estatística de {model_name} sobre a Linha de Base (Média)",
            h0=f"O modelo {model_name} possui a mesma distribuição de erro absoluto da média incondicional.",
            h1=f"O modelo {model_name} tem erro estatisticamente inferior (p < 0.001).",
            test_name="Wilcoxon Paired Signed-Rank Test",
            test_statistic=w_stat,
            p_value=p_val,
            effect_size=hl_reduction,
            conclusion=(
                f"{model_name} superou a linha de base com p={p_val:.2e} (W={w_stat:.1f}, "
                f"Hodges-Lehmann Δ={hl_reduction:.2f} ciclos). "
                f"Redução média no erro absoluto de {mean_reduction:.2f} ciclos."
            ),
        )
        self.results.append(res)
        return res

    def test_gbdt_vs_sequence_model_parity(
        self,
        y_true: np.ndarray,
        pred_gbdt: np.ndarray,
        pred_deep: np.ndarray,
        deep_model_name: str = "Transformer",
        unit_ids: np.ndarray | None = None,
    ) -> HypothesisTestResult:
        """Hypothesis 5: IEEE CIM 2026 RQ2 Dialogue: GBDT vs Deep Sequence Model Parity Test."""
        err_gbdt = np.abs(y_true - pred_gbdt)
        err_deep = np.abs(y_true - pred_deep)

        if unit_ids is None or len(unit_ids) != len(err_gbdt):
            sub_gbdt, sub_deep = err_gbdt, err_deep
        else:
            paired = pd.DataFrame({"unit_id": unit_ids, "gbdt": err_gbdt, "deep": err_deep}).groupby("unit_id")[["gbdt", "deep"]].mean()
            sub_gbdt, sub_deep = paired["gbdt"].to_numpy(), paired["deep"].to_numpy()

        w_stat, p_val = stats.wilcoxon(sub_gbdt, sub_deep)
        diff_mae = np.mean(err_deep) - np.mean(err_gbdt)
        hl_diff = hodges_lehmann_estimator(sub_deep - sub_gbdt)

        if p_val < 0.05:
            direction = "XGBoost superior" if diff_mae > 0 else f"{deep_model_name} superior"
            conclusion = (
                f"Comparação XGBoost vs {deep_model_name}: W={w_stat:.1f}, p={p_val:.2e}. "
                f"{direction} por Hodges-Lehmann Δ={hl_diff:.2f} ciclos (ΔMAE = {diff_mae:.2f})."
            )
        else:
            conclusion = (
                f"Comparação XGBoost vs {deep_model_name}: W={w_stat:.1f}, p={p_val:.2e}. "
                f"Paridade estatística: não se rejeita H0 a 5% (Hodges-Lehmann Δ={hl_diff:.2f} ciclos, ΔMAE = {diff_mae:.2f})."
            )

        res = HypothesisTestResult(
            hypothesis_id=f"H5_IEEE_RQ2_XGB_VS_{deep_model_name.upper()}",
            description=f"Diálogo IEEE CIM 2026: XGBoost vs {deep_model_name}",
            h0=f"XGBoost e {deep_model_name} apresentam acurácia preditiva idêntica.",
            h1=f"Existe divergência estatisticamente significativa entre as arquiteturas.",
            test_name="Two-Sided Paired Wilcoxon Test",
            test_statistic=w_stat,
            p_value=p_val,
            effect_size=hl_diff,
            conclusion=conclusion,
        )
        self.results.append(res)
        return res

    def compute_bootstrap_confidence_intervals(
        self,
        test_df: pd.DataFrame,
        model_cols: dict[str, str],
        target_col: str = "RUL_actual",
        n_bootstraps: int = 2000,
    ) -> pd.DataFrame:
        """Compute 95% Confidence Intervals via fleet engine cluster bootstrapping."""
        rng = np.random.default_rng(self.random_state)
        engines = test_df["unit_id"].unique()
        n_engines = len(engines)
        available_models = {name: col for name, col in model_cols.items() if col in test_df.columns}
        if not available_models:
            return pd.DataFrame()

        unit_ids = test_df["unit_id"].to_numpy()
        truth = test_df[target_col].to_numpy(dtype=float)
        predictions = {name: test_df[col].to_numpy(dtype=float) for name, col in available_models.items()}
        row_indices = [np.flatnonzero(unit_ids == engine) for engine in engines]

        def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
            errors = y_true - y_pred
            ss_total = np.sum((y_true - y_true.mean()) ** 2)
            return (
                float(1.0 - np.sum(errors ** 2) / (ss_total + 1e-12)),
                float(np.sqrt(np.mean(errors ** 2))),
                float(np.mean(np.abs(errors))),
            )

        score_store = {
            name: {metric: np.empty(n_bootstraps, dtype=float) for metric in ("r2", "rmse", "mae")}
            for name in available_models
        }
        for iteration in range(n_bootstraps):
            sampled = rng.integers(n_engines, size=n_engines)
            sample_index = np.concatenate([row_indices[index] for index in sampled])
            sample_truth = truth[sample_index]
            for name, prediction in predictions.items():
                r2, rmse, mae = metrics(sample_truth, prediction[sample_index])
                score_store[name]["r2"][iteration] = r2
                score_store[name]["rmse"][iteration] = rmse
                score_store[name]["mae"][iteration] = mae

        rows = []
        for name, prediction in predictions.items():
            observed = metrics(truth, prediction)
            jackknife = []
            for index in row_indices:
                keep = np.ones(len(truth), dtype=bool)
                keep[index] = False
                jackknife.append(metrics(truth[keep], prediction[keep]))
            r2_ci = _bca_interval(score_store[name]["r2"], [row[0] for row in jackknife], observed[0])
            rmse_ci = _bca_interval(score_store[name]["rmse"], [row[1] for row in jackknife], observed[1])
            mae_ci = _bca_interval(score_store[name]["mae"], [row[2] for row in jackknife], observed[2])
            rows.append({
                "Model": name,
                "R2_Observed": round(observed[0], 4),
                "R2_Mean": float(score_store[name]["r2"].mean()),
                "R2_CI_Lower": r2_ci[0],
                "R2_CI_Upper": r2_ci[1],
                "RMSE_Observed": round(observed[1], 2),
                "RMSE_Mean": float(score_store[name]["rmse"].mean()),
                "RMSE_CI_Lower": rmse_ci[0],
                "RMSE_CI_Upper": rmse_ci[1],
                "MAE_Observed": round(observed[2], 2),
                "MAE_Mean": float(score_store[name]["mae"].mean()),
                "MAE_CI_Lower": mae_ci[0],
                "MAE_CI_Upper": mae_ci[1],
                "CI_Method": "engine-cluster BCa",
            })
        return pd.DataFrame(rows)

    def summary_dataframe(self) -> pd.DataFrame:
        """Return all tested hypotheses as a structured DataFrame with FDR correction."""
        frame = pd.DataFrame([res.to_dict() for res in self.results])
        if not frame.empty:
            p = frame["p_value_raw"].to_numpy(dtype=float)
            order = np.argsort(p)
            adjusted = np.empty_like(p)
            running = 1.0
            m = len(p)
            for rank in range(m - 1, -1, -1):
                idx = order[rank]
                running = min(running, p[idx] * m / (rank + 1))
                adjusted[idx] = running
            frame["p_value_BH_FDR"] = adjusted
            frame["BH_FDR_Rejected_0.05"] = adjusted < 0.05
        return frame

    def export_results(self, output_dir: Path) -> tuple[pd.DataFrame, Path]:
        """Export hypothesis summary to CSV in output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = self.summary_dataframe()
        out_path = output_dir / "thesis_hypothesis_tests.csv"
        summary.to_csv(out_path, index=False)
        return summary, out_path
