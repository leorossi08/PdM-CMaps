"""Run the controlled grouped-CV RUL comparison and score diagnostics."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score

from pdm_cmaps.data import CMapssLoader
from pdm_cmaps.diagnostics import (
    capacity_sweep, compare_r2_subsets, compare_split_strategies,
    distribution_shift, extra_trees_baseline, feature_signal,
    future_cycle_score, learning_curves, permutation_importances,
    repeated_grouped_cv, residual_segments, summarize_learning_curves,
)
from pdm_cmaps.plotting import plot_baseline, plot_learning_curves, plot_regime_scores, plot_sensor_variance
from pdm_cmaps.regimes import RegimeDecoupler
from pdm_cmaps.schema import DEFAULT_HEALTHY_CYCLE_LIMIT, DEFAULT_RANDOM_STATE, DEFAULT_RESULTS_DIR, SENSOR_COLUMNS, SETTING_COLUMNS, SUBSETS
from pdm_cmaps.splitting import split_engine_units
from pdm_cmaps.targets import compute_piecewise_rul, regression_metrics
from pdm_cmaps.virtual_sensor import VirtualSensor


def _per_regime_t50(fd4: pd.DataFrame, output_dir: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit healthy-row T50 virtual sensors by regime on a fixed engine holdout."""
    train_units, test_units = split_engine_units(fd4, test_ratio=0.25, random_state=seed)
    train = fd4.loc[fd4["unit_id"].isin(train_units)].copy()
    test = fd4.loc[fd4["unit_id"].isin(test_units)].copy()
    decoupler = RegimeDecoupler(n_clusters=6, random_state=seed).fit(train)
    train = decoupler.transform(train)
    test = decoupler.transform(test)
    train_healthy = train.loc[train["cycle"] <= DEFAULT_HEALTHY_CYCLE_LIMIT]
    test_healthy = test.loc[test["cycle"] <= DEFAULT_HEALTHY_CYCLE_LIMIT]
    sensors = CMapssLoader.informative_sensors(fd4)
    features = [sensor for sensor in sensors if sensor != "T50"]
    model = VirtualSensor("T50", features, estimator=extra_trees_baseline(seed), random_state=seed).fit(train_healthy)
    prediction = model.predict(test_healthy)
    rows = []
    for regime_id, part in prediction.groupby("regime_id"):
        metric = regression_metrics(part["T50"], part["prediction"])
        rows.append({"Regime_ID": int(regime_id), "Test_rows": len(part), **metric})
    scores = pd.DataFrame(rows)
    scores.to_csv(output_dir / "fd004_healthy_t50_by_regime.csv", index=False)
    variance = RegimeDecoupler.decompose_sensors(train, ["T50", "T24", "T30", "P30", "Nf"])
    variance.to_csv(output_dir / "fd004_sensor_variance.csv", index=False)
    plot_regime_scores(scores, output_dir / "fd004_t50_regime_scores.png")
    plot_sensor_variance(variance, output_dir / "fd004_sensor_variance.png")

    per_engine = []
    for regime_id, part in prediction.groupby("regime_id"):
        residuals = part["T50"].to_numpy() - part["prediction"].to_numpy()
        temp = part[["unit_id", "cycle"]].copy()
        temp["regime_id"] = regime_id
        temp["y"] = part["T50"].to_numpy()
        temp["prediction"] = part["prediction"].to_numpy()
        per_engine.append(temp)
    scored = pd.concat(per_engine, ignore_index=True)
    rng = np.random.default_rng(seed)
    engines = np.sort(scored["unit_id"].unique())
    bootstrap = np.full((2000, len(scores)), np.nan)
    for column, regime_id in enumerate(scores["Regime_ID"]):
        regime = scored.loc[scored["regime_id"] == regime_id]
        by_engine = {engine: part for engine, part in regime.groupby("unit_id")}
        for draw in range(2000):
            sample_ids = rng.choice(engines, size=len(engines), replace=True)
            parts = [by_engine[engine] for engine in sample_ids if engine in by_engine]
            if not parts:
                continue
            sample = pd.concat(parts, ignore_index=True)
            bootstrap[draw, column] = r2_score(sample["y"], sample["prediction"])
    ci_rows = []
    for column, row in scores.reset_index(drop=True).iterrows():
        values = bootstrap[:, column]
        ci_rows.append({"Regime_ID": int(row["Regime_ID"]), "R2": row["R2"], "CI_low_2.5%": np.nanpercentile(values, 2.5), "CI_high_97.5%": np.nanpercentile(values, 97.5)})
    intervals = pd.DataFrame(ci_rows)
    intervals.to_csv(output_dir / "fd004_t50_engine_bootstrap.csv", index=False)
    if 3 in set(intervals["Regime_ID"]):
        regime3_column = list(scores["Regime_ID"]).index(3)
        other_columns = [index for index, regime_id in enumerate(scores["Regime_ID"]) if regime_id != 3]
        difference = bootstrap[:, regime3_column] - np.nanmean(bootstrap[:, other_columns], axis=1)
        pd.DataFrame([{
            "contrast": "Regime 3 minus mean of other regimes",
            "estimate": float(scores.loc[scores["Regime_ID"] == 3, "R2"].iloc[0] - scores.loc[scores["Regime_ID"] != 3, "R2"].mean()),
            "CI_low_2.5%": float(np.nanpercentile(difference, 2.5)),
            "CI_high_97.5%": float(np.nanpercentile(difference, 97.5)),
        }]).to_csv(output_dir / "fd004_t50_regime3_contrast.csv", index=False)
    return scores, variance


def run(output_dir: Path, seed: int, extended: bool = False, nested_cv: bool = False) -> dict[str, pd.DataFrame]:
    """Execute the notebook's cross-subset RUL diagnostics and save tables/figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = CMapssLoader()
    datasets = loader.load_all()
    for subset, frame in datasets.items():
        frame["RUL"] = compute_piecewise_rul(frame)
    features = ["cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]

    baseline, folds, oof = compare_r2_subsets(datasets, features, random_state=seed)
    baseline.to_csv(output_dir / "rul_grouped_cv_baseline.csv", index=False)
    plot_baseline(baseline, output_dir / "rul_grouped_cv_baseline.png")
    for subset, result in folds.items():
        result.to_csv(output_dir / f"{subset.lower()}_baseline_folds.csv", index=False)
        residual_segments(datasets[subset], datasets[subset]["RUL"], oof[subset], subset).to_csv(output_dir / f"{subset.lower()}_residual_bands.csv", index=False)

    split_rows = [{"Subpart": subset, **compare_split_strategies(frame, features, seed)} for subset, frame in datasets.items()]
    split_table = pd.DataFrame(split_rows)
    split_table.to_csv(output_dir / "row_vs_engine_split.csv", index=False)
    future = pd.DataFrame([{"Subpart": subset, **future_cycle_score(frame, features, seed)} for subset, frame in datasets.items()])
    future.to_csv(output_dir / "future_cycle_holdout.csv", index=False)

    profiles = []
    for subset, frame in datasets.items():
        residual = frame["RUL"].to_numpy() - oof[subset]
        profiles.append({
            "Subpart": subset, "RUL_mean": frame["RUL"].mean(), "RUL_std": frame["RUL"].std(),
            "RUL_variance": frame["RUL"].var(), "OOF_RMSE": float(np.sqrt(np.mean(residual ** 2))),
        })
    target_profiles = pd.DataFrame(profiles)
    target_profiles.to_csv(output_dir / "target_profiles.csv", index=False)

    regime_scores, sensor_variance = _per_regime_t50(datasets["FD004"], output_dir, seed)
    results = {
        "baseline": baseline,
        "split_comparison": split_table,
        "future_cycle": future,
        "target_profiles": target_profiles,
        "regime_t50": regime_scores,
        "sensor_variance": sensor_variance,
    }

    if extended:
        capacity = capacity_sweep(datasets, features, seed)
        capacity.to_csv(output_dir / "capacity_sweep.csv", index=False)
        learning_repeats = pd.concat([learning_curves(frame, features, subset, random_state=seed) for subset, frame in datasets.items()], ignore_index=True)
        learning_repeats.to_csv(output_dir / "learning_curve_repeats.csv", index=False)
        learning = summarize_learning_curves(learning_repeats)
        learning.to_csv(output_dir / "learning_curves.csv", index=False)
        plot_learning_curves(learning, output_dir / "learning_curves.png")
        signal = pd.concat([feature_signal(frame, features, subset, random_state=seed) for subset, frame in datasets.items()], ignore_index=True)
        signal.to_csv(output_dir / "feature_signal.csv", index=False)
        importance = pd.concat([permutation_importances(frame, features, subset, seed) for subset, frame in datasets.items()], ignore_index=True)
        importance.to_csv(output_dir / "permutation_importance.csv", index=False)
        repeated = pd.concat([repeated_grouped_cv(frame, features, subset, random_state=seed) for subset, frame in datasets.items()], ignore_index=True)
        repeated.to_csv(output_dir / "repeated_grouped_cv.csv", index=False)
        high, low = str(baseline.iloc[0]["Subpart"]), str(baseline.iloc[-1]["Subpart"])
        shift = distribution_shift(datasets[high], datasets[low], features, high, low, random_state=seed)
        shift.to_csv(output_dir / "feature_distribution_shift.csv", index=False)
        results.update({"capacity": capacity, "learning_curves": learning, "feature_signal": signal, "permutation_importance": importance, "repeated_cv": repeated, "distribution_shift": shift})

    if nested_cv:
        nested = _nested_depth_cv(datasets, features, seed)
        nested.to_csv(output_dir / "nested_depth_cv.csv", index=False)
        results["nested_depth_cv"] = nested

    return results


def _nested_depth_cv(datasets: dict[str, pd.DataFrame], features: list[str], seed: int) -> pd.DataFrame:
    """Select Extra Trees depth inside each outer engine-grouped training fold."""
    rows = []
    depths = (4, 8, 16, None)
    for subset, frame in datasets.items():
        groups = frame["unit_id"].to_numpy()
        x, y = frame[features].to_numpy(dtype=float), frame["RUL"].to_numpy(dtype=float)
        units = np.random.default_rng(seed).permutation(np.unique(groups))
        fold_for = {unit: index % 3 for index, unit in enumerate(units)}
        outer_fold_ids = np.array([fold_for[unit] for unit in groups])
        for outer in range(3):
            train_idx, test_idx = np.flatnonzero(outer_fold_ids != outer), np.flatnonzero(outer_fold_ids == outer)
            train_groups = groups[train_idx]
            inner_units = np.random.default_rng(seed + 1 + outer).permutation(np.unique(train_groups))
            inner_fold_for = {unit: index % 3 for index, unit in enumerate(inner_units)}
            inner_ids = np.array([inner_fold_for[unit] for unit in train_groups])
            scores = {depth: [] for depth in depths}
            for inner in range(3):
                itr = train_idx[inner_ids != inner]
                iva = train_idx[inner_ids == inner]
                for depth in depths:
                    model = extra_trees_baseline(seed + outer + inner, n_estimators=40, max_depth=depth).fit(x[itr], y[itr])
                    scores[depth].append(r2_score(y[iva], model.predict(x[iva])))
            selected = max(depths, key=lambda depth: np.mean(scores[depth]))
            estimator = extra_trees_baseline(seed + outer, n_estimators=40, max_depth=selected).fit(x[train_idx], y[train_idx])
            rows.append({"Subpart": subset, "Outer_fold": outer, "Chosen_depth": "unlimited" if selected is None else selected, "Nested_R2": r2_score(y[test_idx], estimator.predict(x[test_idx]))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR / "r2_investigation")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--extended", action="store_true", help="Run capacity, feature, learning-curve, permutation, repeated-CV diagnostics")
    parser.add_argument("--nested-cv", action="store_true", help="Run nested grouped CV for Extra Trees depth (computationally expensive)")
    args = parser.parse_args()
    results = run(args.output_dir, args.seed, args.extended, args.nested_cv)
    for name, table in results.items():
        print(f"{name}: {len(table)} rows")
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
