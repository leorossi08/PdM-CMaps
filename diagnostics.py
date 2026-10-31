"""Stateless diagnostics for the cross-subset RUL investigation."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from evaluation import grouped_oof
from schema import DEFAULT_RANDOM_STATE
from targets import regression_metrics

SUBSYSTEM_GROUPS = {
    "Fan_Inlet": ["T2", "P2", "Nf", "NRf"],
    "Compressors_LPC_HPC": ["T24", "T30", "P15", "P30", "Nc", "NRc"],
    "Turbines_Combustor": ["T50", "epr", "Ps30", "BPR", "farB", "htBleed", "W31", "W32"],
    "Controls_Demands": ["Nf_dmd", "PCNfR_dmd", "phi"],
    "Flight_Settings": ["setting_1", "setting_2", "setting_3"],
}


def extra_trees_baseline(random_state: int = DEFAULT_RANDOM_STATE, **kwargs) -> ExtraTreesRegressor:
    """Return the common Extra Trees baseline used across FD001-FD004."""
    params = {"n_estimators": 60, "max_depth": 12, "min_samples_leaf": 2, "max_features": 1.0, "n_jobs": -1}
    params.update(kwargs)
    return ExtraTreesRegressor(random_state=random_state, **params)


def compare_rul_subsets(datasets: dict[str, pd.DataFrame], features: list[str], n_splits: int = 5, random_state: int = DEFAULT_RANDOM_STATE):
    """Compute comparable grouped-fold mean and pooled OOF RUL results."""
    fold_results, predictions = {}, {}
    rows = []
    for subset, frame in datasets.items():
        folds, oof = grouped_oof(
            frame, features,
            estimator=extra_trees_baseline(random_state=random_state),
            n_splits=n_splits, target_col="RUL",
        )
        fold_results[subset], predictions[subset] = folds, oof
        pooled = regression_metrics(frame["RUL"], oof)
        rows.append({
            "Subpart": subset,
            "R2_fold_mean": folds["R2"].mean(),
            "R2_fold_std": folds["R2"].std(ddof=1),
            "R2_pooled_OOF": pooled["R2"],
            "RMSE_fold_mean": folds["RMSE"].mean(),
            "MAE_fold_mean": folds["MAE"].mean(),
            "Rows": len(frame),
            "Engines": frame["unit_id"].nunique(),
        })
    summary = pd.DataFrame(rows).sort_values("R2_pooled_OOF", ascending=False).reset_index(drop=True)
    return summary, fold_results, predictions


def compare_split_strategies(frame: pd.DataFrame, features: list[str], n_repeats: int = 10, random_state: int = DEFAULT_RANDOM_STATE) -> dict:
    """Compare repeated random-row splits with grouped engine holdouts."""
    x, y, groups = frame[features].to_numpy(), frame["RUL"].to_numpy(), frame["unit_id"].to_numpy()
    row_scores, group_scores = [], []
    for rep in range(n_repeats):
        row_train, row_test = train_test_split(np.arange(len(frame)), test_size=0.25, random_state=random_state + rep)
        group_train, group_test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state + rep).split(x, y, groups))
        row_model = extra_trees_baseline(random_state + rep).fit(x[row_train], y[row_train])
        group_model = extra_trees_baseline(random_state + rep).fit(x[group_train], y[group_train])
        row_scores.append(float(r2_score(y[row_test], row_model.predict(x[row_test]))))
        group_scores.append(float(r2_score(y[group_test], group_model.predict(x[group_test]))))

    row_r2 = float(np.mean(row_scores))
    group_r2 = float(np.mean(group_scores))
    return {
        "Row_random_R2": row_r2,
        "Engine_holdout_R2": group_r2,
        "Row_split_optimism_delta": row_r2 - group_r2,
        "Row_random_R2_std": float(np.std(row_scores, ddof=1)) if n_repeats > 1 else 0.0,
        "Engine_holdout_R2_std": float(np.std(group_scores, ddof=1)) if n_repeats > 1 else 0.0,
        "Engines_shared_in_row_split": len(set(groups[row_train]).intersection(groups[row_test])),
        "Engines_shared_in_group_split": len(set(groups[group_train]).intersection(groups[group_test])),
    }


def future_cycle_score(frame: pd.DataFrame, features: list[str], random_state: int = DEFAULT_RANDOM_STATE) -> dict:
    """Train on each engine's first 75% of cycles and score its later cycles."""
    cutoff = frame.groupby("unit_id")["cycle"].transform(lambda cycles: np.floor(cycles.max() * 0.75))
    train_mask, future_mask = frame["cycle"] <= cutoff, frame["cycle"] > cutoff
    model = extra_trees_baseline(random_state).fit(frame.loc[train_mask, features], frame.loc[train_mask, "RUL"])
    prediction = model.predict(frame.loc[future_mask, features])
    return {
        "Temporal_future_R2": float(r2_score(frame.loc[future_mask, "RUL"], prediction)),
        "Train_rows": int(train_mask.sum()),
        "Future_rows": int(future_mask.sum()),
        "Engines_shared_by_design": int(frame.loc[train_mask, "unit_id"].nunique()),
    }


def capacity_sweep(datasets: dict[str, pd.DataFrame], features: list[str], random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Compare Extra Trees depths and Ridge strengths on the same grouped folds."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = []
    for subset, frame in datasets.items():
        for depth in (4, 8, 16, None):
            estimator = extra_trees_baseline(random_state, n_estimators=40, max_depth=depth)
            folds, _ = grouped_oof(frame, features, estimator=estimator)
            rows.append({"Subpart": subset, "Family": "Extra Trees", "Configuration": f"max_depth={depth or 'unlimited'}", "Mean_fold_R2": folds["R2"].mean(), "Std_fold_R2": folds["R2"].std(ddof=1)})
        for alpha in (0.1, 1.0, 10.0, 100.0):
            estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            folds, _ = grouped_oof(frame, features, estimator=estimator)
            rows.append({"Subpart": subset, "Family": "Ridge", "Configuration": f"alpha={alpha:g}", "Mean_fold_R2": folds["R2"].mean(), "Std_fold_R2": folds["R2"].std(ddof=1)})
    return pd.DataFrame(rows)


def learning_curves(frame: pd.DataFrame, features: list[str], subset: str, repeats: int = 10, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Measure holdout R² as the number of training engines increases."""
    groups = frame["unit_id"].to_numpy()
    x, y = frame[features].to_numpy(), frame["RUL"].to_numpy()
    rows = []
    for repeat in range(repeats):
        train_idx, valid_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state + repeat).split(x, y, groups))
        train_units = np.unique(groups[train_idx])
        shuffled = np.random.default_rng(random_state + repeat).permutation(train_units)
        for fraction in (0.25, 0.5, 0.75, 1.0):
            count = max(2, int(np.ceil(len(shuffled) * fraction)))
            selected = shuffled[:count]
            partial = train_idx[np.isin(groups[train_idx], selected)]
            model = extra_trees_baseline(random_state, n_estimators=40).fit(x[partial], y[partial])
            rows.append({"Subpart": subset, "Repeat": repeat, "Train_engine_fraction": fraction, "Train_engines": count, "Validation_R2": r2_score(y[valid_idx], model.predict(x[valid_idx]))})
    return pd.DataFrame(rows)


def summarize_learning_curves(repeats: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated learning-curve scores into mean and standard deviation."""
    return repeats.groupby(["Subpart", "Train_engine_fraction"], as_index=False).agg(
        Mean_R2=("Validation_R2", "mean"), Std_R2=("Validation_R2", "std"),
    )


def residual_segments(frame: pd.DataFrame, truth, prediction, subset: str) -> pd.DataFrame:
    """Summarize OOF residual direction and RMSE over quantile-defined RUL bands."""
    residuals = pd.DataFrame({
        "unit_id": frame["unit_id"].to_numpy(), "cycle": frame["cycle"].to_numpy(),
        "RUL": frame["RUL"].to_numpy(), "prediction": np.asarray(prediction),
    })
    residuals["residual"] = np.asarray(truth) - np.asarray(prediction)
    residuals["RUL_band"] = pd.qcut(residuals["RUL"], q=5, duplicates="drop")
    result = residuals.groupby("RUL_band", observed=True).agg(
        rows=("residual", "size"), mean_error=("residual", "mean"),
        mae=("residual", lambda values: np.abs(values).mean()),
        rmse=("residual", lambda values: np.sqrt(np.mean(values ** 2))),
    ).reset_index()
    result.insert(0, "Subpart", subset)
    return result


def feature_signal(frame: pd.DataFrame, features: list[str], subset: str, sample_size: int = 5000, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Return sampled univariate absolute correlation and mutual information with RUL."""
    sample = frame.sample(n=min(sample_size, len(frame)), random_state=random_state)
    x = sample[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mutual_info = mutual_info_regression(x, sample["RUL"], random_state=random_state)
    correlation = pd.Series(np.nan, index=features, dtype=float)
    varying_features = [feature for feature in features if x[feature].nunique() > 1]
    if varying_features and sample["RUL"].nunique() > 1:
        correlation.loc[varying_features] = x[varying_features].corrwith(sample["RUL"]).abs()
    return pd.DataFrame({
        "Subpart": subset,
        "Feature": features,
        "Abs_correlation": [float(correlation[feature]) for feature in features],
        "Mutual_information": mutual_info,
    })


def permutation_importances(frame: pd.DataFrame, features: list[str], subset: str, n_repeats: int = 10, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Compute held-out engine permutation importance with subsystem mapping."""
    x, y, groups = frame[features].to_numpy(), frame["RUL"].to_numpy(), frame["unit_id"].to_numpy()
    fit_idx, score_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state).split(x, y, groups))
    model = extra_trees_baseline(random_state, n_estimators=40).fit(x[fit_idx], y[fit_idx])
    result = permutation_importance(model, x[score_idx], y[score_idx], scoring="r2", n_repeats=n_repeats, random_state=random_state, n_jobs=-1)

    # Map each feature to its physical subsystem
    feature_to_subsystem = {}
    for sub_name, sub_feats in SUBSYSTEM_GROUPS.items():
        for f in sub_feats:
            feature_to_subsystem[f] = sub_name

    subsystems = [feature_to_subsystem.get(f, "Other") for f in features]
    return pd.DataFrame({
        "Subpart": subset,
        "Feature": features,
        "Subsystem": subsystems,
        "R2_drop_mean": result.importances_mean,
        "R2_drop_std": result.importances_std,
    })


def distribution_shift(high_frame: pd.DataFrame, low_frame: pd.DataFrame, features: list[str], high_name: str, low_name: str, sample_size: int = 20000, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Compare feature distributions with KS effect sizes and Wasserstein-1 distances."""
    rows = []
    for feature in [*features, "RUL"]:
        high = high_frame[feature].dropna()
        low = low_frame[feature].dropna()
        high = high.sample(n=min(sample_size, len(high)), random_state=random_state)
        low = low.sample(n=min(sample_size, len(low)), random_state=random_state)
        result = ks_2samp(high, low)

        # Standardized Wasserstein-1 distance
        pooled_vals = np.concatenate([high.to_numpy(), low.to_numpy()])
        p_mean = np.mean(pooled_vals)
        p_std = np.std(pooled_vals) if np.std(pooled_vals) > 1e-8 else 1.0
        w_dist = float(wasserstein_distance((high.to_numpy() - p_mean) / p_std, (low.to_numpy() - p_mean) / p_std))

        rows.append({
            "Feature": feature,
            "KS_statistic": float(result.statistic),
            "Wasserstein_distance": w_dist,
            "p_value": float(result.pvalue),
            "High_subset_median": float(high.median()),
            "Low_subset_median": float(low.median()),
            "High_subset": high_name,
            "Low_subset": low_name,
        })
    return pd.DataFrame(rows).sort_values("KS_statistic", ascending=False).reset_index(drop=True)


def repeated_grouped_cv(frame: pd.DataFrame, features: list[str], subset: str, repeats: int = 5, n_splits: int = 5, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Repeat pooled grouped OOF scoring with shuffled engine-to-fold assignment."""
    groups = frame["unit_id"].to_numpy()
    x, y = frame[features].to_numpy(dtype=float), frame["RUL"].to_numpy(dtype=float)
    rows = []
    for repeat in range(repeats):
        units = np.random.default_rng(random_state + repeat).permutation(np.unique(groups))
        fold_for = {unit: i % n_splits for i, unit in enumerate(units)}
        fold_ids = np.array([fold_for[group] for group in groups])
        predictions = np.full(len(frame), np.nan)
        for fold in range(n_splits):
            valid = np.flatnonzero(fold_ids == fold)
            train = np.flatnonzero(fold_ids != fold)
            model = extra_trees_baseline(random_state + repeat * 10 + fold).fit(x[train], y[train])
            predictions[valid] = model.predict(x[valid])
        rows.append({"Subpart": subset, "Repeat": repeat, "R2_pooled_OOF": float(r2_score(y, predictions))})
    return pd.DataFrame(rows)
