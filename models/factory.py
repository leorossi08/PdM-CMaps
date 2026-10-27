"""Small sklearn model factory shared by virtual sensing and RUL experiments."""

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR


def get_model(name: str, random_state: int = 42, **kwargs):
    """Construct a fresh estimator by canonical model name."""
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key in {"extra_trees", "et"}:
        return ExtraTreesRegressor(n_estimators=kwargs.pop("n_estimators", 60), max_depth=kwargs.pop("max_depth", 12), min_samples_leaf=kwargs.pop("min_samples_leaf", 2), random_state=random_state, n_jobs=kwargs.pop("n_jobs", -1), **kwargs)
    if key in {"random_forest", "rf"}:
        return RandomForestRegressor(n_estimators=kwargs.pop("n_estimators", 60), max_depth=kwargs.pop("max_depth", 12), min_samples_leaf=kwargs.pop("min_samples_leaf", 2), random_state=random_state, n_jobs=kwargs.pop("n_jobs", -1), **kwargs)
    if key in {"hist_gradient_boosting", "gradient_boosting", "gb"}:
        return HistGradientBoostingRegressor(max_iter=kwargs.pop("max_iter", 80), max_depth=kwargs.pop("max_depth", 6), learning_rate=kwargs.pop("learning_rate", 0.08), random_state=random_state, **kwargs)
    if key in {"xgboost", "xgb"}:
        from xgboost import XGBRegressor
        n_estimators = kwargs.pop("n_estimators", kwargs.pop("iterations", 80))
        max_depth = kwargs.pop("max_depth", kwargs.pop("depth", 6))
        return XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=kwargs.pop("learning_rate", 0.08), random_state=random_state, n_jobs=kwargs.pop("n_jobs", -1), **kwargs)
    if key in {"catboost", "cb"}:
        from catboost import CatBoostRegressor
        iterations = kwargs.pop("iterations", kwargs.pop("n_estimators", 100))
        depth = kwargs.pop("depth", kwargs.pop("max_depth", 6))
        return CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=kwargs.pop("learning_rate", 0.08), random_seed=random_state, verbose=kwargs.pop("verbose", 0), thread_count=kwargs.pop("thread_count", kwargs.pop("n_jobs", -1)), **kwargs)
    if key in {"lightgbm", "lgb", "lgbm"}:
        from lightgbm import LGBMRegressor
        n_estimators = kwargs.pop("n_estimators", kwargs.pop("iterations", 80))
        max_depth = kwargs.pop("max_depth", kwargs.pop("depth", 6))
        return LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=kwargs.pop("learning_rate", 0.08), random_state=random_state, n_jobs=kwargs.pop("n_jobs", -1), verbose=kwargs.pop("verbose", -1), **kwargs)
    if key == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=kwargs.pop("alpha", 10.0), **kwargs))
    if key in {"elasticnet", "elastic_net"}:
        return make_pipeline(StandardScaler(), ElasticNet(alpha=kwargs.pop("alpha", 0.5), l1_ratio=kwargs.pop("l1_ratio", 0.5), max_iter=3000, **kwargs))
    if key in {"bayesian_ridge", "bayes"}:
        return make_pipeline(StandardScaler(), BayesianRidge(**kwargs))
    if key in {"linear_svr", "svr"}:
        return make_pipeline(StandardScaler(), LinearSVR(random_state=random_state, max_iter=5000, **kwargs))
    if key == "rbf_svr":
        return make_pipeline(StandardScaler(), SVR(kernel="rbf", **kwargs))
    if key in {"knn", "knn_regressor"}:
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=kwargs.pop("n_neighbors", 15), weights=kwargs.pop("weights", "distance"), **kwargs))
    if key in {"mlp", "neural_net"}:
        model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=kwargs.pop("hidden_layer_sizes", (64, 32)), max_iter=kwargs.pop("max_iter", 180), random_state=random_state, early_stopping=True, **kwargs))
        return TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
    raise ValueError(f"Unknown model {name!r}")


def get_model_pool(random_state: int = 42) -> dict:
    """Return the core model families used for the RUL benchmark."""
    return {
        "Ridge": get_model("ridge", random_state, alpha=10),
        "Random Forest": get_model("rf", random_state, n_estimators=60, max_depth=10),
        "Extra Trees": get_model("extra_trees", random_state, n_estimators=60, max_depth=10),
        "HistGradientBoosting": get_model("gradient_boosting", random_state, max_iter=80, max_depth=6),
        "XGBoost": get_model("xgboost", random_state, n_estimators=80, max_depth=6),
        "CatBoost": get_model("catboost", random_state, iterations=100, depth=6),
        "LightGBM": get_model("lightgbm", random_state, n_estimators=80, max_depth=6),
        "MLP": get_model("mlp", random_state, hidden_layer_sizes=(64, 32), max_iter=180),
    }
