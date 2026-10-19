# PdM-CMaps

Script-based Python version of the C-MAPSS teaching analyses. Shared data and model logic lives at the repository root; the `scripts/` entry points compose it into the maintenance guide and the R² investigation. No notebook files are required.

## Data

The loader expects the NASA text files in `data/CMaps/`, including `train_FD001.txt` through `train_FD004.txt`. It assigns the named 26-column schema (`unit_id`, `cycle`, three settings, and 21 physical sensor names), validates numeric/non-null values and unique engine-cycle keys, and sorts by engine and cycle. The analyses use training trajectories only and create their own engine-level evaluation splits.

## Setup and run

From this directory:

```bash
python -m pip install -r requirements.txt
python -m scripts.run_maintenance_guide --subset FD004
python -m scripts.run_r2_investigation
```

Run the additional sequence-model comparison with:

```bash
python -m scripts.run_maintenance_guide --subset FD004 --include-lstm
```

The R² investigation's capacity sweep, feature diagnostics, learning curves, permutation importance, and repeated grouped CV are optional because they require many model fits:

```bash
python -m scripts.run_r2_investigation --extended
```

Nested grouped CV adds another expensive depth-selection analysis:

```bash
python -m scripts.run_r2_investigation --nested-cv
```

Each script accepts `--output-dir` and `--seed`. Generated CSV tables and PNG figures are written beneath `results/`; existing result files with the same names are replaced on rerun.

Each run organizes its output directory into `tables/` and `figures/`. The maintenance guide includes a held-out failure-detection chart, actual-versus-predicted RUL for the best holdout model, and prediction metrics/plots for all 21 sensor channels. Individual sensor plots color parity points by operating regime and compare actual/predicted cycle-level distributions by held-out engine; they are saved under `figures/sensor_channels/`. Additional regime-specific sensor timelines use 50-cycle windows, with one figure per regime/window under `figures/sensor_cycle_windows/` and matching R²/RMSE/MAE values in `tables/sensor_cycle_window_metrics.csv`. The R² investigation includes grouped out-of-fold RUL parity plots and prediction rows for FD001 through FD004.

## Package responsibilities

- `CMapssLoader`: named schema, input loading, and validation.
- `RegimeDecoupler`: training-fitted operating-condition clusters and variance decomposition.
- `VirtualSensor`: healthy-window, per-regime sensor regressors and calibrated residual scoring.
- `StatisticalAnomalyDetector`: healthy residual calibration and persistent EWMA alarms.
- `RULBenchmark`: holdout and engine-grouped model comparisons.
- `diagnostics.py`: stateless cross-subset R², residual, capacity, split, feature, and learning-curve diagnostics.
- `models/lstm.py`: engine-local sequence building and optional PyTorch LSTM.

RUL and T50 virtual-sensor scores describe different targets and evaluation populations; do not compare their raw R² values as though they were the same task. All results are exploratory analyses on simulated data, not validation for operational aircraft maintenance.

## Tests

From this directory, run:

```bash
pytest
```
