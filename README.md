# PdM-CMaps

Script-based Python version of the C-MAPSS teaching analyses. Shared data/model logic lives in `pdm_cmaps/`; the `scripts/` entry points compose it into the maintenance guide and the R² investigation. No notebook files are required.

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
