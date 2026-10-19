"""Loading and validation for NASA C-MAPSS text files."""

from pathlib import Path

import pandas as pd

from .schema import COLUMNS, DEFAULT_DATA_DIR, SENSOR_COLUMNS, SUBSETS


class CMapssLoader:
    """Load named, validated C-MAPSS training trajectories from a data directory."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)

    def load(self, subset: str) -> pd.DataFrame:
        """Load one training subset and validate its rows and 26-column schema."""
        subset = subset.upper()
        if subset not in SUBSETS:
            raise ValueError(f"subset must be one of {', '.join(SUBSETS)}")
        path = self.data_dir / f"train_{subset}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing C-MAPSS input: {path}")
        frame = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
        self.validate(frame, path)
        return frame.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load FD001 through FD004 using the same validated schema."""
        return {subset: self.load(subset) for subset in SUBSETS}

    @staticmethod
    def validate(frame: pd.DataFrame, source: Path | None = None) -> None:
        """Reject malformed, missing, nonnumeric, or duplicate engine-cycle records."""
        label = f" in {source}" if source is not None else ""
        missing_columns = set(COLUMNS).difference(frame.columns)
        if missing_columns:
            raise ValueError(f"Missing required C-MAPSS columns{label}: {sorted(missing_columns)}")
        if frame.loc[:, COLUMNS].isna().any().any():
            raise ValueError(f"Missing telemetry values{label}")
        if frame.duplicated(["unit_id", "cycle"]).any():
            raise ValueError(f"Duplicate engine-cycle keys{label}")
        if not all(pd.api.types.is_numeric_dtype(frame[column]) for column in COLUMNS):
            raise ValueError(f"Non-numeric telemetry column{label}")
        if (frame["cycle"] < 1).any():
            raise ValueError(f"Cycle values must be positive{label}")

    @staticmethod
    def informative_sensors(frame: pd.DataFrame) -> list[str]:
        """Return sensor channels that vary in the supplied population."""
        return [name for name in SENSOR_COLUMNS if frame[name].nunique() > 1 and frame[name].std() > 1e-6]
