"""Shared C-MAPSS schema and analysis defaults."""

from pathlib import Path

SENSOR_COLUMNS = [
    "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr", "Ps30",
    "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32",
]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
ID_COLUMNS = ["unit_id", "cycle"]
COLUMNS = [*ID_COLUMNS, *SETTING_COLUMNS, *SENSOR_COLUMNS]
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
DEFAULT_RUL_CAP = 125
DEFAULT_HEALTHY_CYCLE_LIMIT = 50
DEFAULT_RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "CMaps"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
