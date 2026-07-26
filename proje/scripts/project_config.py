from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "data.csv"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_READY_DIR = PROJECT_ROOT / "data" / "model_ready"
REPORT_CSV_DIR = PROJECT_ROOT / "reports" / "csv"
REPORT_MD_DIR = PROJECT_ROOT / "reports" / "markdown"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
BUNDLE_DIR = MODELS_DIR / "demand_forecasting_bundle"
APP_DIR = PROJECT_ROOT / "app"
LOGS_DIR = PROJECT_ROOT / "logs"

RANDOM_STATE = 42
MAX_FORECAST_LEAD_DAYS = 180
ACTIVE_TAIL_DAYS = 90
MIN_HISTORY_DAYS = 28
LEAD_DAY_GRID = tuple(
    [1, 2, 3, 4, 5, 6, 7, 14, 21, 30, 45, 60, 90, 120, 150, 180]
)


def ensure_project_dirs() -> None:
    for directory in (
        PROCESSED_DIR,
        REFERENCE_DIR,
        MODEL_READY_DIR,
        REPORT_CSV_DIR,
        REPORT_MD_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        BUNDLE_DIR,
        APP_DIR,
        LOGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
