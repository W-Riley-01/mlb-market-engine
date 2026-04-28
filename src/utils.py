from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"


def load_raw(name: str, **kwargs) -> pd.DataFrame:
    path = DATA_RAW / name
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    return pd.read_csv(path, **kwargs)


def save_processed(df: pd.DataFrame, name: str, index: bool = False, **kwargs) -> Path:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / name
    df.to_csv(path, index=index, **kwargs)
    return path

def load_processed(name: str, **kwargs) -> pd.DataFrame:
    path = DATA_PROCESSED / name
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_csv(path, **kwargs)
