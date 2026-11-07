import pandas as pd

from .. import config
from .common import COLUMNS, download, drop_duplicates_and_conflicts

CFG = config.SOURCES["etda"]


def load() -> tuple[pd.DataFrame, dict]:
    download(CFG["url"], CFG["file"])
    df = pd.read_csv(CFG["file"], dtype=str, encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]
    log = {"rows_raw": len(df), "null_text_dropped": int(df["tweet"].isna().sum())}
    df = df.dropna(subset=["tweet"])

    df = pd.DataFrame({
        "source_id": df["id"].str.strip(),
        "text_raw": df["tweet"].str.strip(),
        "label_raw": df["label"].str.strip().str.lower(),
        "source": "etda",
        "register": CFG["register"],
        "variety": CFG["variety"],
    })[COLUMNS]
    df = drop_duplicates_and_conflicts(df, log)
    log["rows_out"] = len(df)
    return df, log
