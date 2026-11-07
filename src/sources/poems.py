import io

import pandas as pd

from .. import config
from .common import COLUMNS, download, drop_duplicates_and_conflicts

CFG = config.SOURCES["poems"]


def load() -> tuple[pd.DataFrame, dict]:
    download(CFG["url"], CFG["file"])
    raw = CFG["file"].read_bytes()
    lines = raw.split(b"\n")
    text = raw.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), dtype=str)
    log = {
        "rows_raw": len(df),
        "invalid_utf8_sequences": text.count("\ufffd"),
        "lines_at_byte_cap": [i for i, l in enumerate(lines) if len(l) >= CFG["line_byte_cap"]],
    }

    df["emotion"] = df["emotion"].str.strip().str.lower()
    verses = pd.to_numeric(df["verses"], errors="coerce")
    # Poems longer than the source's line cap spill into an extra row; the label is lost.
    broken = ~df["emotion"].isin(config.LABEL_MAP["poems"]) | verses.isna()
    log["broken_rows_dropped"] = int(broken.sum())
    df = df.loc[~broken]

    df = pd.DataFrame({
        "source_id": df["poem_id"].str.strip(),
        "text_raw": df["poem"],
        "label_raw": df["emotion"],
        "source": "poems",
        "register": CFG["register"],
        "variety": CFG["variety"],
    })[COLUMNS]
    df = drop_duplicates_and_conflicts(df, log)
    log["rows_out"] = len(df)
    return df, log
