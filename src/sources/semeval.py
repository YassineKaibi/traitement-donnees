import zipfile

import pandas as pd

from .. import config
from .common import COLUMNS, download, drop_duplicates_and_conflicts

CFG = config.SOURCES["semeval"]
NON_EMOTION_COLUMNS = {"ID", "Tweet"}


def _read_all() -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(CFG["file"]) as archive:
        for member in CFG["members"]:
            with archive.open(member) as fh:
                frame = pd.read_csv(fh, sep="\t", encoding="utf-8", encoding_errors="replace")
                frame["split"] = member.rsplit("-", 1)[-1].removesuffix(".txt")
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (single-label rows, mixed rows kept for qualitative testing, log)."""
    download(CFG["url"], CFG["file"])
    df = _read_all()
    emotions = [c for c in df.columns if c not in NON_EMOTION_COLUMNS | {"split"}]
    df[emotions] = df[emotions].astype(int)
    targets = [k for k in config.LABEL_MAP["semeval"] if k in emotions]

    n_target = df[targets].sum(axis=1)
    n_any = df[emotions].sum(axis=1)
    neutral = n_any.eq(0)
    single = n_target.eq(1)
    mixed = n_target.ge(2)
    log = {
        "rows_raw": len(df),
        "single_target_label": int(single.sum()),
        "neutral": int(neutral.sum()),
        "mixed_target_labels_set_aside": int(mixed.sum()),
        "only_dropped_emotions": int((n_target.eq(0) & ~neutral).sum()),
    }

    label = pd.Series(pd.NA, index=df.index, dtype="object")
    label[single] = df.loc[single, targets].idxmax(axis=1)
    label[neutral] = "none"

    def frame(mask, labels):
        return pd.DataFrame({
            "source_id": df.loc[mask, "ID"],
            "text_raw": df.loc[mask, "Tweet"].str.strip(),
            "label_raw": labels,
            "source": "semeval",
            "register": CFG["register"],
            "variety": CFG["variety"],
        })[COLUMNS]

    kept = frame(single | neutral, label[single | neutral])
    kept = drop_duplicates_and_conflicts(kept, log)
    mixed_rows = frame(mixed, df.loc[mixed, emotions].apply(
        lambda r: "+".join(e for e in emotions if r[e]), axis=1))
    log["rows_out"] = len(kept)
    return kept, mixed_rows, log
