import requests

# Some hosts (saifmohammad.com) reject the default python-requests user agent with 406.
HEADERS = {"User-Agent": "Mozilla/5.0 (dataset download for academic project)"}

COLUMNS = ["source_id", "text_raw", "label_raw", "source", "register", "variety"]


def download(url: str, path, force: bool = False) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, headers=HEADERS, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)


def drop_duplicates_and_conflicts(df, log: dict):
    """Drop exact text duplicates; drop every copy of a text that carries conflicting labels."""
    labels_per_text = df.groupby("text_raw")["label_raw"].nunique()
    conflicting = set(labels_per_text[labels_per_text > 1].index)
    in_conflict = df["text_raw"].isin(conflicting)
    dup = df["text_raw"].duplicated(keep="first") & ~in_conflict
    log["duplicates_dropped"] = int(dup.sum())
    log["conflicting_label_rows_dropped"] = int(in_conflict.sum())
    return df.loc[~dup & ~in_conflict].copy()
