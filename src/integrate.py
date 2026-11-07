"""Build the harmonized multi-source Arabic emotion corpus."""
import json

import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords

from . import arabic, config
from .sources import etda, poems, semeval


def _stopwords() -> set:
    try:
        raw = nltk_stopwords.words("arabic")
    except LookupError:
        import nltk

        nltk.download("stopwords", quiet=True)
        raw = nltk_stopwords.words("arabic")
    return arabic.build_stopwords(raw, config.NORMALIZE_TA_MARBUTA)


def harmonize(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    df = df.copy()
    df["label"] = [config.LABEL_MAP[s].get(l) for s, l in zip(df["source"], df["label_raw"])]
    dropped = df[df["label"].isna()]
    log["unmapped_labels_dropped"] = {
        f"{s}:{l}": int(n) for (s, l), n in dropped.groupby(["source", "label_raw"]).size().items()
    }
    return df.dropna(subset=["label"])


def normalize(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    df = df.copy()
    df["emojis"] = df["text_raw"].map(arabic.extract_emojis)
    df["diacritic_ratio"] = df["text_raw"].map(arabic.diacritic_ratio).round(4)
    df["text_norm"] = df["text_raw"].map(lambda t: arabic.normalize(t, config.NORMALIZE_TA_MARBUTA))
    empty = df["text_norm"].eq("")
    log["empty_after_normalization"] = df.loc[empty, "source"].value_counts().to_dict()
    df = df.loc[~empty]

    # Texts identical after normalization, within or across sources.
    labels_per_text = df.groupby("text_norm")["label"].nunique()
    conflicting = set(labels_per_text[labels_per_text > 1].index)
    in_conflict = df["text_norm"].isin(conflicting)
    dup = df["text_norm"].duplicated(keep="first") & ~in_conflict
    log["normalized_duplicates"] = {
        "dropped_same_label": int(dup.sum()),
        "dropped_conflicting_label": int(in_conflict.sum()),
        "cross_source_groups": int(df[df["text_norm"].duplicated(keep=False)].groupby("text_norm")["source"].nunique().gt(1).sum()),
    }
    return df.loc[~dup & ~in_conflict].copy()


def cap_poems(df: pd.DataFrame, log: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep poems at most MAX_POEM_SHARE of each label; return (corpus, reserve)."""
    share = config.MAX_POEM_SHARE
    keep, reserve = [], []
    for label, group in df.groupby("label"):
        is_poem = group["source"].eq("poems")
        others = int((~is_poem).sum())
        limit = int(others * share / (1 - share))
        poem_rows = group[is_poem].sample(frac=1, random_state=config.SEED)
        keep += [group[~is_poem], poem_rows.iloc[:limit]]
        reserve.append(poem_rows.iloc[limit:])
    log["poem_cap"] = {"max_share": share, "poems_moved_to_reserve": int(sum(len(r) for r in reserve))}
    return pd.concat(keep), pd.concat(reserve)


def add_text_features(df: pd.DataFrame, stopwords: set) -> pd.DataFrame:
    df = df.copy()
    df["text_nostop"] = df["text_norm"].map(lambda t: arabic.remove_stopwords(t, stopwords))
    df["n_words"] = df["text_norm"].str.split().str.len()
    df["n_words_nostop"] = df["text_nostop"].str.split().str.len()
    return df


COLUMNS = ["uid", "source", "source_id", "register", "variety", "label", "label_raw",
           "text_raw", "text_norm", "text_nostop", "emojis", "n_words", "n_words_nostop",
           "diacritic_ratio"]


def run() -> pd.DataFrame:
    log: dict = {"sources": {}}
    p, log["sources"]["poems"] = poems.load()
    e, log["sources"]["etda"] = etda.load()
    s, s_mixed, log["sources"]["semeval"] = semeval.load()

    df = pd.concat([p, e, s], ignore_index=True)
    funnel = {"loaded_after_source_cleaning": len(df)}

    df = harmonize(df, log)
    funnel["after_label_harmonization"] = len(df)
    df = normalize(df, log)
    funnel["after_normalized_dedup"] = len(df)
    df, reserve = cap_poems(df, log)
    funnel["after_poem_cap"] = len(df)

    stop = _stopwords()
    log["stopwords"] = {"source": "nltk arabic, normalized", "size": len(stop),
                        "kept_negations": sorted(arabic.NEGATIONS)}
    df = add_text_features(df, stop).sample(frac=1, random_state=config.SEED)
    df["uid"] = df["source"] + ":" + df["source_id"].astype(str)
    reserve = add_text_features(reserve, stop)
    reserve["uid"] = reserve["source"] + ":" + reserve["source_id"].astype(str)

    log["funnel"] = funnel
    log["label_by_source"] = pd.crosstab(df["label"], df["source"]).to_dict(orient="index")
    log["semeval_mixed_set_aside"] = len(s_mixed)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(config.CORPUS_FILE, index=False, encoding="utf-8")
    reserve[COLUMNS].to_csv(config.POEMS_RESERVE_FILE, index=False, encoding="utf-8")
    s_mixed.to_csv(config.SEMEVAL_MIXED_FILE, index=False, encoding="utf-8")
    config.QUALITY_REPORT.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return df


if __name__ == "__main__":
    run()
    report = json.loads(config.QUALITY_REPORT.read_text(encoding="utf-8"))
    print(json.dumps({k: report[k] for k in ("funnel", "label_by_source")}, indent=2))
