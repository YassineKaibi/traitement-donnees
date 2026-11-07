"""Exploratory analysis of the harmonized corpus: figures and summary statistics."""
import json
from collections import Counter

import arabic_reshaper
import matplotlib
import numpy as np
import pandas as pd
from bidi.algorithm import get_display

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

SOURCE_COLORS = {"poems": "#8a5a44", "etda": "#3b6ea8", "semeval": "#e0a526"}
SOURCE_FR = {"poems": "Poèmes (classique)", "etda": "ETDA (tweets égyptiens)", "semeval": "SemEval (tweets)"}
LABEL_COLORS = {"joy": "#e0a526", "sadness": "#3b6ea8", "love": "#c4553a", "anger": "#8b1e1e",
                "fear": "#5b4a8a", "surprise": "#2a9d8f", "neutral": "#7a7a7a"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


def _ar(text: str) -> str:
    return get_display(arabic_reshaper.reshape(text))


def _save(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)


def label_by_source(df: pd.DataFrame) -> None:
    table = pd.crosstab(df["label"], df["source"]).reindex(config.LABELS)
    table = table[[s for s in SOURCE_COLORS if s in table.columns]]
    order = table.sum(axis=1).sort_values(ascending=False).index
    table = table.loc[order]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    left = np.zeros(len(table))
    names = [config.LABELS_FR[l] for l in table.index]
    for source in table.columns:
        ax.barh(names, table[source], left=left, color=SOURCE_COLORS[source], label=SOURCE_FR[source])
        left += table[source].to_numpy()
    for y, total in enumerate(left):
        ax.text(total, y, f" {int(total)}", va="center")
    ax.invert_yaxis()
    ax.set_xlim(0, left.max() * 1.15)
    ax.set_xlabel("Nombre de textes")
    ax.set_title(f"Distribution des émotions par source ({len(df)} textes)")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "fig01_label_by_source.png")


def source_share_heatmap(df: pd.DataFrame) -> None:
    share = pd.crosstab(df["label"], df["source"], normalize="index").reindex(config.LABELS)
    share = share[[s for s in SOURCE_COLORS if s in share.columns]]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(share.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(share.columns)), [SOURCE_FR[s].split(" (")[0] for s in share.columns], fontsize=10)
    ax.set_yticks(range(len(share.index)), [config.LABELS_FR[l] for l in share.index])
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share.iloc[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", color="white" if v > 0.6 else "black")
    ax.set_title("Part de chaque source dans chaque émotion")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, "fig02_source_share_heatmap.png")


def length_by_source(df: pd.DataFrame) -> None:
    sources = [s for s in SOURCE_COLORS if s in df["source"].unique()]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    box = ax.boxplot([df.loc[df["source"] == s, "n_words"] for s in sources], patch_artist=True,
                     showfliers=False, tick_labels=[SOURCE_FR[s].split(" (")[0] for s in sources])
    for patch, s in zip(box["boxes"], sources):
        patch.set_facecolor(SOURCE_COLORS[s])
        patch.set_alpha(0.8)
    ax.set_yscale("log")
    ax.set_ylabel("Mots par texte (échelle log)")
    ax.set_title("Longueur des textes par source")
    _save(fig, "fig03_length_by_source.png")


def cleaning_summary(report: dict) -> None:
    src = report["sources"]
    norm = report["normalized_duplicates"]
    rows = [
        ("Poèmes : lignes cassées", src["poems"]["broken_rows_dropped"]),
        ("Doublons exacts (toutes sources)", sum(s["duplicates_dropped"] for s in src.values())),
        ("Étiquettes contradictoires", sum(s["conflicting_label_rows_dropped"] for s in src.values())
         + norm["dropped_conflicting_label"]),
        ("Doublons après normalisation", norm["dropped_same_label"]),
        ("ETDA : classe sympathy", report["unmapped_labels_dropped"].get("etda:sympathy", 0)),
        ("SemEval : multi-émotions (mis de côté)", src["semeval"]["mixed_target_labels_set_aside"]),
        ("SemEval : émotions hors cible", src["semeval"]["only_dropped_emotions"]),
        ("Poèmes : plafond 50 % (réserve)", report["poem_cap"]["poems_moved_to_reserve"]),
    ]
    labels, values = zip(*rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels[::-1], values[::-1], color="#555f6b")
    for bar, n in zip(bars, values[::-1]):
        ax.text(n, bar.get_y() + bar.get_height() / 2, f" {n}", va="center")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(0, max(values) * 3)
    ax.set_xlabel("Lignes retirées (échelle log)")
    f = report["funnel"]
    total_raw = sum(s["rows_raw"] for s in src.values())
    ax.set_title(f"Préparation : {total_raw} lignes brutes, {f['after_poem_cap']} dans le corpus")
    _save(fig, "fig04_cleaning_summary.png")


def distinctive_words(df: pd.DataFrame, top: int = 10, min_count: int = 30) -> dict:
    # Smoothed log-ratio of a word's document frequency inside a label vs in all other labels.
    # Document frequency, not token counts: one poem repeating a refrain 40 times would dominate.
    texts = df["text_nostop"].fillna("").str.split().map(set)
    counts = {l: Counter(w for doc in texts[df["label"] == l] for w in doc) for l in config.LABELS}
    total = sum(counts.values(), Counter())
    n_docs = {l: int((df["label"] == l).sum()) for l in config.LABELS}
    n_total = sum(n_docs.values())
    vocab = [w for w, c in total.items() if c >= min_count and len(w) > 1]
    result = {}
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, l in zip(axes.flat, config.LABELS):
        n_in = n_docs[l]
        n_out = n_total - n_in
        scores = {w: np.log((counts[l][w] + 1) / n_in) - np.log((total[w] - counts[l][w] + 1) / n_out)
                  for w in vocab}
        best = sorted(scores, key=scores.get, reverse=True)[:top]
        result[l] = [{"word": w, "log_ratio": round(float(scores[w]), 3), "docs": counts[l][w]} for w in best]
        ax.barh([_ar(w) for w in best][::-1], [scores[w] for w in best][::-1], color=LABEL_COLORS[l])
        ax.set_title(config.LABELS_FR[l])
    axes.flat[-1].axis("off")
    fig.suptitle(f"Mots les plus caractéristiques par émotion (log-ratio, présent dans ≥ {min_count} textes)")
    _save(fig, "fig05_distinctive_words.png")
    return result


def summary_stats(df: pd.DataFrame) -> dict:
    by_source = df.groupby("source")
    return {
        "rows": len(df),
        "label_counts": df["label"].value_counts().to_dict(),
        "source_counts": df["source"].value_counts().to_dict(),
        "variety_counts": df["variety"].value_counts().to_dict(),
        "words_median_by_source": by_source["n_words"].median().to_dict(),
        "words_p90_by_source": by_source["n_words"].quantile(0.9).to_dict(),
        "share_over_256_words_by_source": by_source["n_words"].apply(lambda s: round(float((s > 256).mean()), 4)).to_dict(),
        "share_with_emoji_by_source": by_source["emojis"].apply(lambda s: round(float(s.notna().mean()), 4)).to_dict(),
        "share_with_diacritics_by_source": by_source["diacritic_ratio"].apply(lambda s: round(float((s > 0.02).mean()), 4)).to_dict(),
        "stopword_token_reduction": round(float(1 - df["n_words_nostop"].sum() / df["n_words"].sum()), 4),
        "vocab_size_norm": len(set(" ".join(df["text_norm"]).split())),
        "etda_olympics_share_by_label": _olympics_share(df),
    }


OLYMPICS = r"(?:^| )\S*(?:اوليمبي|ميدالي|البرازيل|جانيرو)\S*(?: |$)|(?:^| )ريو(?: |$)"


def _olympics_share(df: pd.DataFrame) -> dict:
    # ETDA was collected around the 2016 Rio Olympics; measures topic leakage into labels.
    etda = df[df["source"] == "etda"]
    hit = etda["text_norm"].str.contains(OLYMPICS, regex=True)
    return hit.groupby(etda["label"]).mean().round(3).to_dict()


def run() -> dict:
    df = pd.read_csv(config.CORPUS_FILE)
    report = json.loads(config.QUALITY_REPORT.read_text(encoding="utf-8"))
    label_by_source(df)
    source_share_heatmap(df)
    length_by_source(df)
    cleaning_summary(report)
    stats = summary_stats(df)
    stats["distinctive_words"] = distinctive_words(df)
    config.EDA_REPORT.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


if __name__ == "__main__":
    s = run()
    print(json.dumps({k: v for k, v in s.items() if k != "distinctive_words"}, ensure_ascii=False, indent=2))
