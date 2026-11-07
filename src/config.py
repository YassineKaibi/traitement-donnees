from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

CORPUS_FILE = PROCESSED_DIR / "corpus.csv"
POEMS_RESERVE_FILE = PROCESSED_DIR / "poems_reserve.csv"
SEMEVAL_MIXED_FILE = PROCESSED_DIR / "semeval_mixed.csv"
QUALITY_REPORT = REPORTS_DIR / "quality_report.json"
EDA_REPORT = REPORTS_DIR / "eda_stats.json"

SEED = 42

LABELS = ["joy", "sadness", "love", "anger", "fear", "surprise", "neutral"]
LABELS_FR = {
    "joy": "Joie", "sadness": "Tristesse", "love": "Amour", "anger": "Colère",
    "fear": "Peur", "surprise": "Surprise", "neutral": "Neutre",
}

# Source label -> target label. Labels absent from a mapping are dropped.
LABEL_MAP = {
    "poems": {"joy": "joy", "sad": "sadness", "love": "love"},
    "etda": {
        "none": "neutral", "anger": "anger", "joy": "joy", "sadness": "sadness",
        "love": "love", "fear": "fear", "surprise": "surprise",
        # "sympathy" dropped: solidarity/condolence toward others, not the writer's emotion.
    },
    "semeval": {
        "anger": "anger", "joy": "joy", "sadness": "sadness",
        "fear": "fear", "love": "love", "surprise": "surprise", "none": "neutral",
        # optimism, pessimism, anticipation, trust, disgust dropped.
    },
}

# Poems may make up at most this share of any label after capping.
MAX_POEM_SHARE = 0.5

# ة -> ه merges two letters; on because informal and classical text use both inconsistently.
NORMALIZE_TA_MARBUTA = True

SOURCES = {
    "poems": {
        "url": "https://raw.githubusercontent.com/sakibsh/Arabic-Poem-Emotion/main/Arabic%20Poems%20dataset.csv",
        "file": RAW_DIR / "poems.csv",
        "register": "classical_poetry",
        "variety": "classical_msa",
        "line_byte_cap": 32767,
    },
    "etda": {
        "url": "https://raw.githubusercontent.com/AmrMehasseb/Emotional-Tone/master/Emotional-Tone-Dataset.csv",
        "file": RAW_DIR / "etda.csv",
        "register": "tweet",
        "variety": "egyptian",
    },
    "semeval": {
        # Terms of use forbid redistribution: download only, never commit.
        "url": "http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip",
        "file": RAW_DIR / "semeval2018_task1.zip",
        "register": "tweet",
        "variety": "mixed_dialects",
        "members": [
            "SemEval2018-Task1-all-data/Arabic/E-c/2018-E-c-Ar-train.txt",
            "SemEval2018-Task1-all-data/Arabic/E-c/2018-E-c-Ar-dev.txt",
            "SemEval2018-Task1-all-data/Arabic/E-c/2018-E-c-Ar-test-gold.txt",
        ],
    },
}
