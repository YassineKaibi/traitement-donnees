import re
import unicodedata

# Harakat, tanween, shadda, sukun, superscript alef, Quranic annotation marks.
DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TATWEEL = "\u0640"
INVISIBLE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF]")
URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
EMOJI = re.compile(
    "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]"
)
# Three or more identical letters (جدااا, هههه) never occur in standard spelling.
ELONGATION = re.compile(r"(.)\1{2,}")
NON_ARABIC = re.compile(r"[^\u0621-\u063A\u0641-\u064A\s]")
WHITESPACE = re.compile(r"\s+")
ALEF_VARIANTS = re.compile(r"[إأآٱ]")

# Negations change emotional polarity, so they survive stopword removal.
NEGATIONS = {"لا", "لم", "لن", "ليس", "ليست", "ما", "غير", "بلا"}


def diacritic_ratio(text: str) -> float:
    return len(DIACRITICS.findall(text)) / len(text) if text else 0.0


def extract_emojis(text: str) -> str:
    return "".join(EMOJI.findall(text))


def strip_social(text: str) -> str:
    text = URL.sub(" ", text)
    text = MENTION.sub(" ", text)
    return text.replace("#", " ").replace("_", " ")


def normalize_letters(text: str, ta_marbuta: bool = True) -> str:
    text = ALEF_VARIANTS.sub("ا", text)
    text = text.replace("ى", "ي")
    if ta_marbuta:
        text = text.replace("ة", "ه")
    return text


def normalize(text: str, ta_marbuta: bool = True) -> str:
    # NFKC folds Arabic presentation forms (e.g. ﺑ, ﭐ) back to base letters.
    text = unicodedata.normalize("NFKC", text)
    text = INVISIBLE.sub("", text)
    text = strip_social(text)
    text = DIACRITICS.sub("", text).replace(TATWEEL, "")
    text = normalize_letters(text, ta_marbuta)
    text = NON_ARABIC.sub(" ", text)
    text = ELONGATION.sub(r"\1", text)
    return WHITESPACE.sub(" ", text).strip()


def build_stopwords(raw_list, ta_marbuta: bool = True) -> set:
    normalized = {normalize(w, ta_marbuta) for w in raw_list}
    return {w for w in normalized if w} - NEGATIONS


def remove_stopwords(text: str, stopwords: set) -> str:
    return " ".join(t for t in text.split() if t not in stopwords)
