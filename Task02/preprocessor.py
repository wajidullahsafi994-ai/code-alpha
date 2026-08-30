"""
preprocessor.py
---------------
NLP text preprocessing utilities using NLTK.
Handles tokenization, stopword removal, punctuation cleaning, and lemmatization.
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources on first use
def download_nltk_resources():
    resources = [
        ("tokenizers/punkt",         "punkt"),
        ("tokenizers/punkt_tab",     "punkt_tab"),
        ("corpora/stopwords",        "stopwords"),
        ("corpora/wordnet",          "wordnet"),
        ("corpora/omw-1.4",          "omw-1.4"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except (LookupError, OSError):
            # LookupError  → resource not found
            # OSError      → partial/corrupt download (NLTK 3.8+)
            nltk.download(pkg, quiet=True)

download_nltk_resources()

# ── module-level singletons ────────────────────────────────────────────────────
_lemmatizer   = WordNetLemmatizer()
_stop_words   = set(stopwords.words("english"))

# Keep a small set of question words that carry intent – don't strip them
_KEEP_WORDS   = {"how", "when", "where", "why", "who", "which", "can",
                 "do", "does", "will", "would", "should"}
_STOP_WORDS   = _stop_words - _KEEP_WORDS


def clean_text(text: str) -> str:
    """
    Lowercase the text and strip everything except letters and spaces.

    >>> clean_text("Hello, World! 123")
    'hello world'
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z\s]", " ", text)   # remove punctuation / digits
    text = re.sub(r"\s+", " ", text).strip() # collapse whitespace
    return text


def tokenize(text: str) -> list[str]:
    """
    Tokenize cleaned text into a list of word tokens.

    >>> tokenize("how do i reset my password")
    ['how', 'do', 'i', 'reset', 'my', 'password']
    """
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Remove stopwords while preserving question / intent words.

    >>> remove_stopwords(['how', 'do', 'i', 'reset', 'my', 'password'])
    ['how', 'do', 'reset', 'password']
    """
    return [t for t in tokens if t not in _STOP_WORDS]


def lemmatize(tokens: list[str]) -> list[str]:
    """
    Reduce each token to its base (lemma) form.

    >>> lemmatize(['resetting', 'passwords', 'orders'])
    ['resetting', 'password', 'order']
    """
    return [_lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text: str) -> list[str]:
    """
    Full preprocessing pipeline:
      raw text → clean → tokenize → remove stopwords → lemmatize

    Returns a list of processed tokens.

    >>> preprocess("How do I reset my password?")
    ['how', 'do', 'reset', 'password']
    """
    cleaned = clean_text(text)
    tokens  = tokenize(cleaned)
    tokens  = remove_stopwords(tokens)
    tokens  = lemmatize(tokens)
    return tokens


def preprocess_to_string(text: str) -> str:
    """
    Run the full pipeline and return the result as a single space-joined string.
    Useful for feeding into TF-IDF vectorizers.

    >>> preprocess_to_string("How do I reset my password?")
    'how do reset password'
    """
    return " ".join(preprocess(text))


# ── quick smoke-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "How do I reset my password?",
        "What is your return policy?",
        "Can I track my order online?",
        "Do you ship internationally?",
    ]
    print("=== Preprocessor Smoke Test ===\n")
    for s in samples:
        print(f"Input   : {s}")
        print(f"Tokens  : {preprocess(s)}")
        print(f"String  : {preprocess_to_string(s)}")
        print("-" * 45)
