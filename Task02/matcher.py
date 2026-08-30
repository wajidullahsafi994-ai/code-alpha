"""
matcher.py
----------
Similarity matching engine for FAQ retrieval.

Strategy:
  1. Load all FAQs from the JSON dataset.
  2. Preprocess every FAQ question through the NLP pipeline.
  3. Build a TF-IDF matrix over all preprocessed FAQ questions.
  4. At query time, preprocess the user's input, vectorize it with the same
     TF-IDF model, then compute cosine similarity against all FAQ vectors.
  5. Return the FAQ with the highest similarity score (plus the score itself).
"""

import json
import os
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessor import preprocess_to_string

# ── paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_FAQ_PATH  = os.path.join(_BASE_DIR, "data", "faqs.json")

# Minimum cosine similarity score to consider a match valid.
# Below this threshold we return a "I don't know" fallback.
# Set higher to avoid false positives from common question words ("what", "is").
CONFIDENCE_THRESHOLD = 0.20


class FAQMatcher:
    """
    Loads the FAQ dataset, builds a TF-IDF index, and answers user queries
    by returning the most similar FAQ entry.
    """

    def __init__(self, faq_path: str = _FAQ_PATH):
        self.faqs: list[dict]         = []
        self.processed_questions: list[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix             = None

        self._load_faqs(faq_path)
        self._build_index()

    # ── setup ──────────────────────────────────────────────────────────────────

    def _load_faqs(self, path: str) -> None:
        """Read FAQ entries from the JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)

        if not self.faqs:
            raise ValueError(f"FAQ file is empty: {path}")

        print(f"[FAQMatcher] Loaded {len(self.faqs)} FAQs from '{path}'")

    def _build_index(self) -> None:
        """Preprocess all FAQ questions and fit the TF-IDF vectorizer."""
        self.processed_questions = [
            preprocess_to_string(faq["question"]) for faq in self.faqs
        ]

        # TF-IDF with character n-grams added to catch partial word matches
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),   # unigrams + bigrams
            sublinear_tf=True,    # dampen high-frequency terms
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.processed_questions
        )
        print(
            f"[FAQMatcher] TF-IDF index built — "
            f"matrix shape: {self.tfidf_matrix.shape}"
        )

    # ── matching ───────────────────────────────────────────────────────────────

    def match(self, user_input: str) -> dict:
        """
        Find the FAQ entry most similar to *user_input*.

        Returns a dict:
        {
            "question"  : str,   # matched FAQ question
            "answer"    : str,   # matched FAQ answer
            "score"     : float, # cosine similarity (0–1)
            "confident" : bool,  # True when score ≥ CONFIDENCE_THRESHOLD
        }
        """
        processed_input = preprocess_to_string(user_input)

        # Edge case: empty input after preprocessing
        if not processed_input.strip():
            return self._no_match()

        # Vectorize user input using the already-fitted vocabulary
        input_vec = self.vectorizer.transform([processed_input])

        # Cosine similarity against every FAQ vector
        scores     = cosine_similarity(input_vec, self.tfidf_matrix).flatten()
        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < CONFIDENCE_THRESHOLD:
            return self._no_match(best_score)

        return {
            "question" : self.faqs[best_idx]["question"],
            "answer"   : self.faqs[best_idx]["answer"],
            "score"    : round(best_score, 4),
            "confident": True,
        }

    def get_top_matches(self, user_input: str, top_n: int = 3) -> list[dict]:
        """
        Return the *top_n* most similar FAQs (useful for debugging / UI hints).
        """
        processed_input = preprocess_to_string(user_input)
        if not processed_input.strip():
            return []

        input_vec  = self.vectorizer.transform([processed_input])
        scores     = cosine_similarity(input_vec, self.tfidf_matrix).flatten()
        top_idxs   = np.argsort(scores)[::-1][:top_n]

        return [
            {
                "question" : self.faqs[i]["question"],
                "answer"   : self.faqs[i]["answer"],
                "score"    : round(float(scores[i]), 4),
            }
            for i in top_idxs
        ]

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _no_match(score: float = 0.0) -> dict:
        return {
            "question" : None,
            "answer"   : (
                "I'm sorry, I couldn't find a relevant answer to your question. "
                "Please try rephrasing, or contact our support team at "
                "support@shop.com for further help."
            ),
            "score"    : round(score, 4),
            "confident": False,
        }


# ── quick smoke-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    matcher = FAQMatcher()
    test_queries = [
        "How can I return a product?",
        "I forgot my password, what do I do?",
        "Do you deliver to other countries?",
        "How can I earn rewards?",
        "What is the meaning of life?",   # should hit the fallback
        "",                                # edge case: empty string
    ]

    print("\n=== Matcher Smoke Test ===\n")
    for q in test_queries:
        result = matcher.match(q)
        print(f"Query     : {q!r}")
        print(f"Matched   : {result['question']}")
        print(f"Score     : {result['score']}")
        print(f"Confident : {result['confident']}")
        print(f"Answer    : {result['answer'][:80]}...")
        print("-" * 55)
