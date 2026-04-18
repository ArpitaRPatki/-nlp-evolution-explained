"""
ngram_model.py
--------------
A clean, from-scratch N-gram language model with optional Laplace smoothing.

Supports:
  - unigrams, bigrams, trigrams (or any N)
  - raw MLE probability estimation
  - Laplace (additive) smoothing
  - text generation by sampling

Usage:
    from src.ngram_model import NGramModel

    model = NGramModel(n=2)
    model.fit(corpus_sentences)
    print(model.probability("cat", context=("the",)))
    print(model.generate(seed=("the",), max_words=10))
"""

import random
from collections import defaultdict, Counter
from typing import List, Tuple, Optional


class NGramModel:
    """
    A simple N-gram language model.

    Parameters
    ----------
    n : int
        The order of the model. n=1 is unigram, n=2 is bigram, etc.
    smoothing : float
        Additive smoothing constant. Set to 0.0 for raw MLE (will produce
        zero probabilities for unseen N-grams). Set to 1.0 for Laplace
        smoothing. Values in between give Lidstone smoothing.
    """

    def __init__(self, n: int = 2, smoothing: float = 0.0):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.smoothing = smoothing

        # counts[(context_tuple)] -> Counter({next_word: count})
        self._counts: dict = defaultdict(Counter)
        # total words seen for unigram fallback
        self._unigram_counts: Counter = Counter()
        self._vocabulary: set = set()
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, sentences: List[List[str]]) -> "NGramModel":
        """
        Train the model on a list of tokenized sentences.

        Parameters
        ----------
        sentences : list of list of str
            Each inner list is a tokenized sentence.
            e.g. [["the", "cat", "sat"], ["a", "dog", "ran"]]

        Returns
        -------
        self
        """
        for sentence in sentences:
            # Pad sentence with start/end tokens
            padded = ["<s>"] * (self.n - 1) + sentence + ["</s>"]

            for i in range(len(padded) - self.n + 1):
                gram = padded[i : i + self.n]
                context = tuple(gram[:-1])  # all but last token
                next_word = gram[-1]         # last token

                self._counts[context][next_word] += 1
                self._unigram_counts[next_word] += 1
                self._vocabulary.add(next_word)

        self._vocabulary.update(["<s>", "</s>"])
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Probability estimation
    # ------------------------------------------------------------------

    def probability(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Return P(word | context) with optional smoothing.

        Parameters
        ----------
        word    : the word whose probability we want
        context : tuple of (n-1) preceding words

        Returns
        -------
        float probability between 0 and 1
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling probability().")

        context = tuple(context[-(self.n - 1):])  # keep only the last n-1 tokens
        vocab_size = len(self._vocabulary)
        context_count = sum(self._counts[context].values())
        word_count = self._counts[context].get(word, 0)

        # Laplace / Lidstone smoothing
        numerator   = word_count + self.smoothing
        denominator = context_count + self.smoothing * vocab_size

        if denominator == 0:
            return 0.0
        return numerator / denominator

    def log_probability(self, word: str, context: Tuple[str, ...]) -> float:
        """Return log2 probability (useful for perplexity). Avoids log(0)."""
        import math
        p = self.probability(word, context)
        return math.log2(p) if p > 0 else float("-inf")

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def generate(
        self,
        seed: Optional[Tuple[str, ...]] = None,
        max_words: int = 20,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate text by sampling from the model.

        Parameters
        ----------
        seed        : starting context (n-1 tokens). Defaults to <s> tokens.
        max_words   : maximum number of words to generate
        temperature : > 1.0 = more random, < 1.0 = more deterministic

        Returns
        -------
        str : generated sentence
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before generating.")

        if seed is None:
            context = tuple(["<s>"] * (self.n - 1))
        else:
            context = tuple(seed[-(self.n - 1):])

        generated = list(context) if context[0] != "<s>" else []

        for _ in range(max_words):
            candidates = self._counts.get(context, {})
            if not candidates:
                break

            words = list(candidates.keys())
            counts = [candidates[w] ** (1.0 / temperature) for w in words]
            total = sum(counts)
            probs = [c / total for c in counts]

            next_word = random.choices(words, weights=probs, k=1)[0]
            if next_word == "</s>":
                break

            generated.append(next_word)
            context = tuple(list(context[1:]) + [next_word])

        return " ".join(w for w in generated if w not in ("<s>", "</s>"))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def vocab_size(self) -> int:
        return len(self._vocabulary)

    def get_top_next_words(self, context: Tuple[str, ...], top_k: int = 5):
        """Show the top-k most likely next words for a given context."""
        context = tuple(context[-(self.n - 1):])
        candidates = self._counts.get(context, {})
        if not candidates:
            return []
        total = sum(candidates.values())
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [(w, c / total) for w, c in ranked[:top_k]]

    def __repr__(self):
        status = "fitted" if self._fitted else "not fitted"
        return f"NGramModel(n={self.n}, smoothing={self.smoothing}, status={status})"
