"""
N-gram model module: building, storing, and exposing n-gram probability tables
and backoff lookup across all orders from 1 up to NGRAM_ORDER.

Implements MLE (Maximum Likelihood Estimation) with Stupid Backoff — no
discounting or probability redistribution.  At inference time the highest-order
context is tried first; if unseen, the model falls back to progressively
shorter contexts down to unigrams.
"""

import json
import os
from collections import Counter


class NGramModel:
    """
    Build, store, and query n-gram probability tables with backoff.
    Supports any order from 1 up to ``ngram_order``.
    Vocabulary is thresholded so that rare words are replaced by ``<UNK>``.
    """

    UNK = "<UNK>"

    def __init__(self, ngram_order: int, unk_threshold: int) -> None:
        """
        Initialise the model.

        Parameters:
            ngram_order:   Maximum n-gram order (e.g. 4 for 4-grams).
            unk_threshold: Words appearing fewer than this many times are
                           replaced with <UNK>.
        """
        self.ngram_order = ngram_order
        self.unk_threshold = unk_threshold
        self.vocab: set[str] = set()
        self.probabilities: dict[int, dict] = {}  # order → prob table

    # ----------------------------------------------------------- build_vocab method -----------------------------------------------------------
    def build_vocab(self, token_file: str) -> None:
        """
        Build the vocabulary from a token file, applying UNK thresholding.

        Words appearing fewer than ``UNK_THRESHOLD`` times are excluded and
        will be mapped to ``<UNK>`` during count building.  ``<UNK>`` itself
        is always added to the vocabulary.

        Parameters:
            token_file: Path to the token file (one sentence per line).

        Returns:
            None  (sets ``self.vocab``).
        """
        with open(token_file, "r", encoding="utf-8") as f:
            sentences = [line.strip().split() for line in f if line.strip()]

        word_counts: Counter[str] = Counter()
        for tokens in sentences:
            word_counts.update(tokens)

        self.vocab = {
            word for word, count in word_counts.items()
            if count >= self.unk_threshold
        }
        self.vocab.add(self.UNK)

    # ------------------------------------- build_counts_and_probabilities method -----------------------------------------------------------
    def build_counts_and_probabilities(self, token_file: str) -> None:
        """
        Count all n-grams at orders 1 through ``ngram_order`` and compute MLE
        probabilities.

        For an n-gram of order *k* (k >= 2), probability is:
            P(w | context) = C(context + w) / C(context)
        For unigrams (k == 1):
            P(w) = C(w) / total_word_count

        Words not in ``self.vocab`` are mapped to ``<UNK>`` before counting.

        Parameters:
            token_file: Path to the token file.

        Returns:
            None  (sets ``self.probabilities``).
        """
        with open(token_file, "r", encoding="utf-8") as f:
            sentences = [line.strip().split() for line in f if line.strip()]

        # Map OOV words to <UNK>
        mapped_sentences = [
            [w if w in self.vocab else self.UNK for w in tokens]
            for tokens in sentences
        ]

        # --- Count n-grams at every order ---
        # counts[order] is a Counter keyed by tuple
        counts: dict[int, Counter[tuple[str, ...]]] = {
            order: Counter() for order in range(1, self.ngram_order + 1)
        }

        for tokens in mapped_sentences:
            for order in range(1, self.ngram_order + 1):
                for i in range(len(tokens) - order + 1):
                    ngram = tuple(tokens[i : i + order])
                    counts[order][ngram] += 1

        # --- Compute MLE probabilities ---
        self.probabilities = {}

        # Unigrams: P(w) = C(w) / total
        total_words = sum(counts[1].values())
        self.probabilities[1] = {
            word: count / total_words
            for (word,), count in counts[1].items()
        }

        # Higher orders: P(w | context) = C(context w) / C(context)
        for order in range(2, self.ngram_order + 1):
            table: dict[str, dict[str, float]] = {}
            for ngram, count in counts[order].items():
                context = ngram[:-1]
                word = ngram[-1]
                context_count = counts[order - 1][context]
                context_key = " ".join(context)
                if context_key not in table:
                    table[context_key] = {}
                table[context_key][word] = count / context_count
            self.probabilities[order] = table

    # ----------------------------------------------------------- lookup method -----------------------------------------------------------
    def lookup(self, context: list[str]) -> dict[str, float]:
        """
        Backoff lookup: try the highest-order context first, fall back to
        lower orders down to unigrams.

        The context words are mapped to ``<UNK>`` if they are not in the
        vocabulary.

        Parameters:
            context: A list of preceding word tokens (may be empty).

        Returns:
            A dict ``{word: probability}`` from the highest order that
            matches.  Returns an empty dict if no match at any order.
        """
        # Map OOV context words to <UNK>
        context = [w if w in self.vocab else self.UNK for w in context]

        for order in range(self.ngram_order, 0, -1):
            if order == 1:
                # Unigrams always match (no context needed)
                return dict(self.probabilities.get(1, {}))

            # For order k, we need the last (k-1) context words
            ctx_len = order - 1
            if len(context) < ctx_len:
                continue
            ctx_words = context[-ctx_len:]
            ctx_key = " ".join(ctx_words)

            table = self.probabilities.get(order, {})
            if ctx_key in table:
                return dict(table[ctx_key])

        return {}

    # -------------------------------------------------------- save_model method -----------------------------------------------------------
    def save_model(self, model_path: str) -> None:
        """
        Save all probability tables to a JSON file.

        Keys are ``"1gram"``, ``"2gram"``, … up to ``"<NGRAM_ORDER>gram"``.

        Parameters:
            model_path: Destination file path for model.json.

        Returns:
            None
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        out: dict[str, dict] = {}
        for order in range(1, self.ngram_order + 1):
            key = f"{order}gram"
            out[key] = self.probabilities.get(order, {})
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------------- save_vocab method -----------------------------------------------------------
    def save_vocab(self, vocab_path: str) -> None:
        """
        Save the vocabulary list to a JSON file.

        Parameters:
            vocab_path: Destination file path for vocab.json.

        Returns:
            None
        """
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(sorted(self.vocab), f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------ load method -----------------------------------------------------------
    def load(self, model_path: str, vocab_path: str) -> None:
        """
        Load model.json and vocab.json into the instance.

        Called once in main() before passing the model to Predictor.

        Parameters:
            model_path: Path to model.json.
            vocab_path: Path to vocab.json.

        Returns:
            None
        """
        with open(model_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.probabilities = {}
        for key, table in raw.items():
            order = int(key.replace("gram", ""))
            self.probabilities[order] = table

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = set(json.load(f))