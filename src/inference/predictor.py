"""
Inference module: accepting a pre-loaded NGramModel and Normalizer, normalizing
input text, and returning the top-k predicted next words sorted by probability.

Backoff lookup is delegated entirely to NGramModel.lookup().
"""

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel


class Predictor:
    """
    Accept a pre-loaded NGramModel and Normalizer via the constructor, normalize
    input text, and return the top-k predicted next words sorted by probability.

    Backoff lookup is delegated to NGramModel.lookup().
    """


    def __init__(self, model: NGramModel, normalizer: Normalizer) -> None:
        """
        Accept a pre-loaded NGramModel and Normalizer instance.

        Parameters:
            model:      A fully loaded NGramModel instance.
            normalizer: A Normalizer instance (used for text cleaning).
        """
        self.model = model
        self.normalizer = normalizer

    def normalize(self, text: str) -> list[str]:
        """
        Normalize the input text and extract the last NGRAM_ORDER-1 words
        as context for lookup.

        Parameters:
            text: Raw user input string.

        Returns:
            A list of the last (ngram_order - 1) word tokens from the
            normalized text.  May be shorter if the input has fewer words.
        """
        clean = self.normalizer.normalize(text)
        words = clean.split()
        ctx_len = self.model.ngram_order - 1
        return words[-ctx_len:] if words else []

    def map_oov(self, context: list[str]) -> list[str]:
        """
        Replace out-of-vocabulary words in *context* with <UNK>.

        Parameters:
            context: A list of word tokens.

        Returns:
            A new list with OOV words replaced by the model's UNK token.
        """
        return [
            w if w in self.model.vocab else self.model.UNK
            for w in context
        ]

    def predict_next(self, text: str, k: int) -> list[str]:
        """
        Predict the top-k next words for the given input text.

        Pipeline: normalize → map_oov → NGramModel.lookup() → rank by
        probability → return top-k words.

        Parameters:
            text: Raw user input string.
            k:    Number of top predictions to return.

        Returns:
            A list of up to *k* words sorted by descending probability.
            Returns an empty list if no predictions are found at any order.
        """
        context = self.normalize(text)
        context = self.map_oov(context)
        candidates = self.model.lookup(context)
        if not candidates:
            return []
        ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        return [word for word, _ in ranked[:k]]