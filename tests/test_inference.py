"""Tests for the Predictor class (Module 3 — Inference)."""

from src.data_prep.normalizer import Normalizer
from src.inference.predictor import Predictor
from src.model.ngram_model import NGramModel


def _build_predictor(tmp_path):
    """Helper: build a small model and return a Predictor wired to it."""
    token_file = tmp_path / "tokens.txt"
    token_file.write_text(
        "the cat sat on the mat\n"
        "the cat sat on the rug\n"
        "the cat lay on the mat\n"
        "the dog sat on the mat\n"
        "the cat sat on the floor\n",
        encoding="utf-8",
    )
    model = NGramModel(ngram_order=3, unk_threshold=2)
    model.build_vocab(str(token_file))
    model.build_counts_and_probabilities(str(token_file))
    normalizer = Normalizer()
    return Predictor(model, normalizer)


# ------------------------------------------------------------- predict_next: returns exactly k predictions for seen context
def test_predict_next_returns_k_predictions(tmp_path):
    predictor = _build_predictor(tmp_path)
    # "the" is a common context — should have at least 3 continuations
    result = predictor.predict_next("the", k=3)
    assert len(result) == 3


# ------------------------------------------------------------- predict_next: sorted by probability highest first
def test_predict_next_sorted_by_probability(tmp_path):
    predictor = _build_predictor(tmp_path)
    result = predictor.predict_next("the", k=3)
    # Verify ordering: look up raw probabilities and confirm descending
    context = predictor.normalize("the")
    context = predictor.map_oov(context)
    candidates = predictor.model.lookup(context)
    for i in range(len(result) - 1):
        assert candidates[result[i]] >= candidates[result[i + 1]]


# ------------------------------------------------------------- predict_next: handles all-OOV context without crashing
def test_predict_next_all_oov_context(tmp_path):
    predictor = _build_predictor(tmp_path)
    result = predictor.predict_next("zzz qqq rrr", k=3)
    assert isinstance(result, list)


# ------------------------------------------------------------- map_oov: replaces unknown words with <UNK>
def test_map_oov_replaces_unknown_words(tmp_path):
    predictor = _build_predictor(tmp_path)
    result = predictor.map_oov(["xyznotinvocab"])
    assert result == ["<UNK>"]


# ------------------------------------------------------------- map_oov: leaves known words unchanged
def test_map_oov_keeps_known_words(tmp_path):
    predictor = _build_predictor(tmp_path)
    result = predictor.map_oov(["the"])
    assert result == ["the"]
