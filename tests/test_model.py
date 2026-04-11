"""Tests for the NGramModel class (Module 2 — Model)."""

import math

from src.model.ngram_model import NGramModel


def _build_model(tmp_path):
    """Helper: write a small token file and build a model from it."""
    token_file = tmp_path / "tokens.txt"
    # Corpus: 5 sentences, words "the" and "cat" appear often, "xyz" appears once
    token_file.write_text(
        "the cat sat on the mat\n"
        "the cat sat on the rug\n"
        "the cat lay on the mat\n"
        "the dog sat on the mat\n"
        "xyz cat sat on the mat\n",
        encoding="utf-8",
    )
    model = NGramModel(ngram_order=3, unk_threshold=2)
    model.build_vocab(str(token_file))
    model.build_counts_and_probabilities(str(token_file))
    return model, str(token_file)


# ------------------------------------------------------------- build_vocab: replaces low-frequency words with <UNK>
def test_build_vocab_replaces_rare_words(tmp_path):
    model, _ = _build_model(tmp_path)
    assert "xyz" not in model.vocab


# ------------------------------------------------------------- build_vocab: includes <UNK> in vocabulary
def test_build_vocab_includes_unk(tmp_path):
    model, _ = _build_model(tmp_path)
    assert "<UNK>" in model.vocab


# ------------------------------------------------------------- lookup: returns non-empty dict for seen context
def test_lookup_seen_context_returns_nonempty(tmp_path):
    model, _ = _build_model(tmp_path)
    result = model.lookup(["the"])
    assert isinstance(result, dict)
    assert len(result) > 0


# ------------------------------------------------------------- lookup: unseen context falls back to unigram
def test_lookup_unseen_context_falls_back_to_unigram(tmp_path):
    model, _ = _build_model(tmp_path)
    result = model.lookup(["zzzzz", "qqqqq"])
    assert isinstance(result, dict)
    assert len(result) > 0


# ------------------------------------------------------------- lookup: empty dict when all orders fail
def test_lookup_empty_when_no_probabilities():
    model = NGramModel(ngram_order=3, unk_threshold=2)
    model.vocab = set()
    model.probabilities = {}
    result = model.lookup(["anything"])
    assert result == {}


# ------------------------------------------------------------- probabilities sum to approximately 1
def test_probabilities_sum_to_one_unigrams(tmp_path):
    model, _ = _build_model(tmp_path)
    total = sum(model.probabilities[1].values())
    assert math.isclose(total, 1.0, abs_tol=1e-9)


# ------------------------------------------------------------- probabilities sum to approximately 1 for bigram context
def test_probabilities_sum_to_one_bigram_context(tmp_path):
    model, _ = _build_model(tmp_path)
    # Pick "the" as a bigram context — it should have continuations summing to ~1
    if "the" in model.probabilities.get(2, {}):
        total = sum(model.probabilities[2]["the"].values())
        assert math.isclose(total, 1.0, abs_tol=1e-9)
