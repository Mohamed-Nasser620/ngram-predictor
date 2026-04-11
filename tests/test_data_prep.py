"""Tests for the Normalizer class (Module 1 — Data Prep)."""

from src.data_prep.normalizer import Normalizer


# ------------------------------------------------------------- normalize: lowercase
def test_normalize_lowercases():
    n = Normalizer()
    assert n.normalize("HELLO WORLD") == "hello world"


# ------------------------------------------------------------- normalize: remove punctuation
def test_normalize_removes_punctuation():
    n = Normalizer()
    assert n.normalize("hello, world!") == "hello world"


# ------------------------------------------------------------- normalize: remove numbers
def test_normalize_removes_numbers():
    n = Normalizer()
    assert n.normalize("chapter 3") == "chapter"


# ------------------------------------------------------------- normalize: strip whitespace
def test_normalize_strips_whitespace():
    n = Normalizer()
    assert n.normalize("  hello   world  ") == "hello world"


# ------------------------------------------------------------- normalize: all steps in sequence
def test_normalize_all_steps_in_sequence():
    n = Normalizer()
    result = n.normalize("  Hello, World! 42  ")
    assert result == "hello world"


# ------------------------------------------------------------- normalize: unicode punctuation
def test_normalize_removes_unicode_punctuation():
    n = Normalizer()
    result = n.normalize("it\u2019s a \u201ctest\u201d")
    assert "\u2019" not in result
    assert "\u201c" not in result


# ------------------------------------------------------------- strip_gutenberg: removes header and footer
def test_strip_gutenberg_removes_header_and_footer():
    n = Normalizer()
    text = (
        "Header text\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK TITLE ***\n"
        "Body of the book.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TITLE ***\n"
        "Footer text"
    )
    assert n.strip_gutenberg(text) == "Body of the book."


# ------------------------------------------------------------- strip_gutenberg: multiline marker
def test_strip_gutenberg_multiline_marker():
    n = Normalizer()
    text = (
        "Preamble\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK LONG\n"
        "TITLE ***\n"
        "Content here.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK LONG TITLE ***\n"
    )
    assert n.strip_gutenberg(text) == "Content here."


# ------------------------------------------------------------- sentence_tokenize: returns list with at least one element
def test_sentence_tokenize_returns_nonempty_list():
    n = Normalizer()
    result = n.sentence_tokenize("hello world")
    assert isinstance(result, list)
    assert len(result) >= 1


# ------------------------------------------------------------- sentence_tokenize: splits on newlines
def test_sentence_tokenize_splits_on_newlines():
    n = Normalizer()
    result = n.sentence_tokenize("one\ntwo\nthree")
    assert result == ["one", "two", "three"]


# ------------------------------------------------------------- word_tokenize: returns list of strings
def test_word_tokenize_returns_list_of_strings():
    n = Normalizer()
    result = n.word_tokenize("the cat sat")
    assert isinstance(result, list)
    assert all(isinstance(t, str) for t in result)


# ------------------------------------------------------------- word_tokenize: no empty tokens
def test_word_tokenize_no_empty_tokens():
    n = Normalizer()
    result = n.word_tokenize("the cat sat")
    assert all(len(t) > 0 for t in result)

