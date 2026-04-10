"""
Data-preparation module: loading, cleaning, tokenizing, and saving the corpus.

The Normalizer class is used in two contexts:
  - Module 1 (Data Prep): processes whole raw files — load, strip, normalize, tokenize, and save.
  - Module 3 (Inference): only normalize(text) is called on a single input string to prepare it for model lookup.
"""

import os
import re
import string


class Normalizer:
    """
    Handles loading, cleaning, tokenizing, and saving tokenized text.

    Dual-use: full pipeline for data prep, and normalize() alone for inference.
    """

    # --------------------------------------------- load method ---------------------------------------------
    def load(self, folder_path: str) -> list[str]:
        """
        Load all .txt files from a folder and return their contents.

        Parameters:
            folder_path: Path to the directory containing .txt files.

        Returns:
            A list of strings, one per file, in sorted filename order.
        """
        texts = []
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    texts.append(f.read())
        return texts

    # --------------------------------------------- strip_gutenberg method ---------------------------------------------
    def strip_gutenberg(self, text: str) -> str:
        """
        Remove Project Gutenberg header and footer from *text*.

        Removes everything up to and including the line matching
        ``*** START OF THE PROJECT GUTENBERG EBOOK ... ***``
        and everything from and including the line matching
        ``*** END OF THE PROJECT GUTENBERG EBOOK ... ***``.

        Parameters:
            text: The raw text of a Gutenberg ebook.

        Returns:
            The body text with header and footer removed.
        """
        # Header: remove everything up to and including the START marker.
        # The marker may span multiple lines, so we match the full pattern.
        start_pattern = r"(?s).*?\*{3}\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*{3}"
        text = re.sub(start_pattern, "", text, count=1)

        # Footer: remove everything from the END marker onward.
        end_pattern = r"(?s)\*{3}\s*END OF THE PROJECT GUTENBERG EBOOK.*"
        text = re.sub(end_pattern, "", text, count=1)

        return text.strip()

    # --------------------------------------------- lowercase method ---------------------------------------------
    def lowercase(self, text: str) -> str:
        """
        Lowercase all characters in *text*.

        Parameters:
            text: Input string.

        Returns:
            The lowercased string.
        """
        return text.lower()

    # --------------------------------------------- remove_punctuation method ---------------------------------------------
    def remove_punctuation(self, text: str) -> str:
        """
        Remove all punctuation characters from *text*.

        Handles both ASCII punctuation and Unicode punctuation (e.g. smart
        quotes, em-dashes).

        Parameters:
            text: Input string.

        Returns:
            The string with punctuation removed.
        """
        # Remove ASCII punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Remove Unicode punctuation (smart quotes, dashes, etc.)
        text = re.sub(r"[^\w\s]", "", text)
        return text

    # --------------------------------------------- remove_numbers method ---------------------------------------------
    def remove_numbers(self, text: str) -> str:
        """
        Remove all digit characters from *text*.

        Parameters:
            text: Input string.

        Returns:
            The string with digits removed.
        """
        return re.sub(r"\d+", "", text)

    # --------------------------------------------- remove_whitespace method ---------------------------------------------
    def remove_whitespace(self, text: str) -> str:
        """
        Collapse runs of whitespace into single spaces and strip blank lines.

        Parameters:
            text: Input string.

        Returns:
            The string with extra whitespace removed.
        """
        # Replace any run of whitespace (including newlines) that does NOT contain a newline with a single space, but preserve single newlines
        text = re.sub(r"[ \t]+", " ", text)          # horizontal whitespace → single space
        text = re.sub(r"\n[ \t]*\n+", "\n", text)    # blank lines → single newline
        return text.strip()

    # --------------------------------------------- normalize method ---------------------------------------------
    def normalize(self, text: str) -> str:
        """
        Apply all normalization steps in order and return the cleaned text.

        Pipeline: lowercase → remove punctuation → remove numbers → remove whitespace.

        This is the single method other modules call to normalize text
        consistently (both full-corpus prep and single-string inference).

        Parameters:
            text: Input string.

        Returns:
            The fully normalized string.
        """
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text

    # --------------------------------------------- sentence_tokenize method ---------------------------------------------
    def sentence_tokenize(self, text: str) -> list[str]:
        """
        Split *text* into a list of sentences.

        Uses a regex that splits on sentence-ending punctuation or newlines.
        After normalization, punctuation is already removed, so splitting on
        newlines is the primary method.

        Parameters:
            text: Input text (typically already normalized).

        Returns:
            A list of sentence strings (empty strings are discarded).
        """
        # Split on newlines
        sentences = re.split(r"[.!?]\s+|\n", text)
        return [s.strip() for s in sentences if s.strip()]

    # -------------------------------------------- word_tokenize method ---------------------------------------------
    def word_tokenize(self, sentence: str) -> list[str]:
        """
        Split a single sentence into a list of word tokens.

        Parameters:
            sentence: A single sentence string.

        Returns:
            A list of token strings.
        """
        return sentence.split()

    # ---------------------------------------------------- save method ---------------------------------------------
    def save(self, sentences: list[list[str]], filepath: str) -> None:
        """
        Write tokenized sentences to an output file.

        Each sentence is written as one line with tokens separated by spaces.

        Parameters:
            sentences: A list of token-lists (one list per sentence).
            filepath:  Destination file path.

        Returns:
            None
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for tokens in sentences:
                f.write(" ".join(tokens) + "\n")