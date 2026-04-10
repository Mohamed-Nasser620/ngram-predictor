"""N-gram Predictor — main entry point."""

import argparse
import os

from dotenv import load_dotenv

from src.data_prep.normalizer import Normalizer

# Load configuration from config/.env
load_dotenv(os.path.join("config", ".env"))

TRAIN_RAW_DIR = os.getenv("TRAIN_RAW_DIR")
TRAIN_TOKENS = os.getenv("TRAIN_TOKENS")

# Development limit for faster iteration
DEV_SENTENCE_LIMIT = 100


def run_data_prep() -> None:
    """Execute the full Module 1 data-preparation pipeline."""
    normalizer = Normalizer()

    # Step 1 — Load raw texts
    raw_texts = normalizer.load(TRAIN_RAW_DIR)
    print(f"Loaded {len(raw_texts)} file(s) from {TRAIN_RAW_DIR}")

    all_tokenized_sentences: list[list[str]] = []

    for i, raw in enumerate(raw_texts, 1):
        # Step 2 — Strip Gutenberg header/footer
        body = normalizer.strip_gutenberg(raw)

        # Step 3 — Normalize
        clean = normalizer.normalize(body)

        # Step 4 — Sentence tokenize
        sentences = normalizer.sentence_tokenize(clean)

        # Step 5 — Word tokenize each sentence
        tokenized = [normalizer.word_tokenize(s) for s in sentences]
        all_tokenized_sentences.extend(tokenized)

        print(f"  Book {i}: {len(tokenized)} sentences")

        # Dev limit: stop early once we have enough sentences
        if len(all_tokenized_sentences) >= DEV_SENTENCE_LIMIT:
            all_tokenized_sentences = all_tokenized_sentences[:DEV_SENTENCE_LIMIT]
            break

    # Step 6 — Save to output file
    normalizer.save(all_tokenized_sentences, TRAIN_TOKENS)
    print(f"Saved {len(all_tokenized_sentences)} sentences to {TRAIN_TOKENS}")


def main() -> None:
    parser = argparse.ArgumentParser(description="N-gram Predictor")
    parser.add_argument("--step", required=True, choices=["dataprep"],
                        help="Pipeline step to run")
    args = parser.parse_args()

    if args.step == "dataprep":
        run_data_prep()


if __name__ == "__main__":
    main()
