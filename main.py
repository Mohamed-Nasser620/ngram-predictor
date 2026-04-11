"""N-gram Predictor — main entry point."""

import argparse
import os

from dotenv import load_dotenv

from src.data_prep.normalizer import Normalizer
from src.inference.predictor import Predictor
from src.model.ngram_model import NGramModel

# Load configuration from config/.env
load_dotenv(os.path.join("config", ".env"), override=True)

TRAIN_RAW_DIR = os.getenv("TRAIN_RAW_DIR")
TRAIN_TOKENS = os.getenv("TRAIN_TOKENS")
MODEL = os.getenv("MODEL")
VOCAB = os.getenv("VOCAB")
UNK_THRESHOLD = int(os.getenv("UNK_THRESHOLD", "3"))
NGRAM_ORDER = int(os.getenv("NGRAM_ORDER", "4"))
TOP_K = int(os.getenv("TOP_K", "3"))

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


def run_model() -> None:
    """Execute the full Module 2 model-building pipeline."""
    model = NGramModel(ngram_order=NGRAM_ORDER, unk_threshold=UNK_THRESHOLD)

    # Step 1 — Build vocabulary with UNK thresholding
    model.build_vocab(TRAIN_TOKENS)
    print(f"Vocabulary: {len(model.vocab)} words (UNK_THRESHOLD={UNK_THRESHOLD})")

    # Step 2-3 — Build counts and compute MLE probabilities
    model.build_counts_and_probabilities(TRAIN_TOKENS)
    for order in range(1, NGRAM_ORDER + 1):
        table = model.probabilities.get(order, {})
        entries = len(table)
        print(f"  {order}-gram: {entries} entries")

    # Step 4-5 — Save model and vocabulary
    model.save_model(MODEL)
    model.save_vocab(VOCAB)
    print(f"Saved model to {MODEL}")
    print(f"Saved vocab to {VOCAB}")


def run_inference() -> None:
    """Execute the Module 3 interactive CLI prediction loop."""
    # Load pre-built model
    model = NGramModel(ngram_order=NGRAM_ORDER, unk_threshold=UNK_THRESHOLD)
    model.load(MODEL, VOCAB)
    print(f"Loaded model ({NGRAM_ORDER}-gram, vocab={len(model.vocab)})")

    normalizer = Normalizer()
    predictor = Predictor(model, normalizer)

    print("Enter text to predict the next word (type 'quit' or Ctrl+C to exit):")
    try:
        while True:
            text = input("\n> ").strip()
            if text.lower() == "quit":
                break
            predictions = predictor.predict_next(text, TOP_K)
            print(f"Predictions: {predictions}")
    except (KeyboardInterrupt, EOFError):
        pass
    print("\nGoodbye.")


def main() -> None:
    parser = argparse.ArgumentParser(description="N-gram Predictor")
    parser.add_argument("--step", required=True,
                        choices=["dataprep", "model", "inference", "all"],
                        help="Pipeline step to run")
    args = parser.parse_args()

    if args.step == "dataprep":
        run_data_prep()
    elif args.step == "model":
        run_model()
    elif args.step == "inference":
        run_inference()
    elif args.step == "all":
        run_data_prep()
        run_model()
        run_inference()


if __name__ == "__main__":
    main()
