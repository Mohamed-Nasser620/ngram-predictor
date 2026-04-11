# N-gram Predictor

A next-word prediction system built from scratch using n-gram language models with MLE (Maximum Likelihood Estimation) and Stupid Backoff. The model is trained on Project Gutenberg texts and predicts the most likely next word given an input phrase, falling back from higher-order to lower-order n-grams when a context is unseen.

## Requirements

- Python 3.14.3
- Dependencies listed in `requirements.txt`

## Setup

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd ngram-predictor
   ```

2. **Create and activate an Anaconda environment**

   ```bash
   conda create -n adi-py python=3.14.3
   conda activate adi-py
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Populate `config/.env`**

   Create a file `config/.env` with the following content:

   ```
   TRAIN_RAW_DIR = data/raw/train/
   TRAIN_TOKENS  = data/processed/train_tokens.txt
   MODEL         = data/model/model.json
   VOCAB         = data/model/vocab.json
   UNK_THRESHOLD = 3
   TOP_K         = 3
   NGRAM_ORDER   = 4
   ```

5. **Download raw text files**

   Download `.txt` files from [Project Gutenberg](https://www.gutenberg.org/) and place them in `data/raw/train/`.

## Usage

Run each pipeline step via `main.py --step`:

```bash
# Step 1 — Data Prep: load, clean, tokenize raw texts → train_tokens.txt
python main.py --step dataprep

# Step 2 — Model: build vocabulary, count n-grams, compute probabilities → model.json, vocab.json
python main.py --step model

# Step 3 — Inference: interactive CLI prediction loop
python main.py --step inference

# Run all steps in sequence (dataprep → model → inference)
python main.py --step all
```

### Example Session

```
$ python main.py --step inference

Enter text to predict the next word (type 'quit' or Ctrl+C to exit):

> holmes looked at
Predictions: ['blessington']

> the game is
Predictions: ['up', 'and', 'afoot']

> quit
Goodbye.
```

## Project Structure

```
ngram-predictor/
├── main.py                          # Entry point — CLI for all pipeline steps
├── requirements.txt                 # Third-party dependencies
├── README.md
├── config/
│   └── .env                         # Environment variables (not tracked by git)
├── data/
│   ├── raw/
│   │   └── train/                   # Raw .txt files from Project Gutenberg
│   ├── processed/
│   │   └── train_tokens.txt         # Tokenized training corpus
│   └── model/
│       ├── model.json               # N-gram probability tables
│       └── vocab.json               # Vocabulary list
├── src/
│   ├── data_prep/
│   │   └── normalizer.py            # Normalizer — load, clean, tokenize, save
│   ├── model/
│   │   └── ngram_model.py           # NGramModel — build counts, probabilities, backoff lookup
│   └── inference/
│       └── predictor.py             # Predictor — normalize input, predict top-k words
└── tests/
    ├── test_data_prep.py            # Unit tests for Normalizer
    ├── test_model.py                # Unit tests for NGramModel
    └── test_inference.py            # Unit tests for Predictor
```

## Running Tests

```bash
python -m pytest tests/
```

To see verbose output:

```bash
python -m pytest tests/ -v
```