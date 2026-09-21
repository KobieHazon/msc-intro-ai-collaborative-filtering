# Collaborative Filtering Recommender

A CS MSc Introduction to Artificial Intelligence exercise implementing user-based and item-based collaborative filtering. The recovered solution builds a user-item matrix, predicts missing ratings with cosine similarity, recommends unrated products, and compares the model with simple benchmarks using RMSE, precision, and recall.

`docs/report.pdf` contains my submitted report. Dataset user and product values are part of the supplied exercise data, not personal account details belonging to me.

## Requirements

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/)

## Setup

```bash
git clone https://github.com/KobieHazon/msc-intro-ai-collaborative-filtering.git
cd msc-intro-ai-collaborative-filtering
uv sync --dev
```

## Usage

Run the full recovered experiment against the supplied CSV data:

```bash
uv run python src/main.py
```

The supplied data contains roughly 400,000 training ratings and 100,000 test ratings, so the full matrix computation requires substantially more memory and time than the unit tests.

Run the focused deterministic tests with:

```bash
uv run pytest
```

## Repository layout

- `src/`: recommender implementation, evaluation helpers, and the experiment entry point.
- `data/`: supplied training and test CSV files, unchanged.
- `docs/`: my report.
- `tests/`: small deterministic regression tests.

Run commands from the repository root. The experiment resolves supplied data relative to its source location, so launching the script from another working directory also finds the same input files.
