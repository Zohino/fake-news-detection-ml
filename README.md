# fake-news-detection-ml

[![Cookiecutter Data Science logo](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter "Cookiecutter Data Science")](https://cookiecutter-data-science.drivendata.org/)

This repository is a term paper for the Introduction to Machine Learning course at UJEP. It implements a complete data science pipeline to predict the reliability of news articles based on their content.

---

## 🚀 Getting Started

This project uses [uv](https://github.com/astral-sh/uv) for managing environments and dependencies. You can also use `pip` if preferred.

---

### ✅ Setup with `uv` (recommended)

> Make sure you have Python 3.12 and [uv](https://github.com/astral-sh/uv) installed.

```bash
# Create virtual environment
make create_environment

# Activate the environment
source .venv/bin/activate  # macOS/Linux
# OR
.\\.venv\\Scripts\\activate  # Windows

# Install dependencies
make requirements

# Run the full pipeline (download, preprocess, train, predict, report)
make pipeline
```

## 🐍 Setup with pip (alternative)

1. Create and activate a virtual environment:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # macOS/Linux
    # OR
    .\\.venv\\Scripts\\activate  # Windows
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the pipeline step-by-step:

    ```bash
    # Download raw data
    python detection/dataset.py download

    # Preprocess data
    python notebooks/1.0-data-cleaning-and-feature-creation.ipynb

    # Train model
    python train.py

    # Predict on test data
    python predict.py

    # Generate publication report
    python notebooks/4.0-publication.ipynb
    ```

## 🛠 Makefile Commands

```bash
make help          # List all available make commands
make create_environment
make requirements
make data
make preprocess
make train
make predict
make publish
make pipeline      # Run the full end-to-end pipeline
make clean
```

## 📁 Project Organization

```txt
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│   ├── predictions
│   └── trained
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-initial-data-exploration`.
│   ├── 0.1-mz-initial-eda.ipynb
│   ├── 1.1-mz-data-cleaning-and-feature-creation.ipynb
│   ├── 2.1-mz-visualization.ipynb
│   ├── 3.1-mz-modeling.ipynb
│   └── 4.1-mz-publication.ipynb
├── pyproject.toml     <- Project configuration file with package metadata
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│
├── uv.lock
│
└── src          <- Source code for use in this project
    ├── __init__.py
    ├── config.py               <- Project-wide configuration and constants
    ├── dataset.py              <- Scripts to download or generate data
    ├── modeling/
    │   ├── __init__.py         
    |   ├── modeling.py
    │   ├── train.py            <- Train models
    └   └── predict.py          <- Run model inference
```

## 📓 Notebooks Overview

| Notebook                              | Purpose                              |
|---------------------------------------|--------------------------------------|
|1.0-data-cleaning-and-feature-creation | Preprocess raw data                  |
|2.0-visualizations                     | Visual EDA (class dist, text length) |
|3.0-modeling                           | Run model training                   |
|4.0-publication                        | Generate prediction summaries        |

## ✅ Reproducibility

This project is structured using the Cookiecutter Data Science template to ensure clean organization and reproducible results.

## 📄 License

MIT License. See `LICENSE`.
