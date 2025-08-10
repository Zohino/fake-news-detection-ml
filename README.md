# fake-news-detection-ml

[![Cookiecutter Data Science logo](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter "Cookiecutter Data Science")](https://cookiecutter-data-science.drivendata.org/)

This repository is a term paper for the Introduction to Machine Learning course at UJEP. It implements a complete data science pipeline to predict the reliability of news articles based on their content.

---

## Recommended setup

This project is designed to be run using the following tools in a **POSIX-compliant shell** environment (e.g., Linux/macOS):

- [`make`](https://www.gnu.org/software/make/) – to execute project commands defined in the Makefile
- [`uv`](https://github.com/astral-sh/uv) – for efficient Python dependency and virtual environment management

> **Automation Notice**  
> The Makefile is specifically written to automate tasks using `uv` within a `sh`-compatible shell.  
> Running this project with anything other than a POSIX shell (e.g., PowerShell on Windows) may result in failures.

### Compatibility Notes

- A POSIX-compliant shell (like `bash` or `zsh`) is **required**.
- **Windows PowerShell is not supported.**
- On Windows, it's recommended to use [WSL (Windows Subsystem for Linux)](https://learn.microsoft.com/en-us/windows/wsl/) to run the project in a compatible shell environment.

## Alternative: Manual Setup with pip

If you're unable to use POSIX shell, `make` and `uv`, you can still set up the environment manually using `pip`:

```bash
pip install -r requirements.txt
```

---

## Usage

### With recommended setup

To set up and run the project using recommended method:

```bash
# Run the full project pipeline
make pipeline

# Or run it step by step
make help                   # List all available commands
make create_environment     # Create .venv
make requirements           # Install dependencies
make data                   # Download dataset
make preprocess             # Preprocess dataset
make train                  # Train model
make predict                # Infer

# Run notebooks all at once
make notebooks

# Or run notebooks one by one
make eda                    # First exploratory data analysis
make features               # Explore dataset features
make visualize              # Dataset visualization

# Additional make utilities
make clean                  # Delete compiled Python files
make format                 # Format source code with ruff
make lint                   # Lint code with ruff
```

After that, explore each notebook or read the [report](reports/report.md) (in Czech only).

### With manual setup using pip

To set up and run the project using alternative method:

```bash
# Create environment and activate (Optional)
python3 -m venv .venv
source .venv/bin/activate

# Install tools
pip install -r requirements.txt

# Run scripts
python3 /src/dataset.py download
python3 src/preprocessing/preprocessing.py
python3 src/modeling/train.py
python3 src/modeling/predict.py

# Run notebooks
papermill notebooks/0.1-mz-initial-eda.ipynb notebooks/0.1-mz-initial-eda.ipynb
papermill notebooks/1.1-mz-data-cleaning-and-feature-creation.ipynb
papermill notebooks/2.1-mz-visualization.ipynb

# Additional ruff usage for linting and formatting
ruff check --fix
ruff format

ruff format --check
ruff check
```

After that, explore each notebook or read the [report](reports/report.md) (in Czech only).

---

## Project Organization

```txt
├── LICENSE                             <- Open source license
├── Makefile                            <- Makefile with convenience commands and full pipeline
├── README.md                           <- Top-level README
├── data
│   ├── external                        <- Data from third party sources
│   ├── interim                         <- Intermediate data that has been transformed
│   ├── processed                       <- The final, canonical data sets for modeling
│   ├── raw                             <- The original, immutable data dump
├── logs
├── models                              <- Trained and serialized models, model predictions, or model summaries
│   ├── predictions
│   ├── trained
├── notebooks                           <- Jupyter notebooks. Naming convention is a number (for ordering),
│                                           the creator's initials, and a short `-` delimited description,
│                                           e.g. `1.1-mz-initial-data-exploration`.
│   ├── 0.1-mz-initial-eda.ipynb
│   ├── 1.1-mz-data-cleaning-and-feature-creation.ipynb     
│   ├── 2.1-mz-visualization.ipynb
├── pyproject.toml                      <- Project configuration file with package metadata
├── reports                             <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures                         <- Generated graphics and figures to be used in reporting
│   │   ├── enhanced_data_analysis.png
│   ├── report.md
├── requirements.txt                    <- The requirements file for compatibility
├── src                                 <- Source code for use in this project
│   ├── __init__.py
│   ├── config.py                       <- Project-wide configuration and constants
│   ├── dataset.py                      <- Scripts to download or generate data
│   ├── modeling
│   │   ├── __init__.py
│   │   ├── modeling.py                 <- Modeling
│   │   ├── optuna_optimization.py      <- Hyperparameter optimization
│   │   ├── predict.py                  <- Run model inference
│   │   ├── train.py                    <- Train model
│   ├── preprocessing
│   │   ├── __init__.py
│   │   ├── preprocessing.py            <- Dataset preprocessing
├── uv.lock
```

## Notebooks Overview

| Notebook                                  | Purpose|
|-------------------------------------------|-------------------------------------------|
|0.1-mz-initial-eda                         | Exploratory data analysis                 |
|1.1-mz-data-cleaning-and-feature-creation  | Cleaning dataset and exploring features   |
|2.1-mz-visualization                       | Plots and graphics                        |

## Reproducibility

This project is structured using the Cookiecutter Data Science template to ensure clean organization and reproducible results.

## License

MIT License. See [`LICENSE`](LICENSE).
