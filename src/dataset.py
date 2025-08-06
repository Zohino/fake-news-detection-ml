"""
Data downloading and validation utilities for the Fake News Detection project.
"""

import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
from config import DATA_URLS, DATA_FILES
from loguru import logger
import typer

app = typer.Typer()


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """Download file from URL with progress bar."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        progress = tqdm(total=total_size, unit='B', unit_scale=True, desc=dest.name)

        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
        progress.close()
        logger.success(f"Downloaded: {dest}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url} → {dest.name}: {e}")
        return False


def validate_dataset(path: Path, expected_columns: list) -> bool:
    """Check if dataset contains required columns."""
    try:
        df = pd.read_csv(path, nrows=5)
        missing = set(expected_columns) - set(df.columns)
        if missing:
            logger.warning(f"Missing columns in {path.name}: {missing}")
            return False
        logger.info(f"{path.name} validated ✓")
        return True
    except Exception as e:
        logger.error(f"Validation failed for {path.name}: {e}")
        return False


def download_fake_news_data(force: bool = False) -> bool:
    """Download and validate training and test datasets."""
    train_expected = ['id', 'title', 'author', 'text', 'label']
    test_expected = ['id', 'title', 'author', 'text']
    success = True

    tasks = [
        ("train_raw", "fake_news_train", train_expected),
        ("test_raw", "fake_news_test", test_expected)
    ]

    for file_key, url_key, columns in tasks:
        file_path = DATA_FILES[file_key]
        if force or not file_path.exists():
            if not download_file(DATA_URLS[url_key], file_path) or not validate_dataset(file_path, columns):
                success = False
        else:
            logger.info(f"{file_path.name} already exists")

    if success:
        print_dataset_info()
    return success


def print_dataset_info():
    """Print shape, columns, and nulls for downloaded datasets."""
    for name in ["train_raw", "test_raw"]:
        path = DATA_FILES[name]
        if path.exists():
            df = pd.read_csv(path)
            logger.info(f"{name}: shape={df.shape}, columns={list(df.columns)}, missing={df.isnull().sum().sum()}")
            if 'label' in df.columns:
                logger.info(f"Class distribution: {df['label'].value_counts().to_dict()}")


def check_data_availability() -> dict:
    """Return availability and size of local raw CSV files."""
    return {
        name: {
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else 0,
            "path": str(path)
        }
        for name, path in DATA_FILES.items()
        if path.suffix == ".csv" and "raw" in str(path)
    }


@app.command()
def check():
    """Check if raw datasets exist locally."""
    status = check_data_availability()
    for name, info in status.items():
        symbol = "✅" if info["exists"] else "❌"
        size = f"({info['size']} bytes)" if info["exists"] else ""
        print(f"{name}: {symbol} {size}")


@app.command()
def download(force: bool = typer.Option(False, "--force", "-f", help="Force re-download even if files exist")):
    """Download datasets (train/test)."""
    if download_fake_news_data(force=force):
        print("\n🎉 Download complete.")
        print("Next: run `make dataset` or explore `notebooks/`.")
    else:
        print("\n❌ Download failed. See logs.")


if __name__ == "__main__":
    app()
