"""WANDS dataset loading with automatic download."""

import subprocess
from pathlib import Path

import pandas as pd

WANDS_REPO = "https://github.com/wayfair/WANDS.git"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "wands"

GRADE_MAP = {"Exact": 2, "Partial": 1, "Irrelevant": 0}


def ensure_wands(data_dir: Path = DATA_DIR) -> Path:
    """Clone the WANDS dataset repo if not already present."""
    if not (data_dir / "dataset").exists():
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading WANDS dataset to {data_dir} ...")
        subprocess.run(
            ["git", "clone", "--depth=1", WANDS_REPO, str(data_dir)],
            check=True,
        )
    return data_dir


def load_products(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the WANDS product catalog.

    Returns a DataFrame with columns:
        product_id, title, description, category
    """
    ensure_wands(data_dir)
    df = pd.read_csv(data_dir / "dataset" / "product.csv", sep="\t")
    df = df.rename(columns={
        "product_name": "title",
        "product_description": "description",
        "category hierarchy": "category",
    })
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["category"] = df["category"].fillna("")
    return df[["product_id", "title", "description", "category"]].reset_index(drop=True)


def load_queries(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the WANDS queries.

    Returns a DataFrame with columns: query_id, query
    """
    ensure_wands(data_dir)
    return pd.read_csv(data_dir / "dataset" / "query.csv", sep="\t")


def load_judgments(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the WANDS relevance judgments.

    Returns a DataFrame with columns: query_id, product_id, label, grade
    where grade is 2 (Exact), 1 (Partial), or 0 (Irrelevant).
    """
    ensure_wands(data_dir)
    df = pd.read_csv(data_dir / "dataset" / "label.csv", sep="\t")
    df["grade"] = df["label"].map(GRADE_MAP)
    return df[["query_id", "product_id", "label", "grade"]]
