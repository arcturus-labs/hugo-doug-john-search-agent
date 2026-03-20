"""BM25 keyword search over the WANDS product catalog using searcharray."""

import string

import numpy as np
import pandas as pd
import Stemmer
from searcharray import SearchArray

from search_agent.data import load_products

# ---------------------------------------------------------------------------
# Tokenizer (Snowball stemmer, matches Doug Turnbull's cheat-at-search impl)
# ---------------------------------------------------------------------------

_stemmer = Stemmer.Stemmer("english", maxCacheSize=0)

_fold_to_ascii = dict(
    [(ord(x), ord(y)) for x, y in zip("\u2018\u2019\u00b4\u201c\u201d\u2013-", "'''\"\"--")]
)
_punct_trans = str.maketrans({key: " " for key in string.punctuation})
_all_trans = {**_fold_to_ascii, **_punct_trans}


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    text = text.translate(_all_trans).replace("'", " ")
    return _stemmer.stemWords(text.lower().split())


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

# Column weights for BM25 scoring
TITLE_BOOST = 10.0
DESCRIPTION_BOOST = 1.0


def build_index(products: pd.DataFrame) -> pd.DataFrame:
    """Add searcharray BM25 index columns to a products DataFrame.

    Expects columns: product_id, title, description
    Returns the same DataFrame with added index columns (in-place copy).
    """
    index = products.copy()
    print("  Indexing titles...")
    index["title_idx"] = SearchArray.index(index["title"], tokenize)
    print("  Indexing descriptions...")
    index["description_idx"] = SearchArray.index(index["description"], tokenize)
    return index


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(query: str, index: pd.DataFrame, k: int = 10) -> list[dict]:
    """BM25 keyword search over an indexed product DataFrame.

    Args:
        query: keyword query string
        index: DataFrame with searcharray index columns (from build_index)
        k: number of results to return

    Returns:
        List of dicts with keys: product_id, title, score — ordered by score desc.
    """
    tokens = tokenize(query)
    if not tokens:
        return []

    scores = np.zeros(len(index))
    for token in tokens:
        scores += index["title_idx"].array.score(token) * TITLE_BOOST
        scores += index["description_idx"].array.score(token) * DESCRIPTION_BOOST

    top_k_idx = np.argsort(-scores)[:k]
    results = []
    for idx in top_k_idx:
        if scores[idx] == 0.0:
            break
        row = index.iloc[idx]
        results.append({
            "product_id": int(row["product_id"]),
            "title": row["title"],
            "description": row["description"],
            "category": row["category"],
            "score": float(scores[idx]),
        })
    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading WANDS products...")
    products = load_products()
    print(f"Loaded {len(products):,} products.\n")

    print("Building BM25 index...")
    index = build_index(products)
    print("Index ready.\n")

    demo_queries = [
        "blue sectional sofa",
        "outdoor dining table",
        "king size bed frame",
        "modern floor lamp",
    ]

    for query in demo_queries:
        results = search(query, index, k=5)
        print(f"Query: '{query}'")
        for r in results:
            print(f"  [{r['score']:6.2f}] {r}")
        print()
