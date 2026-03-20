"""BM25 keyword search with hierarchical category indexing over the WANDS product catalog."""

import json
import string

import numpy as np
import pandas as pd
import Stemmer
from searcharray import SearchArray

from search_agent.data import load_products

# ---------------------------------------------------------------------------
# Tokenizers
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


def taxonomy_tokenizer(text: str) -> list[str]:
    """Hierarchical tokenizer for category paths like 'Furniture/Living Room/Sofas'.

    Emits one token per prefix level so that a query for 'Furniture' matches
    every product in that top-level category, while 'Furniture/Living Room'
    narrows to that sub-category, etc.

    'Furniture/Living Room/Sofas' ->
        ['Furniture', 'Furniture/Living Room', 'Furniture/Living Room/Sofas']
    """
    if not isinstance(text, str) or not text:
        return []
    parts = [p.strip() for p in text.split("/")]
    tokens = []
    current = ""
    for part in parts:
        current += part
        tokens.append(current)
        current += "/"
    return tokens


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

# Column weights for BM25 scoring
TITLE_BOOST = 10.0
DESCRIPTION_BOOST = 1.0
CATEGORY_BOOST = 5.0


def build_index(products: pd.DataFrame) -> pd.DataFrame:
    """Add searcharray BM25 index columns to a products DataFrame.

    Expects columns: product_id, title, description, category
    Returns the same DataFrame with added index columns (in-place copy).
    The category column is indexed hierarchically so queries at any level
    of the path match all products nested beneath it.
    """
    index = products.copy()
    print("  Indexing titles...")
    index["title_idx"] = SearchArray.index(index["title"], tokenize)
    print("  Indexing descriptions...")
    index["description_idx"] = SearchArray.index(index["description"], tokenize)
    print("  Indexing categories (hierarchical)...")
    index["category_idx"] = SearchArray.index(index["category"], taxonomy_tokenizer)
    return index


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def advanced_search(
    index: pd.DataFrame,
    title_query: str | None = None,
    description_query: str | None = None,
    category_filter: str | None = None,
    k: int = 10,
) -> list[dict]:
    """BM25 keyword search over an indexed product DataFrame with per-field queries.

    Args:
        index: DataFrame with searcharray index columns (from build_index)
        title_query: keyword query matched against the title field
        description_query: keyword query matched against the description field
        category_filter: optional category path prefix to hard-restrict results
            (e.g. 'Furniture/Living Room').  Uses hierarchical indexing so the
            filter matches any product nested beneath that path level.
        k: number of results to return

    Returns:
        List of dicts with keys: product_id, title, description, category, score
        — ordered by score desc.
    """
    scores = np.zeros(len(index))

    if title_query:
        for token in tokenize(title_query):
            scores += index["title_idx"].array.score(token) * TITLE_BOOST

    if description_query:
        for token in tokenize(description_query):
            scores += index["description_idx"].array.score(token) * DESCRIPTION_BOOST

    valid_mask = np.ones(len(index), dtype=bool)
    if category_filter:
        filter_scores = index["category_idx"].array.score(category_filter)
        valid_mask = filter_scores > 0
        scores[~valid_mask] = 0.0

    if not scores.any():
        return []

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
            "category": "/".join(p.strip() for p in row["category"].split("/")),
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
        ("blue sectional sofa", None),
        ("outdoor dining table", "Furniture/Outdoor Furniture"),
        ("king size bed frame", "Furniture/Bedroom Furniture"),
        ("modern floor lamp", None),
    ]

    for query, cat_filter in demo_queries:
        results = advanced_search(index, title_query=query, description_query=query, category_filter=cat_filter, k=5)
        filter_str = f" [filter: {cat_filter}]" if cat_filter else ""
        print(f"Query: '{query}'{filter_str}")
        for r in results:
            # print(f"  [{r['score']:6.2f}] ({r['category']}) {r['title']}")
            print(json.dumps(r, indent=2, default=str))
        print()
