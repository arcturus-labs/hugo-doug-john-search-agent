"""NDCG evaluation harness for search functions over the WANDS dataset.

Usage:
    uv run python -m search_agent.evaluate [--num-queries N] [--seed S] [--k K]
"""

import argparse

import numpy as np
import pandas as pd

from search_agent.data import load_judgments, load_products, load_queries
from search_agent.search import build_index, search


# ---------------------------------------------------------------------------
# NDCG utilities
# ---------------------------------------------------------------------------

def idcg_max(max_grade=2, k=10):
    """IDCG assuming max label at each location is max_grade."""
    rank_discounts = 1 / np.log2(2 ** np.arange(1, k + 1))
    numerator = (2**max_grade) - 1
    gains = rank_discounts * numerator
    return np.sum(gains)


def grade_results(
    judgments: pd.DataFrame, search_results: pd.DataFrame, max_grade=None, k=10
) -> pd.DataFrame:
    """Grade search results based on the labeled queries."""
    search_results = search_results[search_results["rank"] <= k]
    assert "doc_id" in judgments.columns, "judgments must have a 'doc_id' column"
    assert "doc_id" in search_results.columns, (
        "search_results must have a 'doc_id' column"
    )
    if not max_grade:
        max_grade = judgments["grade"].max()
    graded_results = search_results.merge(
        judgments[["query_id", "query", "doc_id", "grade"]],
        on=["query_id", "query", "doc_id"],
        how="left",
    )
    graded_results["grade"] = graded_results["grade"].fillna(0)
    rank_discounts = 1 / np.log2(2 ** graded_results["rank"])
    graded_results["discounted_gain"] = (
        (2 ** graded_results["grade"]) - 1
    ) * rank_discounts
    graded_results["idcg"] = idcg_max(max_grade=max_grade, k=k)
    return graded_results


def reciprocal_rank(graded_results: pd.DataFrame, max_grade: int) -> pd.DataFrame:
    """Compute reciprocal rank per query for the max grade."""
    if graded_results.empty:
        return pd.DataFrame(columns=["query", "query_id", "mrr"])

    hits = graded_results[graded_results["grade"] == max_grade]
    min_ranks = hits.groupby(["query", "query_id"])["rank"].min()
    rr = 1 / min_ranks
    rr = rr.rename("mrr")

    all_queries = graded_results[["query", "query_id"]].drop_duplicates()
    rr = rr.reindex(all_queries.set_index(["query", "query_id"]).index, fill_value=0)
    return rr.reset_index()


def ndcg_per_query(graded_results: pd.DataFrame) -> pd.DataFrame:
    """Compute NDCG per query from graded results."""
    return (
        graded_results
        .groupby(["query_id", "query"])
        .apply(
            lambda g: g["discounted_gain"].sum() / g["idcg"].iloc[0],
            include_groups=False,
        )
        .reset_index(name="ndcg")
    )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(search_fn, queries: pd.DataFrame, judgments: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Run search_fn over queries and return per-query NDCG scores.

    Args:
        search_fn: callable(query: str, k: int) -> list[dict] with 'product_id' key
        queries:   DataFrame with columns query_id, query
        judgments: DataFrame with columns query_id, product_id, grade
        k:         depth at which to evaluate

    Returns:
        DataFrame with columns query_id, query, ndcg
    """
    # Build judgments table with query text and doc_id column
    judg = judgments.merge(queries[["query_id", "query"]], on="query_id")
    judg = judg.rename(columns={"product_id": "doc_id"})

    # Run each query and collect ranked results
    rows = []
    for _, qrow in queries.iterrows():
        results = search_fn(qrow["query"], k=k)
        for rank, r in enumerate(results, start=1):
            rows.append({
                "query_id": qrow["query_id"],
                "query": qrow["query"],
                "doc_id": r["product_id"],
                "rank": rank,
            })

    results_df = pd.DataFrame(rows)
    graded = grade_results(judg, results_df, k=k)
    return ndcg_per_query(graded)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BM25 search with NDCG on WANDS.")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of queries to sample (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for query sampling (default: 42)")
    parser.add_argument("--k", type=int, default=10, help="Ranking depth for NDCG (default: 10)")
    args = parser.parse_args()

    print("Loading data...")
    products = load_products()
    all_queries = load_queries()
    judgments = load_judgments()

    sampled_queries = all_queries.sample(n=args.num_queries, random_state=args.seed).reset_index(drop=True)

    print(f"Building BM25 index over {len(products):,} products...")
    index = build_index(products)

    def search_fn(query: str, k: int = 10):
        return search(query, index, k=k)

    print(f"\nEvaluating {args.num_queries} queries (seed={args.seed}, k={args.k})...\n")
    scores = evaluate(search_fn, sampled_queries, judgments, k=args.k)

    for _, row in scores.iterrows():
        print(f"  {row['ndcg']:.3f}  {row['query']}")

    print(f"\nMean NDCG@{args.k}: {scores['ndcg'].mean():.4f}")
