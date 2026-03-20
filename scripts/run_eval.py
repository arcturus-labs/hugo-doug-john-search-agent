#!/usr/bin/env python3
"""Evaluate plain BM25 or agentic search on WANDS using NDCG.

Builds the index once, runs queries in parallel, then reports per-query
and mean NDCG@k.

Usage:
    uv run python scripts/run_eval.py --type plain
    uv run python scripts/run_eval.py --type agent
    uv run python scripts/run_eval.py --type agent --num-queries 5 --workers 3 --seed 7
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from search_agent.data import load_judgments, load_products, load_queries
from search_agent.evaluate import grade_results, ndcg_per_query
from search_agent.search import build_index
from search_agent.search import search as bm25_search


def run_parallel(
    search_fn,
    queries: pd.DataFrame,
    judgments: pd.DataFrame,
    k: int = 10,
    workers: int = 10,
) -> pd.DataFrame:
    """Run search_fn over queries in parallel and return per-query NDCG scores."""
    judg = judgments.merge(queries[["query_id", "query"]], on="query_id")
    judg = judg.rename(columns={"product_id": "doc_id"})

    all_rows = []

    def run_query(qrow):
        results = search_fn(qrow["query"], k=k)
        return [
            {
                "query_id": qrow["query_id"],
                "query": qrow["query"],
                "doc_id": r["product_id"],
                "rank": rank,
            }
            for rank, r in enumerate(results, start=1)
        ]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_query, row): row["query"]
            for _, row in queries.iterrows()
        }
        for future in as_completed(futures):
            query_text = futures[future]
            try:
                all_rows.extend(future.result())
                print(f"  done: {query_text!r}")
            except Exception as e:
                print(f"\nFatal error on {query_text!r}: {e}", file=sys.stderr)
                executor.shutdown(cancel_futures=True)
                sys.exit(1)

    results_df = pd.DataFrame(all_rows)
    graded = grade_results(judg, results_df, k=k)
    return ndcg_per_query(graded)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 or agentic search with NDCG on WANDS."
    )
    parser.add_argument(
        "--type",
        choices=["plain", "agent"],
        default="plain",
        help="Search type: 'plain' (BM25) or 'agent' (LLM agentic). Default: plain",
    )
    parser.add_argument(
        "--num-queries", type=int, default=10,
        help="Number of queries to sample. Default: 10",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for query sampling. Default: 42",
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Ranking depth for NDCG. Default: 10",
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Number of parallel workers. Default: 10",
    )
    args = parser.parse_args()

    print("Loading data...")
    products = load_products()
    all_queries = load_queries()
    judgments = load_judgments()

    sampled = (
        all_queries
        .sample(n=args.num_queries, random_state=args.seed)
        .reset_index(drop=True)
    )

    print(f"Building BM25 index over {len(products):,} products...")
    index = build_index(products)
    print("Index ready.\n")

    if args.type == "plain":
        def search_fn(query: str, k: int = 10):
            return bm25_search(query, index, k=k)
    else:
        from search_agent.agent import agent_search
        def search_fn(query: str, k: int = 10):
            return agent_search(query, index, k=k)

    print(
        f"Running {args.type} search on {args.num_queries} queries "
        f"(seed={args.seed}, k={args.k}, workers={args.workers})...\n"
    )

    scores = run_parallel(search_fn, sampled, judgments, k=args.k, workers=args.workers)

    print(f"\nNDCG@{args.k} per query:")
    for _, row in scores.sort_values("ndcg", ascending=False).iterrows():
        print(f"  {row['ndcg']:.3f}  {row['query']}")

    print(f"\nMean NDCG@{args.k}: {scores['ndcg'].mean():.4f}")


if __name__ == "__main__":
    main()
