# PRD: Agentic Search Demo

## Overview

A Python project demonstrating how to convert a conventional keyword search function into an agentic search function, evaluated against the Wayfair WANDS dataset using NDCG.

## Goals

- Show a clear, concrete before/after: plain search vs. agentic search
- Both implementations share the same external API so they can be evaluated identically
- Keep things simple and readable — this is a demo/teaching artifact

## Dataset

**WANDS** (Wayfair ANnotated Dataset for Search)
- Product catalog with titles, descriptions, and categories
- Queries with human relevance judgments (Exact / Partial / Irrelevant)
- Pulled down as part of project setup (not committed to the repo)

## Architecture

### Search API (shared interface)

Both the normal search and the agentic search conform to the same interface:

- **Input:** a keyword query string
- **Output:** an ordered list of search results (product ID + score or rank)

This shared interface is the key design constraint — it allows both implementations to plug into the same NDCG evaluation harness.

### 1. Normal Search

A conventional BM25-based keyword search over the WANDS product catalog.

- Indexes product fields (title, description, category)
- Takes a keyword query, returns ranked results
- Serves as the baseline and also as a tool for the agent

### 2. Agentic Search

Wraps the normal search behind an LLM agent loop.

- Externally: same API as normal search (keyword in, ranked results out)
- Internally:
  - The agent receives the original query
  - Has access to a tool that wraps the normal search function
  - Runs an agentic loop: issues keyword searches, inspects results, refines queries
  - Synthesizes a final ranked result list from what it finds
- The agent is free to issue multiple searches, rewrite queries, filter, rerank, etc.

### 3. NDCG Evaluation Harness

- Loads WANDS queries and relevance judgments
- Runs a search function (normal or agentic) against each query
- Computes NDCG@k for each query, reports mean NDCG across the query set
- Accepts any callable that matches the search API interface

## Tech Stack

- **Language:** Python
- **Package / env management:** UV
- **Search:** BM25 (likely via `searcharray` or similar)
- **Agent:** Claude API (Anthropic SDK) or OpenAI — TBD
- **Evaluation:** NDCG computed from WANDS relevance judgments

## Repo Structure (planned)

```
.
├── product/
│   └── PRD.md
├── data/               # gitignored — downloaded WANDS files live here
├── src/
│   └── search_agent/
│       ├── data.py         # WANDS dataset loading
│       ├── search.py       # normal BM25 search
│       ├── agent.py        # agentic search (same API as search.py)
│       └── evaluate.py     # NDCG harness
├── scripts/
│   └── download_wands.py   # one-time data download
├── pyproject.toml          # UV project config
├── uv.lock
└── .gitignore
```

## Success Criteria

- Normal search baseline NDCG established on WANDS
- Agentic search measurably improves NDCG over baseline
- Both plug into the same eval harness with no code changes to the harness
- Code is clean enough to use as a teaching/demo artifact
