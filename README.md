# Agentic Search Demo

A Python project showing how to convert a conventional BM25 keyword search into an LLM-driven agentic search, evaluated with NDCG against the [Wayfair WANDS dataset](https://github.com/wayfair/WANDS).

Built for a live talk by John Berryman, Doug Turnbull ([softwaredoug](https://github.com/softwaredoug)), and Hugo.

## Big lessons

1. It's (probably) easy to wrap a search API in an agent and get better results
2. But you need evals to make sure you're moving in the right direction

## How it works

Both plain search and agentic search share the same external interface — keyword query in, ranked product list out — so they plug into the same NDCG evaluation harness with no changes.

### Plain search

BM25 over WANDS product titles and descriptions using [`searcharray`](https://github.com/softwaredoug/searcharray) with a Snowball stemmer. Serves as the baseline and also as the tool the agent calls.

### Advanced search

An enhanced BM25 baseline that also indexes the product category field using a hierarchical tokenizer. A category path like `Furniture/Living Room/Sofas` is indexed as the cumulative prefixes `["Furniture", "Furniture/Living Room", "Furniture/Living Room/Sofas"]`, so queries at any level of the hierarchy match all nested products. The search function accepts separate `title_query`, `description_query`, and `category_filter` arguments, making it the intended backing search API for a more capable agentic search that can target individual fields.

### Agentic search

Wraps BM25 behind an LLM loop (OpenAI Agents SDK). The agent receives the original query, issues one or more keyword searches, inspects results, refines queries, and returns a final ranked list as structured output. All traces are logged to `logs/agent_trace.jsonl`.

### Evaluation

NDCG@k computed against WANDS human relevance judgments (Exact=2, Partial=1, Irrelevant=0).

## Setup

Requires Python 3.11+ and [UV](https://docs.astral.sh/uv/).

```bash
uv sync
```

Copy `.env.example` to `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

The WANDS dataset is downloaded automatically on first use (gitignored).

## Running search

```bash
# Plain BM25
uv run python -m search_agent.search

# Advanced BM25 (hierarchical category index)
uv run python -m search_agent.advanced_search

# Agentic search
uv run python -m search_agent.agent
```

## Running evals

```bash
# Baseline: plain BM25
uv run python scripts/run_eval.py --type plain

# Advanced BM25
uv run python scripts/run_eval.py --type advanced

# Agentic search
uv run python scripts/run_eval.py --type agent

# Options
uv run python scripts/run_eval.py --type agent --num-queries 20 --workers 5 --seed 7 --k 10
```

| Flag | Default | Description |
|---|---|---|
| `--type` | `plain` | `plain` (BM25), `advanced` (hierarchical BM25), or `agent` (LLM) |
| `--num-queries` | `10` | Number of queries to sample |
| `--seed` | `42` | Random seed for query sampling |
| `--k` | `10` | Ranking depth for NDCG |
| `--workers` | `10` | Parallel workers |

Queries run in parallel. A fatal error on any query prints to stderr and exits immediately.

## Inspecting agent traces

Each `agent_search()` call appends a JSON line to `logs/agent_trace.jsonl`. Use `print_traces.py` to read them:

```bash
# All traces, all messages
uv run python scripts/print_traces.py logs/agent_trace.jsonl .

# Trace 0 only
uv run python scripts/print_traces.py logs/agent_trace.jsonl 0.

# Trace 0, message 2 only
uv run python scripts/print_traces.py logs/agent_trace.jsonl 0.2

# Show full tool outputs and system prompt
uv run python scripts/print_traces.py logs/agent_trace.jsonl . --full

# Show only first line of each tool output
uv run python scripts/print_traces.py logs/agent_trace.jsonl . --truncated
```

The selector format is `trace.message` — either part can be omitted to mean "all".

## Repo structure

```
.
├── src/search_agent/
│   ├── data.py         # WANDS loader (auto-downloads on first use)
│   ├── search.py          # plain BM25 search
│   ├── advanced_search.py # BM25 with hierarchical category index
│   ├── agent.py           # agentic search (same API as search.py)
│   └── evaluate.py     # NDCG harness
├── scripts/
│   ├── run_eval.py     # parallel eval runner
│   └── print_traces.py # agent trace viewer
├── product/PRD.md
├── pyproject.toml
└── .gitignore
```

## Talk outline

- **Goals and big lessons** — evals are the key; agents can help without them you're flying blind
- **Tour the pieces** — search API, NDCG evaluator, trace viewer
- **Live: build the agent** — wrap BM25 as a tool, simple system prompt, structured output, trace logging
- **Evaluate** — explain NDCG, run baseline, run agent, compare
- **Debug with traces** — use `print_traces.py` to find problems and improve the system prompt
- **Hard mode** *(if time)* — field-specific search with name/description/category + query syntax; tighter eval loop
- **"Ralph it"** — close with the vision: an automated outer loop powered by Claude Code that runs the full optimize cycle end-to-end
