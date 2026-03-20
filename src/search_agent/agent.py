"""Agentic search over the WANDS product catalog.

The agent has the same external API as search.search() but internally runs
an LLM loop that can issue multiple BM25 searches before returning a ranked
list of product IDs as structured output.

Each call to agent_search() appends a full trace (system prompt, user query,
all tool calls and responses) as a single JSON line to logs/agent_trace.jsonl.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

load_dotenv()
from agents import Agent, Runner, function_tool
from agents.items import ToolCallItem, ToolCallOutputItem, MessageOutputItem
from pydantic import BaseModel

from search_agent.search import search as bm25_search

LOGS_DIR = Path(__file__).parent.parent.parent / "logs"

# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class ProductRanking(BaseModel):
    """Ordered list of the most relevant product IDs, best first."""
    product_ids: list[int]


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a product search assistant. Given a user query, use the search tool \
to find the most relevant products. You may search multiple times with \
different keywords. When you are satisfied, return the product IDs of the \
best matches in order from most to least relevant.

Always make a single search first, and then based upon what you learned make a parallel search for the best next candidates.
"""


def make_agent(index: pd.DataFrame) -> Agent:
    """Build a search agent with a BM25 search tool backed by the given index."""

    @function_tool
    def search_products(query: str) -> str:
        """Search the product catalog for items matching a keyword query.

        Args:
            query: Keywords to search for.
        """
        results = bm25_search(query, index, k=10)
        if not results:
            return "No results found."
        lines = [
            f"product_id={r['product_id']} | {r['title']} | {r['category']}"
            for r in results
        ]
        return "\n".join(lines)

    return Agent(
        name="ProductSearchAgent",
        instructions=SYSTEM_PROMPT,
        tools=[search_products],
        output_type=ProductRanking,
    )


# ---------------------------------------------------------------------------
# Trace logging
# ---------------------------------------------------------------------------

def _serialize(obj) -> object:
    """Best-effort serialization of an OpenAI SDK object to a plain dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return repr(obj)


def _serialize_item(item) -> dict:
    """Serialize a RunItem into a loggable dict with type + payload."""
    entry = {"type": getattr(item, "type", type(item).__name__)}
    raw = getattr(item, "raw_item", item)
    entry["raw"] = _serialize(raw)
    return entry


def _log_trace(query: str, agent: Agent, result) -> None:
    """Append a single JSON line containing the full trace to the log file."""
    LOGS_DIR.mkdir(exist_ok=True)

    trace = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "system": agent.instructions,
        "messages": [{"role": "user", "content": query}]
        + [_serialize_item(item) for item in result.new_items],
        "output": _serialize(result.final_output),
    }

    log_file = LOGS_DIR / "agent_trace.jsonl"
    with log_file.open("a") as f:
        f.write(json.dumps(trace) + "\n")


def _print_trace(query: str, result) -> None:
    """Print the agent trace to stdout in a readable format."""
    print(f"[trace] query: {query!r}")
    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            raw = item.raw_item
            args = getattr(raw, "arguments", "")
            name = getattr(raw, "name", "tool")
            print(f"  [tool_call] {name}({args})")
        elif isinstance(item, ToolCallOutputItem):
            output = getattr(item.raw_item, "output", repr(item.raw_item))
            # Truncate long tool outputs for readability
            preview = output[:200].replace("\n", " | ")
            print(f"  [tool_result] {preview}{'...' if len(output) > 200 else ''}")
        elif isinstance(item, MessageOutputItem):
            raw = item.raw_item
            role = getattr(raw, "role", "assistant")
            content = getattr(raw, "content", "")
            if content:
                preview = str(content)[:120].replace("\n", " ")
                print(f"  [{role}] {preview}")
    print(f"  [output] {result.final_output}")


# ---------------------------------------------------------------------------
# Agentic search — same API as search.search()
# ---------------------------------------------------------------------------

def agent_search(query: str, index: pd.DataFrame, k: int = 10) -> list[dict]:
    """Agentic BM25 search with the same return format as search.search().

    The agent may issue multiple searches internally before returning a ranked
    list of product IDs as structured output.  The full trace is logged to
    logs/agent_trace.jsonl.

    Args:
        query: keyword query string
        index: DataFrame with searcharray index columns (from build_index)
        k:     maximum number of results to return

    Returns:
        List of dicts with keys: product_id, title, description, category, score
        ordered by agent-ranked relevance.
    """
    agent = make_agent(index)
    result = Runner.run_sync(agent, query)
    ranking: ProductRanking = result.final_output_as(ProductRanking)

    _print_trace(query, result)
    _log_trace(query, agent, result)

    id_to_row = index.set_index("product_id")
    results = []
    for rank, pid in enumerate(ranking.product_ids[:k], start=1):
        if pid not in id_to_row.index:
            continue
        row = id_to_row.loc[pid]
        results.append({
            "product_id": int(pid),
            "title": row["title"],
            "description": row["description"],
            "category": row["category"],
            "score": float(k - rank + 1),  # rank-derived score, best=k
        })
    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from search_agent.data import load_products
    from search_agent.search import build_index

    print("Loading WANDS products...")
    products = load_products()
    print(f"Loaded {len(products):,} products.\n")

    print("Building BM25 index...")
    index = build_index(products)
    print("Index ready.\n")

    query = "desk lamp"
    print(f"Query: '{query}'\n")
    results = agent_search(query, index, k=5)
    print("\nFinal results:")

    for r in results:
        print(f"  [{r['score']:.0f}] {r['title']}")
