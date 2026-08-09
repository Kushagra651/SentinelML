"""
agent/graph.py

LangGraph ReAct agent that answers natural-language questions about the
live SentinelML system.

Flow
────
  user query
      │
      ▼
  [llm_node]  ←─────────────────────┐
      │                              │
      ├─ tool_calls? ──► [tool_node]─┘
      │
      └─ final text ──► return / SSE stream

Model  : claude-haiku-4-5  (cheapest Anthropic model, free-tier friendly)
Reason : already using Anthropic API; no extra vendor; fast enough for
         ReAct loops; open usage policy for OSS projects.

Public API
──────────
  run_agent(question: str) -> str
      Blocking call → returns final answer string.

  stream_agent(question: str) -> Iterator[str]
      Yields SSE-formatted chunks for FastAPI StreamingResponse.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Iterator

from langchain_groq import ChatGroq 
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOLS

log = logging.getLogger(__name__)

# ── Model ─────────────────────────────────────────────────────────────────────

_MODEL_NAME = os.getenv("AGENT_MODEL", "llama-3.1-8b-instant")
_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "1024"))
_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "6"))


def _build_llm() -> ChatGroq:
    return ChatGroq(
        model=_MODEL_NAME,
        max_tokens=_MAX_TOKENS,
        temperature=0,
    )


# ── Graph (built once, reused across requests) ────────────────────────────────

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        llm = _build_llm()
        _graph = create_react_agent(
            model=llm,
            tools=TOOLS,
            prompt=SYSTEM_PROMPT,
        )
    return _graph


# ── Input builder ─────────────────────────────────────────────────────────────


def _build_input(question: str) -> dict:
    return {
        "messages": [
            HumanMessage(content=question),
        ]
    }


# ── Public API ────────────────────────────────────────────────────────────────


def run_agent(question: str) -> str:
    """
    Blocking ReAct loop. Returns the final answer string.
    Called by the non-streaming POST /agent/query route.
    """
    graph = _get_graph()
    config = {"recursion_limit": _MAX_ITERATIONS * 2}

    try:
        result = graph.invoke(_build_input(question), config=config)
        # Last message in the thread is the final AI response
        final = result["messages"][-1]
        return final.content if hasattr(final, "content") else str(final)
    except Exception as exc:
        log.error("Agent run failed: %s", exc)
        return f"Agent error: {exc}"


def stream_agent(question: str) -> Iterator[str]:
    """
    Yields SSE-formatted strings for FastAPI StreamingResponse.

    Format per chunk:
        data: <text_chunk>\n\n

    Final event:
        data: [DONE]\n\n
    """
    graph = _get_graph()
    config = {"recursion_limit": _MAX_ITERATIONS * 2}

    try:
        for event in graph.stream(
            _build_input(question),
            config=config,
            stream_mode="messages",
        ):
            # event is (message_chunk, metadata) when stream_mode="messages"
            if isinstance(event, tuple):
                chunk, meta = event
                # Only stream final AI text, not tool call internals
                if (
                    hasattr(chunk, "content")
                    and chunk.content
                    and meta.get("langgraph_node") == "agent"
                    and not getattr(chunk, "tool_calls", None)
                ):
                    text = chunk.content
                    if isinstance(text, list):
                        # anthropic returns list of content blocks
                        text = " ".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in text
                        )
                    if text:
                        payload = json.dumps({"text": text})
                        yield f"data: {payload}\n\n"
    except Exception as exc:
        log.error("Agent stream failed: %s", exc)
        error_payload = json.dumps({"error": str(exc)})
        yield f"data: {error_payload}\n\n"
    finally:
        yield "data: [DONE]\n\n"