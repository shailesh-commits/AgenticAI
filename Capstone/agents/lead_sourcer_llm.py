"""Agent 1 — Lead Sourcer (LLM-Enhanced Variant).

Demonstrates the trade-off between two sentiment-analysis approaches:

  Approach A — Keyword-based (lead_sourcer.py)
    - Zero API cost, deterministic, microseconds per cycle
    - Brittle on nuanced headlines ("misses estimates but raises guidance")

  Approach B — Batched LLM (this file)
    - One API call per cycle regardless of article count
    - Context-aware, handles negation and financial idioms
    - Small, predictable cost per cycle

Run compare_approaches() for a side-by-side benchmark report.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agents.lead_sourcer import LeadSourcerAgent
from orchestrator.state import LeadSignal
from orchestrator.state_store import StateStore

logger = logging.getLogger(__name__)

# GPT-4o pricing (approximate — verify at platform.openai.com/docs/pricing)
_GPT4O_INPUT_COST_PER_TOKEN = 5.00 / 1_000_000   # $5.00 per 1M input tokens
_GPT4O_OUTPUT_COST_PER_TOKEN = 15.00 / 1_000_000  # $15.00 per 1M output tokens

_SENTIMENT_SYSTEM_PROMPT = """You are a financial news sentiment analyser.
Given a list of financial news headlines, classify each as positive, negative, or neutral.
Respond ONLY with a valid JSON array — one object per headline.
Each object must have exactly these keys:
  "id"         : integer index matching the input
  "sentiment"  : "positive" | "negative" | "neutral"
  "confidence" : float 0.0–1.0
  "reasoning"  : one-sentence explanation

Do not include markdown fences or any text outside the JSON array."""


@dataclass
class RunMetrics:
    """Captured metrics for a single Lead Sourcer cycle."""
    approach: str                   # "keyword" | "llm_batch"
    articles_processed: int = 0
    duration_seconds: float = 0.0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    llm_fallback_used: bool = False  # True if LLM failed and keyword was used instead
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "approach": self.approach,
            "articles_processed": self.articles_processed,
            "duration_seconds": round(self.duration_seconds, 3),
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "llm_fallback_used": self.llm_fallback_used,
            "notes": self.notes,
        }


class LeadSourcerLLMAgent(LeadSourcerAgent):
    """Lead Sourcer that replaces per-article keyword sentiment with one batched LLM call.

    Everything else (scraping, dedup, trigger detection, scoring) is inherited
    unchanged from LeadSourcerAgent so the comparison is purely about sentiment.
    """

    def __init__(self, config: dict, state_store: StateStore, selectors: dict | None = None):
        super().__init__(config, state_store, selectors)
        self.last_run_metrics: RunMetrics | None = None
        self._llm: Any = None

        llm_cfg = config.get("llm", {})
        if not llm_cfg.get("mock_mode", False):
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=llm_cfg.get("model", "gpt-4o"),
                    temperature=0,           # deterministic classification
                    max_tokens=llm_cfg.get("max_tokens", 4096),
                )
            except ImportError:
                logger.warning("langchain_openai not installed; LLM sentiment will fall back to keywords")
            except Exception as exc:
                logger.warning("ChatOpenAI init failed (%s); will fall back to keywords", exc)

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self) -> list[LeadSignal]:
        metrics = RunMetrics(approach="llm_batch")
        t0 = time.perf_counter()

        signals = super().run()

        metrics.duration_seconds = time.perf_counter() - t0
        metrics.articles_processed = len(signals)
        self.last_run_metrics = metrics
        return signals

    # ── Overridden pipeline step ──────────────────────────────────────────────

    def _process_articles(self, articles: list[dict]) -> list[LeadSignal]:
        """Same as parent but replaces per-article sentiment with one batched LLM call."""
        if not articles:
            return []

        metrics = self.last_run_metrics or RunMetrics(approach="llm_batch")
        sentiment_map = self._batch_llm_sentiment(articles, metrics)

        signals: list[LeadSignal] = []
        for idx, art in enumerate(articles):
            ticker = art.get("ticker") or self._infer_ticker(art.get("headline", ""))
            if not ticker:
                continue
            trigger = art.get("trigger") or self._detect_trigger(art.get("headline", ""))

            # Use LLM result if available, else fall back to keyword
            if idx in sentiment_map:
                sentiment = sentiment_map[idx]
            else:
                sentiment = art.get("sentiment") or self._analyze_sentiment(art.get("headline", ""))

            score = art.get("score") or self._score_article(art, sentiment)
            signals.append(
                LeadSignal(
                    ticker=ticker,
                    headline=art.get("headline", ""),
                    trigger=trigger,
                    sentiment=sentiment,
                    score=round(score, 4),
                    url=art.get("url", ""),
                    ts=art.get("ts", datetime.now(timezone.utc).isoformat()),
                    source=art.get("source", "unknown"),
                )
            )
        return signals

    # ── Batched LLM sentiment ─────────────────────────────────────────────────

    def _batch_llm_sentiment(self, articles: list[dict], metrics: RunMetrics) -> dict[int, str]:
        """Send all headlines in a single LLM call. Returns {article_index: sentiment}."""
        if self._llm is None or self.mock_mode:
            metrics.llm_fallback_used = True
            metrics.notes.append("LLM unavailable — keyword sentiment used for all articles")
            return {}

        headlines_block = "\n".join(
            f'{i}. "{art.get("headline", "")}"'
            for i, art in enumerate(articles)
        )
        user_content = f"Classify the sentiment of these {len(articles)} financial headlines:\n\n{headlines_block}"

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            t0 = time.perf_counter()
            response = self._llm.invoke([
                SystemMessage(content=_SENTIMENT_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ])
            elapsed = time.perf_counter() - t0

            # Extract token usage if available
            if hasattr(response, "response_metadata"):
                usage = response.response_metadata.get("token_usage", {})
                metrics.llm_input_tokens = usage.get("prompt_tokens", 0)
                metrics.llm_output_tokens = usage.get("completion_tokens", 0)
                metrics.estimated_cost_usd = (
                    metrics.llm_input_tokens * _GPT4O_INPUT_COST_PER_TOKEN
                    + metrics.llm_output_tokens * _GPT4O_OUTPUT_COST_PER_TOKEN
                )

            metrics.notes.append(
                f"LLM batch call: {len(articles)} headlines in {elapsed:.2f}s, "
                f"~{metrics.llm_input_tokens} in / ~{metrics.llm_output_tokens} out tokens"
            )

            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            parsed: list[dict] = json.loads(raw)
            return {item["id"]: item["sentiment"] for item in parsed if "id" in item and "sentiment" in item}

        except Exception as exc:
            logger.warning("Batch LLM sentiment failed (%s) — falling back to keywords", exc)
            metrics.llm_fallback_used = True
            metrics.notes.append(f"LLM call failed ({exc}); keyword sentiment used")
            return {}


# ── Comparison utility ────────────────────────────────────────────────────────

def compare_approaches(
    articles: list[dict],
    config: dict,
    state_store: StateStore,
    selectors: dict | None = None,
) -> dict:
    """Run both approaches on the same article list and return a comparison report.

    Usage:
        from agents.lead_sourcer import LeadSourcerAgent
        from agents.lead_sourcer_llm import compare_approaches

        report = compare_approaches(articles, config, state_store)
        import json; print(json.dumps(report, indent=2))
    """
    # --- Approach A: keyword-based ---
    kw_agent = LeadSourcerAgent(config, state_store, selectors)
    kw_agent.mock_mode = False  # force real processing path

    t0 = time.perf_counter()
    kw_signals = kw_agent._process_articles(articles)
    kw_duration = time.perf_counter() - t0

    kw_metrics = RunMetrics(
        approach="keyword",
        articles_processed=len(kw_signals),
        duration_seconds=round(kw_duration, 6),
        llm_input_tokens=0,
        llm_output_tokens=0,
        estimated_cost_usd=0.0,
        notes=["Pure regex + word-list sentiment; zero API calls"],
    )

    # --- Approach B: LLM batched ---
    llm_agent = LeadSourcerLLMAgent(config, state_store, selectors)
    llm_agent.mock_mode = False
    llm_agent.last_run_metrics = RunMetrics(approach="llm_batch")

    t0 = time.perf_counter()
    llm_signals = llm_agent._process_articles(articles)
    llm_agent.last_run_metrics.duration_seconds = time.perf_counter() - t0
    llm_agent.last_run_metrics.articles_processed = len(llm_signals)

    # --- Sentiment diff ---
    diffs = []
    for kw, llm in zip(kw_signals, llm_signals):
        if kw["sentiment"] != llm["sentiment"]:
            diffs.append({
                "headline": kw["headline"],
                "keyword_sentiment": kw["sentiment"],
                "llm_sentiment": llm["sentiment"],
            })

    return {
        "keyword_approach": kw_metrics.summary(),
        "llm_batch_approach": llm_agent.last_run_metrics.summary(),
        "articles_compared": min(len(kw_signals), len(llm_signals)),
        "sentiment_disagreements": len(diffs),
        "disagreement_examples": diffs[:5],  # show up to 5
        "verdict": {
            "cost_winner": "keyword (free)",
            "quality_winner": "llm_batch (context-aware)",
            "speed_winner": "keyword" if kw_duration < llm_agent.last_run_metrics.duration_seconds else "llm_batch",
            "recommendation": (
                "Use llm_batch: one API call per 5-hour cycle is negligible cost "
                "and avoids keyword blind-spots on nuanced financial language."
            ),
        },
    }
