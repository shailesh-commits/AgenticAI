"""Agent 1 — Lead Sourcer.

Responsibilities
----------------
* Scrape MoneyControl + Google News (or use mock fixtures).
* Deduplicate articles seen within the last 48 hours.
* Score each article: recency + entity_frequency + sentiment.
* Extract structured LeadSignal records.
* Persist to the state store and emit LEAD_FEED_READY.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

from orchestrator.state import LeadSignal
from orchestrator.state_store import StateStore
from scrapers.google_news import GoogleNewsFetcher
from scrapers.moneycontrol import MoneyControlScraper

logger = logging.getLogger(__name__)

# ── Keyword → trigger-type mapping ───────────────────────────────────────────
_TRIGGER_PATTERNS: list[tuple[str, str]] = [
    (r"\b(earnings|profit|revenue|PAT|EBITDA|Q[1-4]\s*FY|results)\b", "earnings"),
    (r"\b(acqui|merger|M&A|takeover|buy|deal|contract|wins|order)\b", "M&A"),
    (r"\b(upgrade|downgrade|target price|rating|outperform|buy|sell|hold)\b", "rating_change"),
    (r"\b(GDP|inflation|RBI|Fed|interest rate|policy|macro|budget|GST)\b", "macro"),
    (r"\b(sector|index|Nifty|Sensex|market|rally|correction)\b", "sector_move"),
]

# ── Simple positive/negative word lists ──────────────────────────────────────
_POSITIVE_WORDS = {
    "surge", "jump", "rise", "gain", "profit", "growth", "upgrade", "record",
    "strong", "beats", "up", "outperform", "win", "wins", "positive", "approval",
    "nod", "milestone", "expands", "raises", "inflow",
}
_NEGATIVE_WORDS = {
    "fall", "drop", "decline", "loss", "cuts", "narrows", "downgrade", "miss",
    "weak", "layoff", "reduction", "sell", "concern", "risk", "negative",
    "lower", "cut", "slash",
}


class LeadSourcerAgent:
    """Scrapes, deduplicates, scores, and persists financial leads."""

    def __init__(self, config: dict, state_store: StateStore, selectors: dict | None = None):
        self.config = config
        self.state_store = state_store
        self.tickers: list[str] = config.get("watchlist", {}).get("tickers", [])
        self.dedup_window: int = config.get("lead_sourcer", {}).get("deduplication_window_hours", 48)
        self.max_leads: int = config.get("lead_sourcer", {}).get("max_leads_per_cycle", 50)
        self.min_score: float = config.get("lead_sourcer", {}).get("min_relevance_score", 0.3)
        self.mock_mode: bool = config.get("scraping", {}).get("mock_mode", True)

        selectors = selectors or {}
        self.mc_scraper = MoneyControlScraper(config, selectors)
        self.gn_fetcher = GoogleNewsFetcher(config)

        # Load mock leads once so they can be injected in tests / mock mode
        _mock_path = os.path.join(os.path.dirname(__file__), "..", "data", "mock_leads.json")
        with open(os.path.normpath(_mock_path)) as f:
            self._mock_leads: list[dict] = json.load(f)

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self) -> list[LeadSignal]:
        """Execute one scrape cycle. Returns a list of LeadSignal dicts."""
        logger.info("LeadSourcerAgent: starting scrape cycle")

        if self.mock_mode:
            leads = self._mock_leads[:]
            logger.info("LeadSourcerAgent: mock_mode ON — using %d pre-loaded leads", len(leads))
            return [self._ensure_lead_signal(l) for l in leads]

        raw_articles = self._collect_raw_articles()
        seen_urls = self.state_store.get_recent_urls(hours=self.dedup_window)
        deduped = self._deduplicate(raw_articles, seen_urls)
        signals = self._process_articles(deduped)
        signals.sort(key=lambda s: s["score"], reverse=True)
        signals = [s for s in signals if s["score"] >= self.min_score]
        signals = signals[: self.max_leads]

        logger.info("LeadSourcerAgent: cycle complete — %d leads produced", len(signals))
        return signals

    # ── Internal pipeline ─────────────────────────────────────────────────────

    def _collect_raw_articles(self) -> list[dict]:
        mc_articles = self.mc_scraper.fetch()
        gn_articles = self.gn_fetcher.fetch(self.tickers)
        all_articles = mc_articles + gn_articles
        logger.debug("Raw articles: MC=%d GN=%d total=%d", len(mc_articles), len(gn_articles), len(all_articles))
        return all_articles

    def _deduplicate(self, articles: list[dict], seen_urls: set) -> list[dict]:
        unique: dict[str, dict] = {}
        for art in articles:
            url = art.get("url", "")
            if not url or url in seen_urls:
                continue
            if url not in unique:
                unique[url] = art
        logger.debug("Deduplication: %d → %d articles", len(articles), len(unique))
        return list(unique.values())

    def _process_articles(self, articles: list[dict]) -> list[LeadSignal]:
        signals: list[LeadSignal] = []
        for art in articles:
            ticker = art.get("ticker") or self._infer_ticker(art.get("headline", ""))
            if not ticker:
                continue
            trigger = art.get("trigger") or self._detect_trigger(art.get("headline", ""))
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

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_article(self, article: dict, sentiment: str) -> float:
        recency = self._recency_score(article.get("ts", ""))
        entity_freq = self._entity_frequency(article.get("headline", ""))
        sentiment_boost = 0.2 if sentiment != "neutral" else 0.0
        return min(1.0, recency * 0.4 + entity_freq * 0.4 + sentiment_boost)

    @staticmethod
    def _recency_score(ts_str: str) -> float:
        if not ts_str:
            return 0.5
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            age_hours = (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds() / 3600
            return max(0.0, 1.0 - age_hours / 48)
        except Exception:
            return 0.5

    def _entity_frequency(self, headline: str) -> float:
        headline_upper = headline.upper()
        matches = sum(1 for t in self.tickers if t in headline_upper)
        return min(1.0, matches * 0.5)

    # ── Classification ────────────────────────────────────────────────────────

    @staticmethod
    def _detect_trigger(headline: str) -> str:
        for pattern, trigger_type in _TRIGGER_PATTERNS:
            if re.search(pattern, headline, re.IGNORECASE):
                return trigger_type
        return "macro"

    @staticmethod
    def _analyze_sentiment(text: str) -> str:
        lower = text.lower()
        words = set(re.findall(r"\b\w+\b", lower))
        pos = len(words & _POSITIVE_WORDS)
        neg = len(words & _NEGATIVE_WORDS)
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"

    def _infer_ticker(self, headline: str) -> str:
        headline_upper = headline.upper()
        for ticker in self.tickers:
            if ticker in headline_upper:
                return ticker
        return ""

    @staticmethod
    def _ensure_lead_signal(lead: dict) -> LeadSignal:
        return LeadSignal(
            ticker=lead.get("ticker", ""),
            headline=lead.get("headline", ""),
            trigger=lead.get("trigger", "macro"),
            sentiment=lead.get("sentiment", "neutral"),
            score=float(lead.get("score", 0.5)),
            url=lead.get("url", ""),
            ts=lead.get("ts", datetime.now(timezone.utc).isoformat()),
            source=lead.get("source", "mock"),
        )
