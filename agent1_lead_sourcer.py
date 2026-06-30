"""
agents/agent1_lead_sourcer.py
------------------------------
Agent 1 – Lead Sourcer

Responsibilities:
  1. Scrape MoneyControl headlines via the moneycontrol-api PyPI library.
  2. Ingest Google News RSS for watchlist tickers.
  3. Deduplicate articles seen within the prior 48-hour window.
  4. Score and rank leads by recency + entity frequency + sentiment.
  5. Extract structured LeadSignal records and persist to shared state.
  6. Emit LEAD_FEED_READY event to Orchestrator.

Design:
  - LangGraph node function: `run_lead_sourcer(state) -> state`
  - All external calls wrapped in try/except for resilience.
  - Mock mode available for CI/testing (set MOCK_EXTERNAL=true).
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import feedparser
import requests
import yaml

from moneycontrol import moneycontrol_api as mc_api

from state.store import LeadSignal, TriggerType, get_store

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

def _load_config() -> Dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)

CFG = _load_config()

DEDUP_HOURS: int = CFG["moneycontrol"]["dedup_window_hours"]
WATCHLIST: List[str] = CFG["google_news"]["watchlist"]
RSS_TEMPLATE: str = CFG["google_news"]["rss_template"]
MOCK_MODE: bool = os.getenv("MOCK_EXTERNAL", "false").lower() == "true"

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _article_hash(headline: str, url: str = "") -> str:
    """Stable dedup key from headline + url."""
    return hashlib.sha1(f"{headline.strip().lower()}{url}".encode()).hexdigest()


def _is_duplicate(article_hash: str, store) -> bool:
    """Return True if this article was seen within the dedup window."""
    key = f"dedup:{article_hash}"
    record = store.get(key)
    if record is None:
        return False
    seen_at = datetime.fromisoformat(record["ts"])
    cutoff = datetime.now(timezone.utc) - timedelta(hours=DEDUP_HOURS)
    return seen_at > cutoff


def _mark_seen(article_hash: str, store) -> None:
    store.set(f"dedup:{article_hash}", {"ts": datetime.now(timezone.utc).isoformat()})


_TRIGGER_KEYWORDS: Dict[TriggerType, List[str]] = {
    TriggerType.EARNINGS:      ["earnings", "profit", "revenue", "EPS", "results", "quarterly"],
    TriggerType.MA:            ["acquisition", "merger", "takeover", "buyout", "deal"],
    TriggerType.RATING_CHANGE: ["upgrade", "downgrade", "rating", "target price", "buy", "sell", "hold"],
    TriggerType.MACRO:         ["RBI", "Fed", "inflation", "GDP", "interest rate", "policy"],
}

def _classify_trigger(headline: str) -> TriggerType:
    hl = headline.lower()
    for trigger, keywords in _TRIGGER_KEYWORDS.items():
        if any(kw.lower() in hl for kw in keywords):
            return trigger
    return TriggerType.GENERAL


_POSITIVE = ["surge", "gain", "rise", "up", "growth", "profit", "beat", "strong", "bullish", "upgrade"]
_NEGATIVE = ["fall", "drop", "loss", "down", "decline", "miss", "weak", "bearish", "downgrade", "crash"]

def _simple_sentiment(headline: str) -> float:
    """Lightweight sentiment: count positive vs negative signal words."""
    hl = headline.lower()
    pos = sum(1 for w in _POSITIVE if w in hl)
    neg = sum(1 for w in _NEGATIVE if w in hl)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 2)


def _relevance_score(headline: str, ticker: str, ts: datetime) -> float:
    """
    Composite score = 0.4 * recency + 0.3 * entity_match + 0.3 * |sentiment|
    Recency decays linearly over 24 hours.
    """
    age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
    recency = max(0.0, 1.0 - age_hours / 24)
    entity = 1.0 if ticker.lower() in headline.lower() else 0.3
    sentiment_strength = abs(_simple_sentiment(headline))
    return round(0.4 * recency + 0.3 * entity + 0.3 * sentiment_strength, 3)


# ─────────────────────────────────────────────────────────────
# Data ingestion
# ─────────────────────────────────────────────────────────────

def _fetch_moneycontrol_news(feed_type: str = "latest") -> List[Dict]:
    """
    Fetch news via the moneycontrol-api PyPI library.
    Falls back to [] on any error.
    """
    if MOCK_MODE:
        return _mock_moneycontrol()
    try:
        if feed_type == "business":
            raw = mc_api.get_business_news()
        else:
            raw = mc_api.get_latest_news()
        # The library returns a list of dicts with keys: title, link, date
        if isinstance(raw, list):
            return raw
        return []
    except Exception as e:
        print(f"[Agent1] MoneyControl fetch error: {e}")
        return []


def _fetch_google_news(ticker: str) -> List[Dict]:
    """Fetch Google News RSS for a ticker/entity."""
    if MOCK_MODE:
        return _mock_google_news(ticker)
    url = RSS_TEMPLATE.format(query=f"{ticker}+stock+NSE")
    try:
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries[:10]:
            results.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "date": entry.get("published", datetime.now(timezone.utc).isoformat()),
                "ticker": ticker,
            })
        return results
    except Exception as e:
        print(f"[Agent1] Google News fetch error for {ticker}: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# Mock data (for CI / offline demos)
# ─────────────────────────────────────────────────────────────

def _mock_moneycontrol() -> List[Dict]:
    return [
        {"title": "Reliance Industries Q4 profit beats estimates by 12%", "link": "https://moneycontrol.com/mock/1", "date": datetime.now(timezone.utc).isoformat()},
        {"title": "HDFC Bank gets upgrade to BUY from Motilal Oswal", "link": "https://moneycontrol.com/mock/2", "date": datetime.now(timezone.utc).isoformat()},
        {"title": "TCS announces major acquisition in European market", "link": "https://moneycontrol.com/mock/3", "date": datetime.now(timezone.utc).isoformat()},
        {"title": "RBI keeps repo rate unchanged at 6.5%, signals easing", "link": "https://moneycontrol.com/mock/4", "date": datetime.now(timezone.utc).isoformat()},
        {"title": "Infosys revenue guidance cut amid weak demand signals", "link": "https://moneycontrol.com/mock/5", "date": datetime.now(timezone.utc).isoformat()},
    ]

def _mock_google_news(ticker: str) -> List[Dict]:
    return [
        {"title": f"{ticker} surges 5% after strong FII buying", "link": f"https://news.google.com/mock/{ticker}/1", "date": datetime.now(timezone.utc).isoformat(), "ticker": ticker},
        {"title": f"{ticker} Q3 results: earnings growth of 18% YoY", "link": f"https://news.google.com/mock/{ticker}/2", "date": datetime.now(timezone.utc).isoformat(), "ticker": ticker},
    ]


# ─────────────────────────────────────────────────────────────
# Core agent logic
# ─────────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> datetime:
    """Best-effort date parser."""
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(date_str).replace(tzinfo=timezone.utc)
    except Exception:
        pass
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


def scrape_and_build_leads(cycle_id: str) -> List[LeadSignal]:
    """
    Full scrape-deduplicate-score pipeline.
    Returns list of LeadSignal objects for this cycle.
    """
    store = get_store()
    raw_articles: List[Dict] = []

    # --- Source 1: MoneyControl ---
    mc_news = _fetch_moneycontrol_news(CFG["moneycontrol"]["feed_type"])
    for item in mc_news:
        raw_articles.append({
            "headline": item.get("title", ""),
            "url": item.get("link", ""),
            "date": item.get("date", datetime.now(timezone.utc).isoformat()),
            "ticker": _guess_ticker(item.get("title", "")),
            "source": "MoneyControl",
        })

    # --- Source 2: Google News RSS per watchlist ticker ---
    for ticker in WATCHLIST:
        gn_articles = _fetch_google_news(ticker)
        for item in gn_articles:
            raw_articles.append({
                "headline": item.get("title", ""),
                "url": item.get("link", ""),
                "date": item.get("date", datetime.now(timezone.utc).isoformat()),
                "ticker": ticker,
                "source": "GoogleNews",
            })

    # --- Deduplicate, score, filter ---
    leads: List[LeadSignal] = []
    for art in raw_articles:
        headline = art["headline"].strip()
        if not headline:
            continue

        h = _article_hash(headline, art["url"])
        if _is_duplicate(h, store):
            continue
        _mark_seen(h, store)

        ts = _parse_date(art["date"])
        sentiment = _simple_sentiment(headline)
        score = _relevance_score(headline, art["ticker"], ts)

        leads.append(LeadSignal(
            ticker=art["ticker"],
            headline=headline,
            trigger=_classify_trigger(headline),
            sentiment=sentiment,
            score=score,
            url=art["url"],
            source=art["source"],
            ts=ts,
            cycle_id=cycle_id,
        ))

    # Sort by score descending
    leads.sort(key=lambda x: x.score, reverse=True)
    return leads


def _guess_ticker(headline: str) -> str:
    """Try to match a watchlist ticker from the headline text."""
    hl = headline.upper()
    company_map = {
        "RELIANCE": "RELIANCE", "HDFC": "HDFCBANK", "TCS": "TCS",
        "INFOSYS": "INFY", "WIPRO": "WIPRO", "BAJAJ FINANCE": "BAJFINANCE",
        "TITAN": "TITAN", "AXIS BANK": "AXISBANK", "SBI": "SBIN", "ICICI": "ICICIBANK",
    }
    for kw, ticker in company_map.items():
        if kw in hl:
            return ticker
    return "NIFTY50"   # fallback


# ─────────────────────────────────────────────────────────────
# LangGraph node entrypoint
# ─────────────────────────────────────────────────────────────

def run_lead_sourcer(graph_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function.
    Reads: graph_state["cycle_id"]
    Writes: graph_state["lead_feed"], graph_state["agent1_status"]
    Emits: LEAD_FEED_READY event
    """
    cycle_id = graph_state.get("cycle_id", str(uuid.uuid4()))
    print(f"\n[Agent1] ▶  Lead Sourcer started  cycle={cycle_id}")

    store = get_store()
    leads = scrape_and_build_leads(cycle_id)

    # Persist to shared state
    lead_dicts = [l.model_dump(mode="json") for l in leads]
    store.set(f"leads:{cycle_id}", lead_dicts)

    # Emit event
    store.emit_event("LEAD_FEED_READY", {"cycle_id": cycle_id, "count": len(leads)})

    print(f"[Agent1] ✔  {len(leads)} leads extracted and stored.")

    return {
        **graph_state,
        "lead_feed": lead_dicts,
        "agent1_status": "complete",
        "cycle_id": cycle_id,
    }
