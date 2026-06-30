"""Google News RSS fetcher — feedparser-based, mock-first."""

import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

logger = logging.getLogger(__name__)

try:
    import feedparser
    _FEEDPARSER_AVAILABLE = True
except ImportError:
    _FEEDPARSER_AVAILABLE = False


def _mock_articles_for(ticker: str) -> list[dict[str, Any]]:
    return [
        {
            "ticker": ticker,
            "headline": f"{ticker}: Strong institutional buying detected; FII inflows up 18% MoM",
            "url": f"https://news.google.com/mock/{ticker.lower()}-fii-buying",
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": "google_news",
        },
        {
            "ticker": ticker,
            "headline": f"{ticker} included in MSCI Emerging Markets index rebalance — inflows expected",
            "url": f"https://news.google.com/mock/{ticker.lower()}-msci",
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": "google_news",
        },
    ]


class GoogleNewsFetcher:
    """Fetches ticker-specific news from Google News RSS."""

    def __init__(self, config: dict):
        self.config = config
        self.mock_mode: bool = config.get("scraping", {}).get("mock_mode", True)
        self.rss_template: str = config.get("scraping", {}).get("google_news", {}).get(
            "rss_template",
            "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en",
        )
        self.max_per_ticker: int = config.get("scraping", {}).get("google_news", {}).get(
            "max_articles_per_ticker", 5
        )

    def fetch(self, tickers: list[str]) -> list[dict[str, Any]]:
        if self.mock_mode:
            logger.info("GoogleNewsFetcher: mock_mode ON — returning fixtures for %d tickers", len(tickers))
            articles: list[dict[str, Any]] = []
            for ticker in tickers:
                articles.extend(_mock_articles_for(ticker))
            return articles

        if not _FEEDPARSER_AVAILABLE:
            logger.warning("feedparser not available; falling back to mock data")
            articles = []
            for ticker in tickers:
                articles.extend(_mock_articles_for(ticker))
            return articles

        return self._live_fetch(tickers)

    def _live_fetch(self, tickers: list[str]) -> list[dict[str, Any]]:
        articles: list[dict[str, Any]] = []
        for ticker in tickers:
            query = f"{ticker} NSE stock"
            url = self.rss_template.format(query=query.replace(" ", "+"))
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[: self.max_per_ticker]:
                    published = datetime.now(timezone.utc)
                    if hasattr(entry, "published"):
                        try:
                            published = parsedate_to_datetime(entry.published)
                        except Exception:
                            pass
                    articles.append(
                        {
                            "ticker": ticker,
                            "headline": entry.get("title", ""),
                            "url": entry.get("link", ""),
                            "ts": published.isoformat(),
                            "source": "google_news",
                        }
                    )
            except Exception as exc:
                logger.warning("Google News fetch failed for %s: %s", ticker, exc)
                articles.extend(_mock_articles_for(ticker))

        logger.info("GoogleNewsFetcher: fetched %d articles for %d tickers", len(articles), len(tickers))
        return articles
