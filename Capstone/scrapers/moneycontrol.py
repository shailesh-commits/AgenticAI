"""MoneyControl scraper — CSS-selector-driven, httpx + BeautifulSoup.

In mock_mode the scraper returns pre-generated fixtures so CI runs never
hit the live site.  CSS/XPath selectors are externalised to config/selectors.yaml
so structure changes require only a config update, not code changes.
"""

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

import httpx

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False


_MOCK_ARTICLES = [
    {
        "ticker": "RELIANCE",
        "headline": "Reliance Industries eyes $10B green hydrogen investment over next 5 years",
        "url": "https://www.moneycontrol.com/mock/reliance-green-hydrogen",
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "moneycontrol",
    },
    {
        "ticker": "HDFCBANK",
        "headline": "HDFC Bank Q1 PAT ₹16,175 cr, up 35% YoY; NIM expands to 3.6%",
        "url": "https://www.moneycontrol.com/mock/hdfcbank-q1-results",
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "moneycontrol",
    },
    {
        "ticker": "BAJFINANCE",
        "headline": "Bajaj Finance AUM hits ₹4 lakh crore; FY27 guidance raised to 25% growth",
        "url": "https://www.moneycontrol.com/mock/bajfinance-aum",
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "moneycontrol",
    },
    {
        "ticker": "SUNPHARMA",
        "headline": "Sun Pharma gets USFDA nod for specialty derm drug; ₹1,800 cr revenue potential",
        "url": "https://www.moneycontrol.com/mock/sunpharma-fda",
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "moneycontrol",
    },
    {
        "ticker": "LT",
        "headline": "L&T wins ₹12,500 cr NHAI highway order; order book touches all-time high",
        "url": "https://www.moneycontrol.com/mock/lt-nhai-order",
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "moneycontrol",
    },
]


class MoneyControlScraper:
    """Scrapes MoneyControl market news headlines and market-mover data."""

    def __init__(self, config: dict, selectors: dict):
        self.config = config
        self.selectors = selectors.get("moneycontrol", {})
        self.mock_mode: bool = config.get("scraping", {}).get("mock_mode", True)
        self.base_url: str = config.get("scraping", {}).get("moneycontrol", {}).get(
            "base_url", "https://www.moneycontrol.com"
        )
        self.news_path: str = config.get("scraping", {}).get("moneycontrol", {}).get(
            "news_path", "/news/business/markets"
        )
        self.timeout: int = config.get("scraping", {}).get("timeout_seconds", 30) * 1000

    def fetch(self) -> list[dict[str, Any]]:
        if self.mock_mode:
            logger.info("MoneyControlScraper: mock_mode ON — returning fixtures")
            return _MOCK_ARTICLES

        if not _BS4_AVAILABLE:
            logger.warning("BeautifulSoup not available; falling back to mock data")
            return _MOCK_ARTICLES

        return self._live_fetch()

    def _live_fetch(self) -> list[dict[str, Any]]:
        url = f"{self.base_url}{self.news_path}"
        articles: list[dict[str, Any]] = []

        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; WAM-Bot/1.0)"}
            timeout = self.timeout / 1000  # httpx uses seconds
            response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")
            container_sel = self.selectors.get("news_list", {}).get("container", "li.clearfix")
            headline_sel = self.selectors.get("news_list", {}).get("headline", "h2 a")
            url_sel = self.selectors.get("news_list", {}).get("url", "h2 a")
            ts_sel = self.selectors.get("news_list", {}).get("timestamp", "span.ago")

            for item in soup.select(container_sel)[:30]:
                headline_tag = item.select_one(headline_sel)
                if not headline_tag:
                    continue
                headline = headline_tag.get_text(strip=True)
                article_url = headline_tag.get("href", "")
                if article_url and not article_url.startswith("http"):
                    article_url = self.base_url + article_url
                ts_tag = item.select_one(ts_sel)
                ts_text = ts_tag.get_text(strip=True) if ts_tag else ""
                articles.append(
                    {
                        "ticker": "",          # enriched downstream by lead_sourcer
                        "headline": headline,
                        "url": article_url,
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "ts_text": ts_text,
                        "source": "moneycontrol",
                    }
                )
        except Exception as exc:
            logger.error("MoneyControl live scrape failed: %s", exc)
            return _MOCK_ARTICLES

        logger.info("MoneyControlScraper: fetched %d articles", len(articles))
        return articles
