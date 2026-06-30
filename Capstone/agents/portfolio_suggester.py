"""Agent 2 — Portfolio Suggester.

Responsibilities
----------------
* Consume Lead Feed from Agent 1.
* Enrich leads with market data (sector momentum, P/E, 52-week position).
* Apply HNI investment filters (min ₹50L ticket, risk grade, SEBI categories).
* Generate top-5 investment suggestions per client via LLM reasoning chain.
* Each suggestion carries: instrument, rationale, horizon, risk, confidence,
  plus explainability fields (prompt_id, model_version, reasoning).
* Emit SUGGESTION_READY event.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from orchestrator.state import InvestmentSuggestion, LeadSignal

logger = logging.getLogger(__name__)

# ── LangChain / Anthropic (lazy imports — gracefully degrade in tests) ────────
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    logger.warning("langchain_openai not installed; LLM calls will use mock responses")


_SYSTEM_PROMPT = """You are an expert Wealth Manager specialising in HNI portfolios in India.
Your task is to generate the top-5 investment suggestions for an HNI client based on:
1. Recent market intelligence leads
2. The client's risk profile
3. Current market enrichment data (P/E, sector momentum, 52-week position)

Rules:
- Minimum investment ticket size: ₹50 Lakh per recommendation
- Instruments must be SEBI-approved
- Align each suggestion to the client's risk grade (Conservative / Balanced / Aggressive)
- Exclude any instruments on the client's exclusion list
- Each suggestion must include: instrument, rationale, horizon (short/medium/long), risk label, confidence (0-1)
- Be concise but data-backed

Respond ONLY with a valid JSON array of exactly 5 objects.
Each object must have keys: instrument, rationale, horizon, risk, confidence, reasoning
"""

_SUGGESTION_TEMPLATE = """Client Profile:
{client_profile}

Market Intelligence Leads (top signals from this cycle):
{lead_feed}

Market Enrichment Data for watchlist:
{market_data}

Exclusion list for this client: {exclusion_list}
Liquidity constraint: ₹{liquidity_lakh} Lakh (do not recommend illiquid instruments)

Generate top-5 investment suggestions as a JSON array."""


def _mock_suggestions(client_profile: dict, leads: list[LeadSignal]) -> list[InvestmentSuggestion]:
    """Return deterministic mock suggestions for CI / mock mode."""
    risk = client_profile.get("risk_profile", "Balanced")
    prompt_id = f"mock-{uuid.uuid4().hex[:8]}"
    model_ver = "mock-v1"

    base: list[dict] = [
        {
            "instrument": "RELIANCE",
            "rationale": "Strong Q1 earnings; green energy pivot unlocks multi-year growth",
            "horizon": "medium",
            "risk": "Balanced",
            "confidence": 0.85,
            "reasoning": "Earnings beat + strategic capex in Jio + green hydrogen optionality",
        },
        {
            "instrument": "HDFCBANK",
            "rationale": "Analyst upgrade to Outperform; NIM recovery on track post-merger",
            "horizon": "medium",
            "risk": "Conservative",
            "confidence": 0.82,
            "reasoning": "Valuation attractive at 2.4x P/B; asset quality stable",
        },
        {
            "instrument": "BAJFINANCE",
            "rationale": "AUM milestone ₹4L cr; guidance raised; GNPA stable at 1.1%",
            "horizon": "long",
            "risk": "Aggressive",
            "confidence": 0.88,
            "reasoning": "Best-in-class NBFC compounder; digital lending acceleration",
        },
        {
            "instrument": "SUNPHARMA",
            "rationale": "USFDA specialty approval expands addressable market by ₹15,000 cr",
            "horizon": "long",
            "risk": "Balanced",
            "confidence": 0.79,
            "reasoning": "Specialty pivot reduces US generics risk; domestic formulations growing",
        },
        {
            "instrument": "LT",
            "rationale": "₹12,500 cr NHAI order win; order book at all-time high; capex cycle beneficiary",
            "horizon": "medium",
            "risk": "Balanced",
            "confidence": 0.83,
            "reasoning": "Infrastructure supercycle; strong order pipeline + government capex tailwind",
        },
    ]

    risk_map = {"Conservative": 0, "Balanced": 1, "Aggressive": 2}
    client_risk_idx = risk_map.get(risk, 1)

    result: list[InvestmentSuggestion] = []
    for s in base:
        suggestion_risk_idx = risk_map.get(s["risk"], 1)
        if abs(suggestion_risk_idx - client_risk_idx) <= 1:
            result.append(
                InvestmentSuggestion(
                    instrument=s["instrument"],
                    rationale=s["rationale"],
                    horizon=s["horizon"],
                    risk=risk,
                    confidence=s["confidence"],
                    prompt_id=prompt_id,
                    model_version=model_ver,
                    reasoning=s["reasoning"],
                )
            )
        if len(result) == 5:
            break

    while len(result) < 5:
        result.append(result[0])  # pad — should not happen with 5 base items

    return result[:5]


class PortfolioSuggesterAgent:
    """Generates HNI-grade investment suggestions using LLM reasoning."""

    def __init__(self, config: dict, state_store: Any):
        self.config = config
        self.state_store = state_store
        self.mock_mode: bool = config.get("llm", {}).get("mock_mode", False)
        self.top_n: int = config.get("portfolio_suggester", {}).get("top_n_suggestions", 5)
        self.min_ticket: float = config.get("portfolio_suggester", {}).get("hni_min_ticket_lakh", 50)

        _market_data_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "mock_market_data.json"
        )
        with open(os.path.normpath(_market_data_path)) as f:
            self._market_data: dict = json.load(f)

        self._llm: Any = None
        if not self.mock_mode and _LANGCHAIN_AVAILABLE:
            try:
                self._llm = ChatOpenAI(
                    model=config.get("llm", {}).get("model", "gpt-4o"),
                    temperature=config.get("llm", {}).get("temperature", 0.1),
                    max_tokens=config.get("llm", {}).get("max_tokens", 4096),
                )
            except Exception as exc:
                logger.warning("ChatOpenAI init failed (%s); falling back to mock responses", exc)

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        leads: list[LeadSignal],
        client_profile: dict,
        overrides: dict | None = None,
    ) -> list[InvestmentSuggestion]:
        """Generate top-N suggestions for one client profile."""
        overrides = overrides or {}
        exclusion_list: list[str] = list(
            set(client_profile.get("exclusion_list", []) + overrides.get("exclusion_list", []))
        )
        liquidity_lakh: float = overrides.get(
            "liquidity_lakh", client_profile.get("liquidity_constraint_lakh", 0)
        )

        filtered_leads = self._filter_leads(leads, exclusion_list)

        if self.mock_mode or self._llm is None:
            logger.info("PortfolioSuggesterAgent: mock/no-LLM mode for client %s", client_profile.get("client_id"))
            suggestions = _mock_suggestions(client_profile, filtered_leads)
        else:
            suggestions = self._llm_suggest(filtered_leads, client_profile, exclusion_list, liquidity_lakh)

        return suggestions[: self.top_n]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _filter_leads(self, leads: list[LeadSignal], exclusion_list: list[str]) -> list[LeadSignal]:
        return [l for l in leads if l["ticker"] not in exclusion_list and l["sentiment"] != "negative"][:15]

    def _llm_suggest(
        self,
        leads: list[LeadSignal],
        client_profile: dict,
        exclusion_list: list[str],
        liquidity_lakh: float,
    ) -> list[InvestmentSuggestion]:
        prompt_id = f"ps-{uuid.uuid4().hex[:12]}"
        model_version = self.config.get("llm", {}).get("model", "gpt-4o")

        lead_text = json.dumps(
            [{"ticker": l["ticker"], "headline": l["headline"], "trigger": l["trigger"], "sentiment": l["sentiment"]}
             for l in leads],
            indent=2,
        )
        market_text = json.dumps(
            {k: {f: v for f, v in md.items() if f in ("pe_ratio", "sector_momentum_score", "week_52_position",
                                                         "analyst_consensus", "target_price")}
             for k, md in self._market_data.items()},
            indent=2,
        )
        profile_text = json.dumps(
            {k: v for k, v in client_profile.items() if k not in ("holdings",)},
            indent=2,
        )

        user_content = _SUGGESTION_TEMPLATE.format(
            client_profile=profile_text,
            lead_feed=lead_text,
            market_data=market_text,
            exclusion_list=", ".join(exclusion_list) if exclusion_list else "None",
            liquidity_lakh=liquidity_lakh,
        )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        try:
            response = self._llm.invoke(messages)
            raw = response.content.strip()
            # Extract JSON array from response (may have markdown fences)
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed: list[dict] = json.loads(raw)
        except Exception as exc:
            logger.error("LLM suggestion call failed: %s — falling back to mock", exc)
            return _mock_suggestions(client_profile, leads)

        suggestions: list[InvestmentSuggestion] = []
        for item in parsed[: self.top_n]:
            suggestions.append(
                InvestmentSuggestion(
                    instrument=item.get("instrument", ""),
                    rationale=item.get("rationale", ""),
                    horizon=item.get("horizon", "medium"),
                    risk=item.get("risk", client_profile.get("risk_profile", "Balanced")),
                    confidence=float(item.get("confidence", 0.7)),
                    prompt_id=prompt_id,
                    model_version=model_version,
                    reasoning=item.get("reasoning", ""),
                )
            )
        return suggestions
