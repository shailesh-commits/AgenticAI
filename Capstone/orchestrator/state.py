"""Shared state type definitions for the WAM Agentic Platform."""

import operator
from typing import TypedDict, List, Dict, Annotated


class LeadSignal(TypedDict):
    ticker: str
    headline: str
    trigger: str        # earnings | M&A | rating_change | macro | sector_move
    sentiment: str      # positive | negative | neutral
    score: float        # 0.0 – 1.0 composite relevance score
    url: str
    ts: str             # ISO-8601 timestamp
    source: str         # moneycontrol | google_news


class InvestmentSuggestion(TypedDict):
    instrument: str
    rationale: str
    horizon: str        # short (<1yr) | medium (1–3yr) | long (>3yr)
    risk: str           # Conservative | Balanced | Aggressive
    confidence: float   # 0.0 – 1.0
    prompt_id: str
    model_version: str
    reasoning: str


class RebalanceAction(TypedDict):
    action: str         # add | reduce | hold
    instrument: str
    target_weight: float
    current_weight: float
    justification: str


class OrchestratorState(TypedDict):
    cycle_id: str
    event: str          # IDLE | LEAD_FEED_READY | SUGGESTION_READY | DIVERSIFICATION_REPORT | ERROR
    lead_feed: List[LeadSignal]
    suggestions: Dict[str, List[InvestmentSuggestion]]          # keyed by client_id
    diversification_reports: Dict[str, List[RebalanceAction]]   # keyed by client_id
    client_profiles: Dict[str, dict]
    portfolio_holdings: Dict[str, List[dict]]
    errors: Annotated[List[str], operator.add]
    completed_steps: Annotated[List[str], operator.add]
    metadata: Dict[str, str]
