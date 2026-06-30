"""LangGraph-based Central Orchestrator.

Graph topology
--------------
START
  └─► lead_sourcer_node
        ├─(LEAD_FEED_READY)─► portfolio_suggester_node
        │                          ├─(SUGGESTION_READY)─► portfolio_diversifier_node
        │                          │                           └─► conflict_resolver_node ─► END
        │                          └─(ERROR)─► END
        └─(ERROR)─► END

Inter-agent events flow through shared OrchestratorState; nodes never call
each other directly (loose coupling per design principle).
"""

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.lead_sourcer import LeadSourcerAgent
from agents.portfolio_diversifier import PortfolioDiversifierAgent
from agents.portfolio_suggester import PortfolioSuggesterAgent
from orchestrator.state import OrchestratorState
from orchestrator.state_store import StateStore

logger = logging.getLogger(__name__)

# ── Checkpointer (SqliteSaver) — lazy import ──────────────────────────────────
try:
    from langgraph.checkpoint.sqlite import SqliteSaver as _SqliteSaver
    _CHECKPOINTER_AVAILABLE = True
except ImportError:
    _CHECKPOINTER_AVAILABLE = False
    logger.warning("langgraph.checkpoint.sqlite not available; running without checkpointing")


def _load_client_data(config: dict) -> tuple[dict, dict]:
    """Load client profiles and portfolio holdings from mock data file."""
    portfolios_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "mock_portfolios.json")
    )
    with open(portfolios_path) as f:
        portfolios: list[dict] = json.load(f)

    client_profiles: dict[str, dict] = {}
    portfolio_holdings: dict[str, list] = {}
    for p in portfolios:
        cid = p["client_id"]
        client_profiles[cid] = {k: v for k, v in p.items() if k != "holdings"}
        portfolio_holdings[cid] = p.get("holdings", [])

    return client_profiles, portfolio_holdings


def build_graph(config: dict, state_store: StateStore, selectors: dict | None = None):
    """Compile and return the LangGraph orchestrator graph."""
    selectors = selectors or {}

    lead_sourcer = LeadSourcerAgent(config, state_store, selectors)
    portfolio_suggester = PortfolioSuggesterAgent(config, state_store)
    portfolio_diversifier = PortfolioDiversifierAgent(config, state_store)

    # ── Node definitions ──────────────────────────────────────────────────────

    def lead_sourcer_node(state: OrchestratorState) -> dict:
        logger.info("[Orchestrator] lead_sourcer_node — cycle %s", state.get("cycle_id"))
        try:
            leads = lead_sourcer.run()
            state_store.save_leads(state["cycle_id"], leads)
            return {
                "lead_feed": leads,
                "event": "LEAD_FEED_READY",
                "completed_steps": ["lead_sourcer"],
                "metadata": {**state.get("metadata", {}), "lead_sourcer_ts": datetime.now(timezone.utc).isoformat()},
            }
        except Exception as exc:
            logger.exception("lead_sourcer_node failed")
            return {"errors": [f"lead_sourcer: {exc}"], "event": "ERROR"}

    def portfolio_suggester_node(state: OrchestratorState) -> dict:
        logger.info("[Orchestrator] portfolio_suggester_node — cycle %s", state.get("cycle_id"))
        try:
            all_suggestions: dict = {}
            for client_id, profile in state["client_profiles"].items():
                overrides = {}
                suggestions = portfolio_suggester.run(state["lead_feed"], profile, overrides)
                all_suggestions[client_id] = suggestions
                state_store.save_suggestions(client_id, state["cycle_id"], suggestions)
                logger.debug("Generated %d suggestions for %s", len(suggestions), client_id)
            return {
                "suggestions": all_suggestions,
                "event": "SUGGESTION_READY",
                "completed_steps": ["portfolio_suggester"],
                "metadata": {**state.get("metadata", {}), "suggester_ts": datetime.now(timezone.utc).isoformat()},
            }
        except Exception as exc:
            logger.exception("portfolio_suggester_node failed")
            return {"errors": [f"portfolio_suggester: {exc}"], "event": "ERROR"}

    def portfolio_diversifier_node(state: OrchestratorState) -> dict:
        logger.info("[Orchestrator] portfolio_diversifier_node — cycle %s", state.get("cycle_id"))
        try:
            all_reports: dict = {}
            for client_id, holdings in state["portfolio_holdings"].items():
                suggestions = state["suggestions"].get(client_id, [])
                profile = state["client_profiles"].get(client_id, {})
                report = portfolio_diversifier.run(holdings, suggestions, profile)
                all_reports[client_id] = report
                state_store.save_diversification_report(client_id, state["cycle_id"], report)
                logger.debug("Generated %d actions for %s", len(report), client_id)
            return {
                "diversification_reports": all_reports,
                "event": "DIVERSIFICATION_REPORT",
                "completed_steps": ["portfolio_diversifier"],
                "metadata": {**state.get("metadata", {}), "diversifier_ts": datetime.now(timezone.utc).isoformat()},
            }
        except Exception as exc:
            logger.exception("portfolio_diversifier_node failed")
            return {"errors": [f"portfolio_diversifier: {exc}"], "event": "ERROR"}

    def conflict_resolver_node(state: OrchestratorState) -> dict:
        """Remove suggestions that directly conflict with diversification reduce-actions."""
        logger.info("[Orchestrator] conflict_resolver_node — cycle %s", state.get("cycle_id"))
        resolved: dict = {}
        for client_id, suggestions in state["suggestions"].items():
            reports = state["diversification_reports"].get(client_id, [])
            reduce_instruments = {r["instrument"] for r in reports if r["action"] == "reduce"}
            clean = [s for s in suggestions if s["instrument"] not in reduce_instruments]
            resolved[client_id] = clean
            removed = len(suggestions) - len(clean)
            if removed:
                logger.info("Conflict resolver: removed %d conflicting suggestion(s) for %s", removed, client_id)

        return {
            "suggestions": resolved,
            "completed_steps": ["conflict_resolver"],
            "metadata": {
                **state.get("metadata", {}),
                "conflict_resolver_ts": datetime.now(timezone.utc).isoformat(),
                "cycle_complete": "true",
            },
        }

    # ── Routing functions ─────────────────────────────────────────────────────

    def route_after_lead_sourcer(state: OrchestratorState) -> str:
        if state.get("event") == "LEAD_FEED_READY" and state.get("lead_feed"):
            return "continue"
        return "error"

    def route_after_suggester(state: OrchestratorState) -> str:
        if state.get("event") == "SUGGESTION_READY" and state.get("suggestions"):
            return "continue"
        return "error"

    # ── Graph assembly ────────────────────────────────────────────────────────

    builder = StateGraph(OrchestratorState)

    builder.add_node("lead_sourcer", lead_sourcer_node)
    builder.add_node("portfolio_suggester", portfolio_suggester_node)
    builder.add_node("portfolio_diversifier", portfolio_diversifier_node)
    builder.add_node("conflict_resolver", conflict_resolver_node)

    builder.add_edge(START, "lead_sourcer")
    builder.add_conditional_edges(
        "lead_sourcer",
        route_after_lead_sourcer,
        {"continue": "portfolio_suggester", "error": END},
    )
    builder.add_conditional_edges(
        "portfolio_suggester",
        route_after_suggester,
        {"continue": "portfolio_diversifier", "error": END},
    )
    builder.add_edge("portfolio_diversifier", "conflict_resolver")
    builder.add_edge("conflict_resolver", END)

    # Attach SQLite checkpointer when available
    checkpointer = None
    if _CHECKPOINTER_AVAILABLE:
        db_path = config.get("state_store", {}).get("sqlite_path", "./data/state.db")
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = _SqliteSaver(conn)

    graph = builder.compile(checkpointer=checkpointer)
    logger.info("Orchestrator graph compiled (checkpointer=%s)", "sqlite" if checkpointer else "none")
    return graph


def run_cycle(config: dict, state_store: StateStore, selectors: dict | None = None) -> OrchestratorState:
    """Execute one full intelligence cycle and return the final state."""
    cycle_id = f"cycle-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    logger.info("Starting orchestrator cycle: %s", cycle_id)

    client_profiles, portfolio_holdings = _load_client_data(config)

    initial_state: OrchestratorState = {
        "cycle_id": cycle_id,
        "event": "IDLE",
        "lead_feed": [],
        "suggestions": {},
        "diversification_reports": {},
        "client_profiles": client_profiles,
        "portfolio_holdings": portfolio_holdings,
        "errors": [],
        "completed_steps": [],
        "metadata": {"started_at": datetime.now(timezone.utc).isoformat()},
    }

    graph = build_graph(config, state_store, selectors)
    thread_config = {"configurable": {"thread_id": cycle_id}}

    final_state: OrchestratorState = graph.invoke(initial_state, config=thread_config)
    logger.info(
        "Cycle %s complete — steps: %s  errors: %d",
        cycle_id,
        final_state.get("completed_steps", []),
        len(final_state.get("errors", [])),
    )
    return final_state
