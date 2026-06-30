"""FastAPI REST layer — RM Workbench Integration.

Endpoints
---------
GET  /health                           → service health check
GET  /leads                            → latest lead feed (last cycle or ?cycle_id=)
GET  /suggestions/{client_id}          → investment suggestions for a client
GET  /diversification/{client_id}      → diversification report for a client
POST /run-cycle                        → trigger an on-demand intelligence cycle
GET  /clients                          → list all known client IDs
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from orchestrator.state_store import StateStore

logger = logging.getLogger(__name__)

# ── App-level state container (populated at startup) ──────────────────────────
_app_state: dict[str, Any] = {}


def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(os.path.normpath(config_path)) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = _load_config()
    _app_state["config"] = config
    _app_state["state_store"] = StateStore(config)
    logger.info("WAM Platform API started")
    yield
    logger.info("WAM Platform API shutdown")


# ── Response models ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class LeadSignalResponse(BaseModel):
    ticker: str
    headline: str
    trigger: str
    sentiment: str
    score: float
    url: str
    ts: str
    source: str


class SuggestionResponse(BaseModel):
    instrument: str
    rationale: str
    horizon: str
    risk: str
    confidence: float
    reasoning: str
    prompt_id: str
    model_version: str


class RebalanceActionResponse(BaseModel):
    action: str
    instrument: str
    target_weight: float
    current_weight: float
    justification: str


class CycleResponse(BaseModel):
    cycle_id: str
    completed_steps: list[str]
    errors: list[str]
    lead_count: int
    client_ids: list[str]


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="WAM Agentic Platform API",
        description="RM Intelligence Platform — Lead Feed, Portfolio Suggestions & Diversification Reports",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ────────────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse, tags=["Platform"])
    async def health():
        cfg = _app_state.get("config", {})
        return HealthResponse(
            status="ok",
            version=cfg.get("app", {}).get("version", "1.0.0"),
            environment=cfg.get("app", {}).get("env", "development"),
        )

    # ── Leads ─────────────────────────────────────────────────────────────────

    @app.get("/leads", response_model=list[LeadSignalResponse], tags=["Agent 1 — Lead Sourcer"])
    async def get_leads(
        cycle_id: str | None = Query(None, description="Filter by cycle ID"),
        limit: int = Query(50, ge=1, le=200),
    ):
        store: StateStore = _app_state["state_store"]
        leads = store.get_leads(cycle_id=cycle_id, limit=limit)
        if not leads:
            raise HTTPException(status_code=404, detail="No leads found. Run a cycle first.")
        return leads

    # ── Suggestions ───────────────────────────────────────────────────────────

    @app.get(
        "/suggestions/{client_id}",
        response_model=list[SuggestionResponse],
        tags=["Agent 2 — Portfolio Suggester"],
    )
    async def get_suggestions(
        client_id: str,
        cycle_id: str | None = Query(None, description="Filter by cycle ID"),
    ):
        store: StateStore = _app_state["state_store"]
        suggestions = store.get_suggestions(client_id=client_id, cycle_id=cycle_id)
        if not suggestions:
            raise HTTPException(
                status_code=404,
                detail=f"No suggestions found for client '{client_id}'. Run a cycle first.",
            )
        return suggestions

    # ── Diversification ───────────────────────────────────────────────────────

    @app.get(
        "/diversification/{client_id}",
        response_model=list[RebalanceActionResponse],
        tags=["Agent 3 — Portfolio Diversifier"],
    )
    async def get_diversification(
        client_id: str,
        cycle_id: str | None = Query(None, description="Filter by cycle ID"),
    ):
        store: StateStore = _app_state["state_store"]
        report = store.get_diversification_report(client_id=client_id, cycle_id=cycle_id)
        if not report:
            raise HTTPException(
                status_code=404,
                detail=f"No diversification report for client '{client_id}'. Run a cycle first.",
            )
        return report

    # ── On-demand cycle trigger ───────────────────────────────────────────────

    @app.post("/run-cycle", response_model=CycleResponse, tags=["Orchestrator"])
    async def run_cycle_endpoint():
        import asyncio

        config = _app_state["config"]
        state_store = _app_state["state_store"]

        # Run synchronously in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        from orchestrator.graph import run_cycle

        final_state = await loop.run_in_executor(None, run_cycle, config, state_store)

        return CycleResponse(
            cycle_id=final_state.get("cycle_id", ""),
            completed_steps=final_state.get("completed_steps", []),
            errors=final_state.get("errors", []),
            lead_count=len(final_state.get("lead_feed", [])),
            client_ids=list(final_state.get("suggestions", {}).keys()),
        )

    # ── Client list ───────────────────────────────────────────────────────────

    @app.get("/clients", response_model=list[str], tags=["Platform"])
    async def list_clients():
        import json

        portfolios_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "data", "mock_portfolios.json")
        )
        with open(portfolios_path) as f:
            portfolios = json.load(f)
        return [p["client_id"] for p in portfolios]

    return app


# Module-level app instance for uvicorn
app = create_app()
