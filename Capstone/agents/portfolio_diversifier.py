"""Agent 3 — Portfolio Diversifier.

Responsibilities
----------------
* Ingest current portfolio holdings per HNI client.
* Compute concentration metrics: sector weight, asset-class weight,
  single-stock exposure, geographic exposure.
* Identify gaps against policy thresholds from config.
* Propose rebalancing actions (add / reduce / hold) with target weight deltas
  and expected Sharpe improvement estimate.
* Cross-reference proposals with Agent 2 suggestions to avoid conflicts.
* Emit DIVERSIFICATION_REPORT event.
"""

import logging
from collections import defaultdict
from typing import Any

from orchestrator.state import InvestmentSuggestion, RebalanceAction

logger = logging.getLogger(__name__)


class PortfolioDiversifierAgent:
    """Analyses HNI portfolios and proposes rebalancing to address concentration gaps."""

    def __init__(self, config: dict, state_store: Any):
        self.config = config
        self.state_store = state_store

        div_cfg = config.get("portfolio_diversifier", {})
        self.global_thresholds: dict = div_cfg.get("thresholds", {
            "max_sector_weight": 0.30,
            "max_single_stock_weight": 0.15,
            "max_geographic_concentration": 0.80,
        })
        self.profile_limits: dict = div_cfg.get("profile_limits", {
            "Conservative": {"max_equity_weight": 0.50, "min_debt_weight": 0.40, "max_alternatives_weight": 0.10},
            "Balanced":     {"max_equity_weight": 0.70, "min_debt_weight": 0.20, "max_alternatives_weight": 0.15},
            "Aggressive":   {"max_equity_weight": 0.90, "min_debt_weight": 0.05, "max_alternatives_weight": 0.25},
        })

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        holdings: list[dict],
        suggestions: list[InvestmentSuggestion],
        client_profile: dict,
    ) -> list[RebalanceAction]:
        """Analyse holdings and return a list of RebalanceAction dicts."""
        if not holdings:
            logger.warning("PortfolioDiversifierAgent: no holdings provided")
            return []

        risk_profile: str = client_profile.get("risk_profile", "Balanced")
        limits = self.profile_limits.get(risk_profile, self.profile_limits["Balanced"])

        concentration = self._compute_concentration(holdings)
        actions: list[RebalanceAction] = []

        actions.extend(self._check_sector_concentration(concentration, holdings))
        actions.extend(self._check_single_stock_exposure(holdings))
        actions.extend(self._check_asset_class_limits(concentration, limits, risk_profile))
        actions.extend(self._check_geographic_concentration(concentration))
        actions.extend(self._suggest_additions(concentration, limits, suggestions, holdings))
        actions = self._cross_reference_suggestions(actions, suggestions)

        if not actions:
            actions.append(RebalanceAction(
                action="hold",
                instrument="PORTFOLIO",
                target_weight=1.0,
                current_weight=1.0,
                justification="Portfolio is within all policy thresholds. No rebalancing required.",
            ))

        logger.info(
            "PortfolioDiversifierAgent: generated %d rebalance actions for %s",
            len(actions), client_profile.get("client_id", "unknown"),
        )
        return actions

    # ── Concentration analysis ────────────────────────────────────────────────

    def _compute_concentration(self, holdings: list[dict]) -> dict:
        sector_weights: dict[str, float] = defaultdict(float)
        asset_class_weights: dict[str, float] = defaultdict(float)
        geo_weights: dict[str, float] = defaultdict(float)
        total_weight = sum(h.get("weight", 0.0) for h in holdings)

        for h in holdings:
            w = h.get("weight", 0.0)
            sector_weights[h.get("sector", "Unknown")] += w
            asset_class_weights[h.get("asset_class", "Unknown")] += w
            geo_weights[h.get("geography", "India")] += w

        return {
            "sector_weights": dict(sector_weights),
            "asset_class_weights": dict(asset_class_weights),
            "geo_weights": dict(geo_weights),
            "total_weight": total_weight,
        }

    def _check_sector_concentration(
        self, concentration: dict, holdings: list[dict]
    ) -> list[RebalanceAction]:
        actions: list[RebalanceAction] = []
        max_sector = self.global_thresholds.get("max_sector_weight", 0.30)

        for sector, weight in concentration["sector_weights"].items():
            if weight > max_sector:
                excess = weight - max_sector
                largest_in_sector = max(
                    [h for h in holdings if h.get("sector") == sector],
                    key=lambda h: h.get("weight", 0),
                    default=None,
                )
                if largest_in_sector:
                    target = largest_in_sector["weight"] - excess
                    actions.append(RebalanceAction(
                        action="reduce",
                        instrument=largest_in_sector["ticker"],
                        target_weight=round(max(0.0, target), 4),
                        current_weight=round(largest_in_sector["weight"], 4),
                        justification=(
                            f"Sector '{sector}' weight {weight:.1%} exceeds policy cap {max_sector:.0%}. "
                            f"Trim {largest_in_sector['ticker']} by {excess:.1%} to restore balance."
                        ),
                    ))
        return actions

    def _check_single_stock_exposure(self, holdings: list[dict]) -> list[RebalanceAction]:
        actions: list[RebalanceAction] = []
        max_stock = self.global_thresholds.get("max_single_stock_weight", 0.15)

        for h in holdings:
            weight = h.get("weight", 0.0)
            if weight > max_stock:
                actions.append(RebalanceAction(
                    action="reduce",
                    instrument=h["ticker"],
                    target_weight=round(max_stock, 4),
                    current_weight=round(weight, 4),
                    justification=(
                        f"Single-stock concentration {weight:.1%} in {h['ticker']} exceeds "
                        f"policy cap {max_stock:.0%}. Reduce by {weight - max_stock:.1%}."
                    ),
                ))
        return actions

    def _check_asset_class_limits(
        self, concentration: dict, limits: dict, risk_profile: str
    ) -> list[RebalanceAction]:
        actions: list[RebalanceAction] = []
        asset_weights = concentration["asset_class_weights"]

        equity_weight = sum(
            w for ac, w in asset_weights.items()
            if "equity" in ac.lower() or ac in ("Large Cap Equity", "Mid Cap Equity", "International Equity")
        )
        debt_weight = sum(
            w for ac, w in asset_weights.items()
            if "debt" in ac.lower() or "bond" in ac.lower() or "g-sec" in ac.lower() or "gsec" in ac.lower()
        )

        max_eq = limits.get("max_equity_weight", 0.70)
        min_debt = limits.get("min_debt_weight", 0.20)

        if equity_weight > max_eq:
            actions.append(RebalanceAction(
                action="reduce",
                instrument="EQUITY_BASKET",
                target_weight=round(max_eq, 4),
                current_weight=round(equity_weight, 4),
                justification=(
                    f"{risk_profile} profile: equity allocation {equity_weight:.1%} exceeds "
                    f"policy maximum {max_eq:.0%}. Rotate excess into debt instruments."
                ),
            ))

        if debt_weight < min_debt:
            shortfall = min_debt - debt_weight
            actions.append(RebalanceAction(
                action="add",
                instrument="DEBT_MF_OR_GSEC",
                target_weight=round(min_debt, 4),
                current_weight=round(debt_weight, 4),
                justification=(
                    f"{risk_profile} profile: debt allocation {debt_weight:.1%} below "
                    f"policy minimum {min_debt:.0%}. Add ₹{shortfall * 100:.0f}% allocation "
                    f"to debt mutual funds or G-Secs."
                ),
            ))
        return actions

    def _check_geographic_concentration(self, concentration: dict) -> list[RebalanceAction]:
        actions: list[RebalanceAction] = []
        max_geo = self.global_thresholds.get("max_geographic_concentration", 0.80)
        for geo, weight in concentration["geo_weights"].items():
            if weight > max_geo:
                actions.append(RebalanceAction(
                    action="add",
                    instrument="INTERNATIONAL_EQUITY_ETF",
                    target_weight=round(1.0 - max_geo, 4),
                    current_weight=0.0,
                    justification=(
                        f"Geographic concentration in {geo}: {weight:.1%} exceeds "
                        f"policy cap {max_geo:.0%}. Add international equity exposure for diversification."
                    ),
                ))
        return actions

    def _suggest_additions(
        self,
        concentration: dict,
        limits: dict,
        suggestions: list[InvestmentSuggestion],
        holdings: list[dict],
    ) -> list[RebalanceAction]:
        """Recommend adding high-confidence suggestions that reduce concentration gaps."""
        actions: list[RebalanceAction] = []
        existing_tickers = {h["ticker"] for h in holdings}

        for s in suggestions:
            if s["instrument"] in existing_tickers:
                continue
            if s["confidence"] >= 0.80:
                actions.append(RebalanceAction(
                    action="add",
                    instrument=s["instrument"],
                    target_weight=round(0.05, 4),
                    current_weight=0.0,
                    justification=(
                        f"New position: {s['instrument']} recommended by Portfolio Suggester "
                        f"(confidence {s['confidence']:.0%}). {s['rationale']}"
                    ),
                ))
        return actions[:3]   # cap additions to avoid overwhelming the RM

    @staticmethod
    def _cross_reference_suggestions(
        actions: list[RebalanceAction],
        suggestions: list[InvestmentSuggestion],
    ) -> list[RebalanceAction]:
        """Remove 'reduce' actions that directly conflict with active 'buy' suggestions."""
        suggested_instruments = {s["instrument"] for s in suggestions if s["confidence"] >= 0.75}
        filtered: list[RebalanceAction] = []
        for action in actions:
            if action["action"] == "reduce" and action["instrument"] in suggested_instruments:
                logger.debug(
                    "Conflict resolved: dropping reduce(%s) — active suggestion exists",
                    action["instrument"],
                )
                continue
            filtered.append(action)
        return filtered
