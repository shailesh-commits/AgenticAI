"""WAM Agentic Platform — entry point.

Usage
-----
# Start the API server (with scheduler running in background):
    python main.py

# Run one intelligence cycle immediately and exit:
    python main.py --run-once

# Start API without the background scheduler:
    python main.py --no-scheduler
"""

import argparse
import logging
import os
import sys

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("wam_platform")


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_selectors(path: str = "config/selectors.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="WAM Agentic Platform")
    parser.add_argument("--run-once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--no-scheduler", action="store_true", help="Start API without background scheduler")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    # Change to project root so relative paths in config resolve correctly
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    config = load_config(args.config)
    selectors = load_selectors("config/selectors.yaml")

    from orchestrator.state_store import StateStore
    state_store = StateStore(config)

    if args.run_once:
        logger.info("Running single intelligence cycle…")
        from orchestrator.graph import run_cycle
        final_state = run_cycle(config, state_store, selectors)
        leads = final_state.get("lead_feed", [])
        suggestions = final_state.get("suggestions", {})
        reports = final_state.get("diversification_reports", {})
        errors = final_state.get("errors", [])
        print(f"\n{'='*60}")
        print(f"Cycle ID : {final_state.get('cycle_id')}")
        print(f"Leads    : {len(leads)}")
        print(f"Clients  : {list(suggestions.keys())}")
        for cid, sugg in suggestions.items():
            print(f"\n  [{cid}] Top suggestions:")
            for s in sugg[:3]:
                print(f"    • {s['instrument']:15s}  conf={s['confidence']:.2f}  {s['rationale'][:60]}")
        for cid, report in reports.items():
            print(f"\n  [{cid}] Rebalance actions: {len(report)}")
            for r in report[:3]:
                print(f"    • [{r['action'].upper()}] {r['instrument']:20s}  {r['justification'][:60]}")
        if errors:
            print(f"\nErrors: {errors}")
        print(f"{'='*60}\n")
        sys.exit(0)

    # ── Start API + optional background scheduler ─────────────────────────────
    import uvicorn
    from api.main import create_app

    if not args.no_scheduler:
        from scheduler.scheduler import CycleScheduler
        scheduler = CycleScheduler(config, state_store)
        scheduler.start()
        logger.info("Background scheduler started (interval: %dh)", config.get("scheduler", {}).get("interval_hours", 5))

    api_cfg = config.get("api", {})
    app = create_app()

    uvicorn.run(
        app,
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        log_level=config.get("app", {}).get("log_level", "info").lower(),
    )


if __name__ == "__main__":
    main()
