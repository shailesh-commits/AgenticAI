"""APScheduler wrapper — triggers intelligence cycles on a configurable interval.

Default: every 5 hours (configurable via config.yaml scheduler.interval_hours).
"""

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from orchestrator.graph import run_cycle
from orchestrator.state_store import StateStore

logger = logging.getLogger(__name__)


def _cycle_job(config: dict, state_store: StateStore) -> None:
    """APScheduler job function — wraps run_cycle with error isolation."""
    try:
        logger.info("Scheduler: triggering intelligence cycle")
        final_state = run_cycle(config, state_store)
        errors = final_state.get("errors", [])
        if errors:
            logger.warning("Cycle completed with %d error(s): %s", len(errors), errors)
        else:
            logger.info(
                "Cycle %s completed successfully — %d leads, %d clients",
                final_state.get("cycle_id"),
                len(final_state.get("lead_feed", [])),
                len(final_state.get("suggestions", {})),
            )
    except Exception:
        logger.exception("Scheduler: cycle job raised unhandled exception")


class CycleScheduler:
    """Manages the recurring intelligence cycle via APScheduler."""

    def __init__(self, config: dict, state_store: StateStore):
        self.config = config
        self.state_store = state_store
        interval_hours: int = config.get("scheduler", {}).get("interval_hours", 5)
        max_instances: int = config.get("scheduler", {}).get("max_instances", 1)

        self._scheduler = BackgroundScheduler()
        self._scheduler.add_job(
            _cycle_job,
            trigger=IntervalTrigger(hours=interval_hours),
            args=[config, state_store],
            id="intelligence_cycle",
            max_instances=max_instances,
            replace_existing=True,
        )
        logger.info("CycleScheduler initialised — interval: %d hour(s)", interval_hours)

    def start(self) -> None:
        self._scheduler.start()
        logger.info("CycleScheduler started")

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("CycleScheduler stopped")

    def run_now(self) -> None:
        """Immediately trigger one cycle outside the schedule."""
        _cycle_job(self.config, self.state_store)
