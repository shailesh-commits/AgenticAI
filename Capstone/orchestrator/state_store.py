"""SQLite-backed state store for lead feed, suggestions, and diversification reports."""

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import Column, DateTime, Float, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class LeadRecord(Base):
    __tablename__ = "leads"

    id = Column(String(64), primary_key=True)   # md5 of url
    cycle_id = Column(String(64), nullable=False)
    ticker = Column(String(32))
    headline = Column(Text)
    trigger = Column(String(32))
    sentiment = Column(String(16))
    score = Column(Float)
    url = Column(Text)
    ts = Column(DateTime)
    source = Column(String(32))
    created_at = Column(DateTime, default=datetime.utcnow)


class SuggestionRecord(Base):
    __tablename__ = "suggestions"

    id = Column(String(128), primary_key=True)  # {client_id}_{cycle_id}
    client_id = Column(String(64), nullable=False)
    cycle_id = Column(String(64), nullable=False)
    data = Column(Text)                         # JSON-serialised list
    created_at = Column(DateTime, default=datetime.utcnow)


class DiversificationRecord(Base):
    __tablename__ = "diversification_reports"

    id = Column(String(128), primary_key=True)
    client_id = Column(String(64), nullable=False)
    cycle_id = Column(String(64), nullable=False)
    data = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class StateStore:
    def __init__(self, config: dict):
        backend = config.get("state_store", {}).get("backend", "sqlite")
        if backend == "sqlite":
            db_path = config["state_store"].get("sqlite_path", "./data/state.db")
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            url = f"sqlite:///{db_path}"
        elif backend == "postgresql":
            url = config["state_store"]["postgresql_url"]
        else:
            url = "sqlite:///./data/state.db"

        self.engine = create_engine(url, connect_args={"check_same_thread": False} if "sqlite" in url else {})
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info("StateStore initialised: backend=%s", backend)

    # ── Lead feed ────────────────────────────────────────────────────────────

    def get_recent_urls(self, hours: int = 48) -> set:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        with self.Session() as session:
            rows = session.execute(
                text("SELECT url FROM leads WHERE created_at >= :cutoff"),
                {"cutoff": cutoff},
            ).fetchall()
        return {row[0] for row in rows}

    def save_leads(self, cycle_id: str, leads: List[dict]) -> None:
        with self.Session() as session:
            for lead in leads:
                record_id = hashlib.md5(lead["url"].encode()).hexdigest()
                ts = datetime.fromisoformat(lead["ts"].replace("Z", "+00:00")) if lead.get("ts") else datetime.utcnow()
                session.merge(
                    LeadRecord(
                        id=record_id,
                        cycle_id=cycle_id,
                        ticker=lead.get("ticker", ""),
                        headline=lead.get("headline", ""),
                        trigger=lead.get("trigger", ""),
                        sentiment=lead.get("sentiment", "neutral"),
                        score=lead.get("score", 0.0),
                        url=lead.get("url", ""),
                        ts=ts.replace(tzinfo=None),
                        source=lead.get("source", ""),
                    )
                )
            session.commit()
        logger.debug("Saved %d leads for cycle %s", len(leads), cycle_id)

    def get_leads(self, cycle_id: Optional[str] = None, limit: int = 100) -> List[dict]:
        with self.Session() as session:
            query = session.query(LeadRecord)
            if cycle_id:
                query = query.filter(LeadRecord.cycle_id == cycle_id)
            rows = query.order_by(LeadRecord.created_at.desc()).limit(limit).all()
        return [self._lead_to_dict(r) for r in rows]

    @staticmethod
    def _lead_to_dict(record: LeadRecord) -> dict:
        return {
            "ticker": record.ticker,
            "headline": record.headline,
            "trigger": record.trigger,
            "sentiment": record.sentiment,
            "score": record.score,
            "url": record.url,
            "ts": record.ts.isoformat() if record.ts else "",
            "source": record.source,
        }

    # ── Suggestions ───────────────────────────────────────────────────────────

    def save_suggestions(self, client_id: str, cycle_id: str, suggestions: List[dict]) -> None:
        record_id = f"{client_id}_{cycle_id}"
        with self.Session() as session:
            session.merge(
                SuggestionRecord(
                    id=record_id,
                    client_id=client_id,
                    cycle_id=cycle_id,
                    data=json.dumps(suggestions),
                )
            )
            session.commit()

    def get_suggestions(self, client_id: str, cycle_id: Optional[str] = None) -> List[dict]:
        with self.Session() as session:
            query = session.query(SuggestionRecord).filter(SuggestionRecord.client_id == client_id)
            if cycle_id:
                query = query.filter(SuggestionRecord.cycle_id == cycle_id)
            record = query.order_by(SuggestionRecord.created_at.desc()).first()
        return json.loads(record.data) if record else []

    # ── Diversification reports ───────────────────────────────────────────────

    def save_diversification_report(self, client_id: str, cycle_id: str, report: List[dict]) -> None:
        record_id = f"{client_id}_{cycle_id}"
        with self.Session() as session:
            session.merge(
                DiversificationRecord(
                    id=record_id,
                    client_id=client_id,
                    cycle_id=cycle_id,
                    data=json.dumps(report),
                )
            )
            session.commit()

    def get_diversification_report(self, client_id: str, cycle_id: Optional[str] = None) -> List[dict]:
        with self.Session() as session:
            query = session.query(DiversificationRecord).filter(
                DiversificationRecord.client_id == client_id
            )
            if cycle_id:
                query = query.filter(DiversificationRecord.cycle_id == cycle_id)
            record = query.order_by(DiversificationRecord.created_at.desc()).first()
        return json.loads(record.data) if record else []
