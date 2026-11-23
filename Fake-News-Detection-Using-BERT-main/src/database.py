"""
Persistent SQLite Database & Audit History Engine for Fake News Intelligence.
Logs all verifications with cryptographic IDs, metadata, and export capabilities.
"""
import os
import json
import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class VerificationRecord(Base):
    __tablename__ = "verification_audits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    audit_id = Column(String(64), index=True, unique=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    text = Column(Text, nullable=False)
    source_url = Column(String(512), default="")
    prediction = Column(String(32), index=True)
    confidence = Column(Float)
    fake_probability = Column(Float)
    real_probability = Column(Float)
    model_backend = Column(String(128))
    sensationalism_score = Column(Integer, default=0)
    ai_generated_probability = Column(Float, default=0.0)
    fact_check_summary = Column(String(512), default="")


class AuditDatabase:
    """
    Manages SQLite database connections and historical audit queries.
    """
    def __init__(self, db_path: Optional[str] = None):
        target_path = db_path or os.path.join("outputs", "audit_history.db")
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        self.engine = create_engine(f"sqlite:///{target_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def log_verification(
        self,
        audit_id: str,
        text: str,
        prediction: str,
        confidence: float,
        fake_prob: float,
        real_prob: float,
        model_backend: str,
        source_url: str = "",
        sensationalism_score: int = 0,
        ai_prob: float = 0.0,
        fact_check_summary: str = ""
    ) -> VerificationRecord:
        """Saves a verification event into the database."""
        session = self.SessionLocal()
        try:
            record = VerificationRecord(
                audit_id=audit_id,
                text=text,
                source_url=source_url,
                prediction=prediction,
                confidence=round(confidence, 4),
                fake_probability=round(fake_prob, 4),
                real_probability=round(real_prob, 4),
                model_backend=model_backend,
                sensationalism_score=sensationalism_score,
                ai_generated_probability=round(ai_prob, 4),
                fact_check_summary=fact_check_summary[:500]
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record
        finally:
            session.close()

    def get_history(
        self,
        query: str = "",
        verdict: str = "",
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Retrieves and filters audit history."""
        session = self.SessionLocal()
        try:
            q = session.query(VerificationRecord)
            if verdict and verdict.upper() in ["FAKE", "REAL"]:
                q = q.filter(VerificationRecord.prediction == verdict.upper())
            if query:
                q = q.filter(VerificationRecord.text.ilike(f"%{query}%"))
            
            records = q.order_by(VerificationRecord.timestamp.desc()).offset(offset).limit(limit).all()
            return [
                {
                    "audit_id": r.audit_id,
                    "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "text": r.text,
                    "source_url": r.source_url,
                    "prediction": r.prediction,
                    "confidence": r.confidence,
                    "fake_probability": r.fake_probability,
                    "real_probability": r.real_probability,
                    "sensationalism_score": r.sensationalism_score,
                    "ai_generated_probability": r.ai_generated_probability,
                    "fact_check_summary": r.fact_check_summary
                }
                for r in records
            ]
        finally:
            session.close()

    def get_count(self) -> int:
        session = self.SessionLocal()
        try:
            return session.query(VerificationRecord).count()
        finally:
            session.close()
