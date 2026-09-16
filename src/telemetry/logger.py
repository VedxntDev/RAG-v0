import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

@dataclass
class RetrievalEvent:
    timestamp_s: float
    query: str
    trigger: str  # "provisional" | "multi_intent" | "delta"

@dataclass
class TelemetryStats:
    time_to_first_token_ms: float
    total_latency_ms: float
    token_cost: float
    retrieval_recall_at_k: Optional[float] = None

@dataclass
class StructuredOutputEvent:
    retrieval_events: List[RetrievalEvent] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    answer: str = ""
    answer_version: int = 1
    citations: List[str] = field(default_factory=list)
    uncertainty: Optional[str] = None
    telemetry: TelemetryStats = field(default_factory=lambda: TelemetryStats(0.0, 0.0, 0.0))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class TelemetryLogger:
    """
    Structured telemetry logger ensuring 100% compliance with Gate G6 log schema.
    """
    def __init__(self):
        self.logs: List[StructuredOutputEvent] = []

    def create_event(
        self,
        retrieval_events: List[Dict[str, Any]],
        sub_queries: List[str],
        answer: str,
        answer_version: int,
        citations: List[str],
        uncertainty: Optional[str],
        ttft_ms: float,
        total_latency_ms: float,
        token_cost: float,
        recall_at_k: Optional[float] = None
    ) -> StructuredOutputEvent:
        events = [
            RetrievalEvent(
                timestamp_s=e["timestamp_s"],
                query=e["query"],
                trigger=e["trigger"]
            )
            for e in retrieval_events
        ]
        stats = TelemetryStats(
            time_to_first_token_ms=round(ttft_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            token_cost=round(token_cost, 6),
            retrieval_recall_at_k=recall_at_k
        )
        event = StructuredOutputEvent(
            retrieval_events=events,
            sub_queries=sub_queries,
            answer=answer,
            answer_version=answer_version,
            citations=citations,
            uncertainty=uncertainty,
            telemetry=stats
        )
        self.logs.append(event)
        return event
