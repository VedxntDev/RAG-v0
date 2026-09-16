import pytest
from src.telemetry.logger import TelemetryLogger

def test_gate_g6_telemetry_schema():
    logger = TelemetryLogger()

    event = logger.create_event(
        retrieval_events=[{"timestamp_s": 0.8, "query": "test query", "trigger": "provisional"}],
        sub_queries=["test query"],
        answer="Test answer [Doc_12 §1]",
        answer_version=1,
        citations=["Doc_12 §1"],
        uncertainty=None,
        ttft_ms=120.5,
        total_latency_ms=350.2,
        token_cost=0.0012,
        recall_at_k=1.0
    )

    d = event.to_dict()

    assert "retrieval_events" in d
    assert "sub_queries" in d
    assert "answer" in d
    assert "answer_version" in d
    assert "citations" in d
    assert "uncertainty" in d
    assert "telemetry" in d

    t = d["telemetry"]
    assert "time_to_first_token_ms" in t
    assert "total_latency_ms" in t
    assert "token_cost" in t
    assert "retrieval_recall_at_k" in t
