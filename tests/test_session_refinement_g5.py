import pytest
from src.engine import StreamingLiveRAGEngine
from src.synthesis.delta_refiner import SessionState

def test_gate_g5_session_refinement_and_versioning():
    engine = StreamingLiveRAGEngine()
    session = SessionState(session_id="test_g5")

    # Turn 1: Initial query
    chunks_t1 = [
        {"timestamp_s": 0.3, "text": "What are the manager pre-approval rules for business travel?"}
    ]
    event_t1 = engine.process_streaming_chunks(chunks_t1, session=session)

    assert event_t1.answer_version == 1, f"Initial turn answer version should be 1, got {event_t1.answer_version}"
    assert "Doc_58 §1" in event_t1.citations

    # Turn 2: Late constraint addition ("trip was international")
    chunks_t2 = [
        {"timestamp_s": 0.3, "text": "Oh, I forgot to mention, the trip was international."}
    ]
    event_t2 = engine.process_streaming_chunks(chunks_t2, session=session)

    assert event_t2.answer_version == 2, f"Delta turn answer version should increment to 2, got {event_t2.answer_version}"
    # Prior citation Doc_58 §1 should be preserved AND new citation Doc_58 §2 added
    assert "Doc_58 §1" in event_t2.citations, "Prior established citations must be preserved."
    assert "Doc_58 §2" in event_t2.citations, "New delta citation must be included."
