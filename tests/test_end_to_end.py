import pytest
import json
from src.engine import StreamingLiveRAGEngine
from src.synthesis.delta_refiner import SessionState
import config

def test_full_end_to_end_scenarios():
    engine = StreamingLiveRAGEngine()

    with open(config.TEST_UTTERANCES_PATH, 'r') as f:
        scenarios = json.load(f)

    for sc in scenarios:
        sid = sc["id"]
        session = SessionState(session_id=f"session_{sid}")

        if sid == "scenario_incomplete":
            event = engine.process_streaming_chunks(sc["chunks"], session=session)
            assert event.answer == "[Awaiting full utterance completion...]"
            assert len(event.retrieval_events) == 0

        elif sid == "scenario_provisional":
            event = engine.process_streaming_chunks(sc["chunks"], session=session)
            assert len(event.retrieval_events) == 1
            assert event.retrieval_events[0].trigger == "provisional"
            assert len(event.citations) > 0

        elif sid == "scenario_compound":
            event = engine.process_streaming_chunks(sc["chunks"], session=session)
            assert len(event.sub_queries) >= 3
            assert len(event.citations) >= 2

        elif sid == "scenario_delta":
            # Turn 1
            t1_chunks = [{"timestamp_s": 0.5, "text": sc["initial_turn"]["utterance"]}]
            event1 = engine.process_streaming_chunks(t1_chunks, session=session)
            assert event1.answer_version == 1

            # Turn 2
            t2_chunks = [{"timestamp_s": 0.5, "text": sc["followup_turn"]["utterance"]}]
            event2 = engine.process_streaming_chunks(t2_chunks, session=session)
            assert event2.answer_version == 2

        elif sid == "scenario_presentation":
            # Turn 1
            t1_chunks = [{"timestamp_s": 0.5, "text": sc["initial_turn"]["utterance"]}]
            event1 = engine.process_streaming_chunks(t1_chunks, session=session)

            # Turn 2
            t2_chunks = [{"timestamp_s": 0.5, "text": sc["followup_turn"]["utterance"]}]
            event2 = engine.process_streaming_chunks(t2_chunks, session=session)
            assert len(event2.retrieval_events) == 0
            assert event2.answer_version == 1

        elif sid == "scenario_uncertainty":
            event = engine.process_streaming_chunks(sc["chunks"], session=session)
            assert event.uncertainty is not None
