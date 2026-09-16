import pytest
from src.controller.retrieval_controller import RetrievalController

def test_gate_g2_early_retrieval_and_suppression():
    controller = RetrievalController()

    # 1. Incomplete utterance -> WAIT
    incomplete = "I need to plan a customer workshop in"
    decision, reason = controller.evaluate_stream_chunk(incomplete, is_stream_end=False)
    assert decision == "WAIT", f"Expected WAIT for incomplete utterance, got {decision}"

    # 2. Mid-stream searchable stabilization -> RETRIEVE
    provisional = "What is the venue capacity for Executive Hall B"
    decision, reason = controller.evaluate_stream_chunk(provisional, is_stream_end=False)
    assert decision == "RETRIEVE", f"Expected RETRIEVE for stabilized clause, got {decision}"

    # 3. Presentation-only turn suppression -> NO_RETRIEVAL
    presentation = "Please reformat the previous answer as bullet points"
    decision, reason = controller.evaluate_stream_chunk(presentation, is_stream_end=True, has_session_context=True)
    assert decision == "NO_RETRIEVAL", f"Expected NO_RETRIEVAL for presentation query, got {decision}"
