import pytest
from src.decomposer.multi_intent import MultiIntentDecomposer
from src.llm.provider import LLMProvider

def test_gate_g3_multi_intent_decomposition():
    llm = LLMProvider(provider="mock")
    decomposer = MultiIntentDecomposer(llm)

    compound_utterance = "What is the venue capacity for a customer workshop, what is the cancellation policy, and what catering options are available?"
    sub_queries, stats = decomposer.decompose(compound_utterance)

    assert len(sub_queries) >= 3, f"Expected at least 3 sub-queries, got {len(sub_queries)}"
    assert any("venue capacity" in sq.lower() for sq in sub_queries)
    assert any("cancellation" in sq.lower() for sq in sub_queries)
    assert any("catering" in sq.lower() for sq in sub_queries)

    # Test single-intent over-fragmentation guard
    single_utterance = "What is the hotel per diem for domestic travel?"
    sub_queries_single, _ = decomposer.decompose(single_utterance)
    assert len(sub_queries_single) == 1, f"Single intent should not be over-fragmented, got {len(sub_queries_single)}"
