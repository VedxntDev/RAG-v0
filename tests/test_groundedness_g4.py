import pytest
from src.synthesis.grounded_synthesizer import GroundedSynthesizer
from src.corpus.schema import CorpusChunk
from src.llm.provider import LLMProvider

def test_gate_g4_grounding_and_uncertainty():
    llm = LLMProvider(provider="mock")
    synthesizer = GroundedSynthesizer(llm)

    retrieved_chunk = CorpusChunk(
        chunk_id="Doc_12 §1",
        doc_id="Doc_12",
        section="§1",
        title="Venue Capacities",
        content="Conference Room A supports up to 25 attendees."
    )

    # 1. Valid Grounded Synthesis
    query = "What is the capacity of Conference Room A?"
    ans, citations, uncertainty, stats = synthesizer.synthesize(query, [retrieved_chunk])

    assert len(citations) > 0, "Synthesized answer must contain citations."
    assert "Doc_12 §1" in citations, f"Citation must be valid chunk ID, got {citations}"

    # 2. Uncertainty Flagging for unevidenced sub-intent
    query_unsupported = "What is the hotel per diem rate and scuba diving policy?"
    ans_u, citations_u, uncertainty_u, stats_u = synthesizer.synthesize(query_unsupported, [retrieved_chunk])
    assert uncertainty_u is not None, "Uncertainty flag must be set when corpus evidence is missing."
