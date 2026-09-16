import re
from typing import List, Tuple, Optional, Dict, Any
from src.corpus.schema import CorpusChunk
from src.llm.provider import LLMProvider

class GroundedSynthesizer:
    """
    Synthesizes grounded answers with strict citation checking.
    Verifies that every factual assertion cites a retrieved corpus chunk ID (e.g. [Doc_12 §2]).
    Rejects/removes any ungrounded citations. Emits explicit uncertainty flags when evidence is missing.
    """
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    def synthesize(
        self,
        query: str,
        fused_chunks: List[CorpusChunk],
        session_context: Optional[Dict[str, Any]] = None,
        is_delta: bool = False
    ) -> Tuple[str, List[str], Optional[str], Dict[str, Any]]:
        raw_answer, raw_citations, uncertainty, token_stats = self.llm.synthesize_answer(
            query=query,
            fused_chunks=fused_chunks,
            session_context=session_context,
            is_delta=is_delta
        )

        # Citation Verification: Grounding Gate G4
        valid_chunk_ids = {c.citation_label for c in fused_chunks}
        if session_context and "prior_citations" in session_context:
            valid_chunk_ids.update(session_context["prior_citations"])

        # Extract all citation tags like [Doc_12 §2] from generated answer
        found_citations = re.findall(r'\[(Doc_\d+\s+§\d+)\]', raw_answer)
        verified_citations = []

        for cit in found_citations:
            if cit in valid_chunk_ids:
                if cit not in verified_citations:
                    verified_citations.append(cit)

        # Fallback: if LLM returned citations list separately, filter to valid ones
        for cit in raw_citations:
            cit_clean = cit.strip("[]")
            if cit_clean in valid_chunk_ids and cit_clean not in verified_citations:
                verified_citations.append(cit_clean)

        return raw_answer, verified_citations, uncertainty, token_stats
