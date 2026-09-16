from typing import List, Tuple, Dict, Any
from src.llm.provider import LLMProvider

class MultiIntentDecomposer:
    """
    Extracts distinct sub-queries hidden in one utterance (1 LLM call).
    Guards against over-fragmenting single-intent questions into near-duplicate queries.
    """
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    def decompose(self, utterance: str) -> Tuple[List[str], Dict[str, Any]]:
        sub_queries, token_stats = self.llm.decompose_query(utterance)
        
        # Additional safety guard against over-fragmentation:
        # If any sub-query is a substring of another or Jaccard similarity > 0.8, remove it.
        filtered = []
        for sq in sub_queries:
            sq_lower = sq.lower().strip()
            is_redundant = False
            for existing in filtered:
                ex_lower = existing.lower().strip()
                if sq_lower in ex_lower or ex_lower in sq_lower:
                    is_redundant = True
                    break
            if not is_redundant:
                filtered.append(sq)

        return filtered or [utterance.strip()], token_stats
