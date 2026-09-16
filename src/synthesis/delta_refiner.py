from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Optional

@dataclass
class SessionState:
    session_id: str
    version: int = 1
    last_query: Optional[str] = None
    last_answer: Optional[str] = None
    last_citations: List[str] = field(default_factory=list)
    accumulated_chunk_ids: Set[str] = field(default_factory=set)
    turn_history: List[Dict[str, Any]] = field(default_factory=list)

    def reset(self):
        self.version = 1
        self.last_query = None
        self.last_answer = None
        self.last_citations.clear()
        self.accumulated_chunk_ids.clear()
        self.turn_history.clear()

class SessionDeltaRefiner:
    """
    Manages session-bound state and incremental answer refinement.
    Detects late-arriving constraints/corrections, dispatches targeted delta searches,
    mutates affected claims in-place, and increments answer version without restarting pipeline.
    """
    DELTA_KEYWORDS = [
        "oh", "forgot", "actually", "international", "domestic", "what if", "instead",
        "update", "correction", "trip was", "event was", "budget is"
    ]

    def is_delta_update(self, utterance: str, session: SessionState) -> bool:
        if not session.last_answer:
            return False
        text = utterance.lower()
        return any(kw in text for kw in self.DELTA_KEYWORDS)

    def update_session(
        self,
        session: SessionState,
        query: str,
        answer: str,
        citations: List[str],
        is_delta: bool
    ) -> int:
        if is_delta:
            session.version += 1
        else:
            session.version = 1

        session.last_query = query
        session.last_answer = answer
        session.last_citations = list(dict.fromkeys(session.last_citations + citations if is_delta else citations))
        session.accumulated_chunk_ids.update(citations)
        session.turn_history.append({
            "version": session.version,
            "query": query,
            "answer": answer,
            "citations": list(citations),
            "is_delta": is_delta
        })
        return session.version
