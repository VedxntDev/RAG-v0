import re
from typing import Tuple, Optional
import config

class RetrievalController:
    """
    Zero-LLM heuristic controller for real-time retrieval decisions.
    Determines whether an utterance requires:
    - WAIT (utterance still forming / incomplete)
    - RETRIEVE (provisional or full search required)
    - NO_RETRIEVAL (presentation-only transform on existing context)
    """

    def evaluate_stream_chunk(
        self,
        accumulated_text: str,
        is_stream_end: bool = False,
        has_session_context: bool = False
    ) -> Tuple[str, Optional[str]]:
        text_clean = accumulated_text.strip().lower()
        words = re.findall(r'\w+', text_clean)

        if not words:
            return "WAIT", "empty_input"

        # 1. Check Presentation-Only / Transform turn
        if has_session_context:
            for kw in config.PRESENTATION_KEYWORDS:
                if kw in text_clean:
                    return "NO_RETRIEVAL", "presentation_only_transform"

        # 2. Check for Incomplete utterance (trailing conjunction/preposition)
        last_word = words[-1]
        if last_word in config.INCOMPLETE_TRAILING_WORDS:
            return "WAIT", f"trailing_incomplete_word_{last_word}"

        if not is_stream_end:
            if len(words) < config.MIN_STABILITY_WORDS:
                return "WAIT", "utterance_too_short"

        # 3. Check if searchable intent has stabilized (provisional search)
        if len(words) >= 4 or is_stream_end:
            trigger_reason = "provisional_clause_stabilized" if not is_stream_end else "final_utterance_complete"
            return "RETRIEVE", trigger_reason

        return "WAIT", "awaiting_more_tokens"
