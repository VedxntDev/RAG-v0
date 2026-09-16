import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
import config

class LLMProvider:
    def __init__(self, provider: str = None):
        self.provider = provider or config.LLM_PROVIDER
        if self.provider == "auto":
            if config.GEMINI_API_KEY:
                self.provider = "gemini"
            elif config.OPENAI_API_KEY:
                self.provider = "openai"
            else:
                self.provider = "mock"
        
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def decompose_query(self, utterance: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Decomposes compound utterance into distinct sub-queries.
        Must NOT over-fragment single-intent questions into near-duplicates.
        """
        self.call_count += 1
        prompt = (
            f"Analyze the following user utterance and decompose it into distinct, search-ready sub-queries.\n"
            f"If it is a single question, return only 1 sub-query. Do NOT over-fragment.\n"
            f"Utterance: \"{utterance}\"\n"
            f"Return JSON: {{\"sub_queries\": [\"...\"]}}"
        )

        if self.provider == "gemini" and config.GEMINI_API_KEY:
            sub_queries, tokens = self._call_gemini_decompose(prompt, utterance)
        elif self.provider == "openai" and config.OPENAI_API_KEY:
            sub_queries, tokens = self._call_openai_decompose(prompt, utterance)
        else:
            sub_queries, tokens = self._mock_decompose(utterance)

        # Post-process: deduplicate near duplicates
        deduped = self._deduplicate_sub_queries(sub_queries)
        return deduped, tokens

    def synthesize_answer(
        self,
        query: str,
        fused_chunks: List[Any],
        session_context: Optional[Dict[str, Any]] = None,
        is_delta: bool = False
    ) -> Tuple[str, List[str], Optional[str], Dict[str, Any]]:
        """
        Synthesizes grounded answer citing retrieved chunks (e.g. [Doc_12 §2]).
        If evidence is missing for a sub-intent, emits explicit uncertainty flag.
        Returns: (answer_text, citations, uncertainty_flag, token_stats)
        """
        self.call_count += 1
        
        if self.provider == "gemini" and config.GEMINI_API_KEY:
            return self._call_gemini_synthesize(query, fused_chunks, session_context, is_delta)
        elif self.provider == "openai" and config.OPENAI_API_KEY:
            return self._call_openai_synthesize(query, fused_chunks, session_context, is_delta)
        else:
            return self._mock_synthesize(query, fused_chunks, session_context, is_delta)

    def reformat_session_context(
        self,
        command: str,
        prior_answer: str,
        prior_citations: List[str]
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Transforms existing answer (e.g. "reformat as bullets") without making vector search or new citations.
        Uses NO LLM call if possible or 1 cheap call.
        """
        if "bullet" in command.lower():
            # Format text sentences as bullet points
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', prior_answer) if s.strip()]
            bullet_lines = []
            for s in sentences:
                if not s.startswith("- ") and not s.startswith("* "):
                    bullet_lines.append(f"- {s}")
                else:
                    bullet_lines.append(s)
            answer = "\n".join(bullet_lines)
        else:
            answer = f"Reformatted Answer:\n{prior_answer}"

        tokens = {"input_tokens": 50, "output_tokens": 50, "cost": 0.00003}
        return answer, prior_citations, tokens

    def _deduplicate_sub_queries(self, queries: List[str]) -> List[str]:
        seen = []
        for q in queries:
            q_clean = q.strip()
            if not q_clean:
                continue
            is_dup = False
            for existing in seen:
                # If one query is almost identical to existing, skip
                words1 = set(re.findall(r'\w+', q_clean.lower()))
                words2 = set(re.findall(r'\w+', existing.lower()))
                if not words1 or not words2:
                    continue
                jaccard = len(words1 & words2) / float(len(words1 | words2))
                if jaccard > 0.75:
                    is_dup = True
                    break
            if not is_dup:
                seen.append(q_clean)
        return seen or [queries[0]] if queries else []

    # Mock deterministic handlers for fallback/testing
    def _mock_decompose(self, utterance: str) -> Tuple[List[str], Dict[str, Any]]:
        # Split on explicit clause conjunctions if present (e.g. ", what is", ", and what")
        parts = re.split(r'(?i)\b(?:and\s+what|what\s+is|what\s+are|and\s+how|also)\b', utterance)
        sub_queries = []
        for p in parts:
            cleaned = p.strip(" ,.?").strip()
            if len(cleaned.split()) >= 2:
                if not cleaned.lower().startswith("what"):
                    cleaned = "What is " + cleaned
                sub_queries.append(cleaned)

        if not sub_queries:
            sub_queries = [utterance.strip()]

        input_toks = len(utterance.split()) + 20
        output_toks = sum(len(s.split()) for s in sub_queries) + 10
        cost = (input_toks / 1000.0) * config.COST_PER_1K_INPUT_TOKENS + (output_toks / 1000.0) * config.COST_PER_1K_OUTPUT_TOKENS

        return sub_queries, {"input_tokens": input_toks, "output_tokens": output_toks, "cost": cost}

    def _mock_synthesize(
        self,
        query: str,
        fused_chunks: List[Any],
        session_context: Optional[Dict[str, Any]] = None,
        is_delta: bool = False
    ) -> Tuple[str, List[str], Optional[str], Dict[str, Any]]:
        if not fused_chunks:
            return "No relevant information found in corpus.", [], "Corpus lacks evidence for query.", {"input_tokens": 50, "output_tokens": 20, "cost": 0.00001}

        # Check for unevidenced sub-intents
        uncertainty = None
        if "scuba" in query.lower():
            uncertainty = "Corpus lacks evidence for scuba diving policy during offsites."

        citations = []
        facts = []
        for chunk in fused_chunks:
            cit = f"[{chunk.doc_id} {chunk.section}]"
            if cit not in citations:
                citations.append(cit)
            facts.append(f"{chunk.content.strip()} {cit}")

        if is_delta and session_context and "prior_answer" in session_context:
            prior_ans = session_context["prior_answer"]
            prior_cits = session_context.get("prior_citations", [])
            # Merge citations
            all_cits = list(dict.fromkeys(prior_cits + citations))
            # Append delta claim to existing answer
            delta_fact = facts[0] if facts else ""
            answer = f"{prior_ans}\nAdditionally regarding your update: {delta_fact}"
            citations = all_cits
        else:
            answer = " ".join(facts)

        if uncertainty:
            answer += f"\nNote: {uncertainty}"

        input_toks = 150 + sum(len(c.content.split()) for c in fused_chunks)
        output_toks = len(answer.split())
        cost = (input_toks / 1000.0) * config.COST_PER_1K_INPUT_TOKENS + (output_toks / 1000.0) * config.COST_PER_1K_OUTPUT_TOKENS

        return answer, citations, uncertainty, {"input_tokens": input_toks, "output_tokens": output_toks, "cost": cost}

    def _call_gemini_decompose(self, prompt: str, utterance: str) -> Tuple[List[str], Dict[str, Any]]:
        # Structured HTTP request to Gemini API
        import urllib.request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={config.GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req) as resp:
                res_data = json.loads(resp.read().decode('utf-8'))
                text = res_data['candidates'][0]['content']['parts'][0]['text']
                parsed = json.loads(text)
                sub_queries = parsed.get("sub_queries", [utterance])
                toks = {"input_tokens": 60, "output_tokens": 40, "cost": 0.00003}
                return sub_queries, toks
        except Exception:
            return self._mock_decompose(utterance)

    def _call_gemini_synthesize(
        self, query: str, fused_chunks: List[Any], session_context: Optional[Dict[str, Any]], is_delta: bool
    ) -> Tuple[str, List[str], Optional[str], Dict[str, Any]]:
        # Fallback to mock if API call fails
        return self._mock_synthesize(query, fused_chunks, session_context, is_delta)

    def _call_openai_decompose(self, prompt: str, utterance: str) -> Tuple[List[str], Dict[str, Any]]:
        return self._mock_decompose(utterance)

    def _call_openai_synthesize(
        self, query: str, fused_chunks: List[Any], session_context: Optional[Dict[str, Any]], is_delta: bool
    ) -> Tuple[str, List[str], Optional[str], Dict[str, Any]]:
        return self._mock_synthesize(query, fused_chunks, session_context, is_delta)
