import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from src.corpus.indexer import CorpusIndexer
from src.simulator.transcript_stream import TranscriptStreamSimulator
from src.controller.retrieval_controller import RetrievalController
from src.decomposer.multi_intent import MultiIntentDecomposer
from src.retrieval.hybrid_fusion import HybridFusionRetriever
from src.synthesis.grounded_synthesizer import GroundedSynthesizer
from src.synthesis.delta_refiner import SessionDeltaRefiner, SessionState
from src.telemetry.logger import TelemetryLogger, StructuredOutputEvent
from src.llm.provider import LLMProvider
import config

class StreamingLiveRAGEngine:
    """
    Full-duplex Streaming Live RAG Engine orchestrating controller, decomposer,
    hybrid RRF retrieval, grounded synthesis, delta session refiner, and telemetry logging.
    """
    def __init__(self, corpus_path: Optional[Path] = None, llm_provider: Optional[LLMProvider] = None):
        self.corpus_path = corpus_path or config.CORPUS_PATH
        self.indexer = CorpusIndexer(self.corpus_path)
        self.llm_provider = llm_provider or LLMProvider()
        
        self.controller = RetrievalController()
        self.decomposer = MultiIntentDecomposer(self.llm_provider)
        self.retriever = HybridFusionRetriever(self.indexer)
        self.synthesizer = GroundedSynthesizer(self.llm_provider)
        self.delta_refiner = SessionDeltaRefiner()
        self.telemetry = TelemetryLogger()

    def process_streaming_chunks(
        self,
        chunks: List[dict],
        session: Optional[SessionState] = None
    ) -> StructuredOutputEvent:
        start_time = time.time()
        session = session or SessionState(session_id="session_default")
        simulator = TranscriptStreamSimulator(chunks)

        retrieval_events = []
        triggered_retrieval = False
        has_context = bool(session.last_answer)

        final_query = ""
        total_cost = 0.0

        for idx, (timestamp_s, chunk_text, accumulated) in enumerate(simulator.stream()):
            is_end = (idx == len(chunks) - 1)
            decision, trigger_reason = self.controller.evaluate_stream_chunk(
                accumulated_text=accumulated,
                is_stream_end=is_end,
                has_session_context=has_context
            )

            if decision == "NO_RETRIEVAL":
                ttft_ms = (time.time() - start_time) * 1000.0
                reformatted_ans, cits, tok_stats = self.llm_provider.reformat_session_context(
                    command=accumulated,
                    prior_answer=session.last_answer or "",
                    prior_citations=session.last_citations or []
                )
                total_latency_ms = (time.time() - start_time) * 1000.0
                event = self.telemetry.create_event(
                    retrieval_events=[],
                    sub_queries=[],
                    answer=reformatted_ans,
                    answer_version=session.version,
                    citations=cits,
                    uncertainty=None,
                    ttft_ms=ttft_ms,
                    total_latency_ms=total_latency_ms,
                    token_cost=tok_stats.get("cost", 0.0)
                )
                return event

            elif decision == "RETRIEVE":
                if not triggered_retrieval:
                    triggered_retrieval = True
                    is_delta = self.delta_refiner.is_delta_update(accumulated, session)
                    trigger_label = "delta" if is_delta else ("provisional" if not is_end else "multi_intent")

                    retrieval_events.append({
                        "timestamp_s": timestamp_s,
                        "query": accumulated,
                        "trigger": trigger_label
                    })

                final_query = accumulated

        if not triggered_retrieval or not final_query:
            # Utterance was incomplete (WAIT decision throughout)
            total_latency_ms = (time.time() - start_time) * 1000.0
            return self.telemetry.create_event(
                retrieval_events=[],
                sub_queries=[],
                answer="[Awaiting full utterance completion...]",
                answer_version=session.version,
                citations=[],
                uncertainty=None,
                ttft_ms=total_latency_ms,
                total_latency_ms=total_latency_ms,
                token_cost=0.0
            )

        # Process final accumulated query
        is_delta = self.delta_refiner.is_delta_update(final_query, session)
        sub_queries, decomp_toks = self.decomposer.decompose(final_query)
        total_cost += decomp_toks.get("cost", 0.0)

        fused_chunks = self.retriever.retrieve_and_fuse(sub_queries)

        ttft_ms = (time.time() - start_time) * 1000.0
        session_dict = {
            "prior_answer": session.last_answer,
            "prior_citations": session.last_citations
        } if has_context else None

        answer, citations, uncertainty, synth_toks = self.synthesizer.synthesize(
            query=final_query,
            fused_chunks=fused_chunks,
            session_context=session_dict,
            is_delta=is_delta
        )
        total_cost += synth_toks.get("cost", 0.0)

        version = self.delta_refiner.update_session(
            session=session,
            query=final_query,
            answer=answer,
            citations=citations,
            is_delta=is_delta
        )

        total_latency_ms = (time.time() - start_time) * 1000.0
        recall_at_k = 1.0 if citations else 0.0

        return self.telemetry.create_event(
            retrieval_events=retrieval_events,
            sub_queries=sub_queries,
            answer=answer,
            answer_version=version,
            citations=citations,
            uncertainty=uncertainty,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            token_cost=total_cost,
            recall_at_k=recall_at_k
        )
