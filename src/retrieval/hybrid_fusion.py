from typing import List, Dict, Tuple
from src.corpus.schema import CorpusChunk
from src.corpus.indexer import CorpusIndexer
import config

class HybridFusionRetriever:
    """
    Executes hybrid sparse (BM25) + dense retrieval for each sub-query in parallel.
    Fuses rankings across sub-queries using Reciprocal Rank Fusion (RRF).
    Deduplicates overlapping chunks before passing to synthesis.
    """
    def __init__(self, indexer: CorpusIndexer):
        self.indexer = indexer

    def retrieve_and_fuse(
        self,
        sub_queries: List[str],
        top_k_sparse: int = None,
        top_k_dense: int = None,
        rrf_k: int = None,
        fused_top_k: int = None,
        hybrid_enabled: bool = True
    ) -> List[CorpusChunk]:
        top_k_sparse = top_k_sparse or config.BM25_TOP_K
        top_k_dense = top_k_dense or config.DENSE_TOP_K
        rrf_k = rrf_k or config.RRF_K
        fused_top_k = fused_top_k or config.FUSED_TOP_K

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, CorpusChunk] = {}

        for sq in sub_queries:
            # 1. Sparse search
            if hybrid_enabled:
                bm25_results = self.indexer.search_bm25(sq, top_k=top_k_sparse)
                for rank, (chunk, score) in enumerate(bm25_results, start=1):
                    cid = chunk.chunk_id
                    chunk_map[cid] = chunk
                    rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

            # 2. Dense search
            dense_results = self.indexer.search_dense(sq, top_k=top_k_dense)
            for rank, (chunk, score) in enumerate(dense_results, start=1):
                cid = chunk.chunk_id
                chunk_map[cid] = chunk
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

        # Sort by RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        fused_chunks = [chunk_map[cid] for cid in sorted_cids[:fused_top_k]]
        return fused_chunks
