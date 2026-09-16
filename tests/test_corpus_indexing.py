import pytest
from src.corpus.indexer import CorpusIndexer
import config

def test_corpus_indexing_and_search():
    indexer = CorpusIndexer(config.CORPUS_PATH)
    assert len(indexer.chunks) > 0, "Corpus chunks should be loaded."

    # Test Sparse BM25
    bm25_res = indexer.search_bm25("cancellation policy", top_k=3)
    assert len(bm25_res) > 0, "Sparse BM25 search should return results."
    assert any("Doc_31" in chunk.doc_id for chunk, _ in bm25_res), "Doc_31 should be found for cancellation query."

    # Test Dense Search
    dense_res = indexer.search_dense("hotel per diem domestic travel", top_k=3)
    assert len(dense_res) > 0, "Dense search should return results."
    assert any("Doc_58" in chunk.doc_id for chunk, _ in dense_res), "Doc_58 should be found for travel query."
