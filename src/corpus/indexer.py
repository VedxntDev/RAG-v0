import json
import math
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict
from src.corpus.schema import CorpusChunk

class CorpusIndexer:
    def __init__(self, corpus_path: Path):
        self.corpus_path = corpus_path
        self.chunks: List[CorpusChunk] = []
        self.bm25_model = None
        self.encoder_model = None
        self.use_rank_bm25 = False
        self.use_sentence_transformers = False
        self._tokenized_corpus: List[List[str]] = []
        
        self.load_corpus()
        self.build_index()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def load_corpus(self):
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.chunks = []
        for item in data:
            doc_id = item["doc_id"]
            section = item["section"]
            chunk_id = f"{doc_id} {section}"
            title = item.get("title", "")
            content = item.get("content", "")
            chunk = CorpusChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                section=section,
                title=title,
                content=f"{title}: {content}" if title else content
            )
            self.chunks.append(chunk)

    def build_index(self):
        # 1. Sparse BM25 Index
        try:
            from rank_bm25 import BM25Okapi
            self._tokenized_corpus = [self._tokenize(c.content) for c in self.chunks]
            self.bm25_model = BM25Okapi(self._tokenized_corpus)
            self.use_rank_bm25 = True
        except ImportError:
            self.use_rank_bm25 = False
            self._tokenized_corpus = [self._tokenize(c.content) for c in self.chunks]

        # 2. Dense Vector Index
        self._tokenized_corpus = [self._tokenize(c.content) for c in self.chunks]
        if os.getenv("USE_SENTENCE_TRANSFORMERS", "0") == "1":
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder_model = SentenceTransformer('all-MiniLM-L6-v2')
                embeddings = self.encoder_model.encode([c.content for c in self.chunks], show_progress_bar=False)
                for idx, c in enumerate(self.chunks):
                    c.embedding = embeddings[idx].tolist()
                self.use_sentence_transformers = True
                return
            except Exception:
                pass

        # Fast robust TF-IDF normalized vector representation
        self.use_sentence_transformers = False
        self._build_fallback_dense_index()

    def _build_fallback_dense_index(self):
        vocab: Dict[str, int] = {}
        for doc_tokens in self._tokenized_corpus:
            for t in doc_tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        
        num_docs = len(self.chunks)
        df: Dict[str, int] = {}
        for doc_tokens in self._tokenized_corpus:
            unique_t = set(doc_tokens)
            for t in unique_t:
                df[t] = df.get(t, 0) + 1

        idf = {t: math.log((num_docs + 1) / (df[t] + 1)) + 1.0 for t in vocab}

        for idx, c in enumerate(self.chunks):
            tokens = self._tokenized_corpus[idx]
            vec = [0.0] * len(vocab)
            for t in tokens:
                vec[vocab[t]] += 1.0 * idf[t]
            norm = math.sqrt(sum(v*v for v in vec)) or 1.0
            c.embedding = [v / norm for v in vec]
        self._vocab = vocab
        self._idf = idf

    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[CorpusChunk, float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        if self.use_rank_bm25 and self.bm25_model:
            scores = self.bm25_model.get_scores(query_tokens)
            indexed_scores = [(self.chunks[i], float(scores[i])) for i in range(len(self.chunks))]
        else:
            indexed_scores = []
            q_set = set(query_tokens)
            for i, c in enumerate(self.chunks):
                doc_tokens = self._tokenized_corpus[i]
                overlap = sum(1 for t in doc_tokens if t in q_set)
                indexed_scores.append((c, float(overlap)))

        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]

    def search_dense(self, query: str, top_k: int = 10) -> List[Tuple[CorpusChunk, float]]:
        if self.use_sentence_transformers and self.encoder_model:
            q_emb = self.encoder_model.encode(query, show_progress_bar=False)
            import numpy as np
            q_vec = np.array(q_emb)
            indexed_scores = []
            for c in self.chunks:
                c_vec = np.array(c.embedding)
                sim = float(np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-9))
                indexed_scores.append((c, sim))
        else:
            q_tokens = self._tokenize(query)
            q_vec = [0.0] * len(self._vocab)
            for t in q_tokens:
                if t in self._vocab:
                    q_vec[self._vocab[t]] += 1.0 * self._idf[t]
            norm = math.sqrt(sum(v*v for v in q_vec)) or 1.0
            q_vec = [v / norm for v in q_vec]

            indexed_scores = []
            for c in self.chunks:
                sim = sum(q_vec[i] * c.embedding[i] for i in range(len(q_vec)))
                indexed_scores.append((c, float(sim)))

        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]
