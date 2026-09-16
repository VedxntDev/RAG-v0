# System Architecture Brief — Streaming Live RAG Engine

## 1. Executive Summary & Design Rationale

**Streaming Live RAG** is a full-duplex retrieval-augmented answering engine designed to achieve low time-to-first-token (TTFT) and high citation precision without exceeding strict compute and latency budgets. Unlike traditional batch RAG systems—which wait for the user to finish speaking before initiating query processing—Streaming Live RAG operates directly on timestamped streaming transcript chunks.

### Core Architectural Principles:
1. **Architectural Parsimony**: Heavyweight LLM orchestrators and multi-agent loops are intentionally excluded. Every component must justify its latency and compute cost. At most **two LLM calls** occur per conversational turn (one for decomposition, one for synthesis).
2. **Zero-LLM Retrieval Controller**: The decision to WAIT, RETRIEVE, or skip retrieval (`NO_RETRIEVAL`) is made by a zero-LLM heuristic controller running in $<1\text{ ms}$, preventing unnecessary inference costs.
3. **Hybrid RRF Retrieval**: Sparse BM25 and dense embeddings are fused using Reciprocal Rank Fusion (RRF) to maximize document recall across diverse sub-intents.
4. **Session-Bound Delta Refinement**: Late-arriving user constraints trigger targeted delta searches, mutating affected claims in-place and incrementing answer versioning without clearing session context or re-searching the full corpus.

---

## 2. Component Pipeline Architecture

```
[Timestamped Transcript Chunk Stream]
               │
               ▼
   [1] Retrieval Controller  ──────► [Decision: NO_RETRIEVAL] ──► [Reformat Session Context]
       (Zero-LLM Heuristic)                                             (0 Vector/BM25 Calls)
               │
               ▼ (Decision: RETRIEVE)
   [2] Multi-Intent Decomposer ──────► Over-Fragmentation Guard
       (1 LLM Call, JSON Schema)
               │
               ▼
   [3] Hybrid BM25 + Dense Search ──► Reciprocal Rank Fusion (RRF)
                                     (Formula: RRF_k = 60)
               │
               ▼
   [4] Grounded Synthesizer ────────► Verifiable Citation Validator
       (1 LLM Call)                   (Matches [Doc_ID §Section] vs Chunk Index)
               │
               ▼
   [5] Session Delta Refiner ───────► In-Place Claim Mutation & Versioning
                                      (v1 ──► v2)
               │
               ▼
   [Structured Telemetry JSON Event Log]
```

---

## 3. Component Deep Dive

### 3.1 Retrieval Controller (Zero-LLM Heuristics)
The Retrieval Controller inspects incoming transcript chunks word-by-word. It enforces three decisions:
- `WAIT`: Triggered when the utterance is semantically incomplete (e.g., $<3$ words or ending in incomplete conjunctions/prepositions like `"in"`, `"to"`, `"for"`).
- `NO_RETRIEVAL`: Triggered when presentation keywords (`"reformat"`, `"bullet points"`, `"summarize"`, `"shorten"`, `"translate"`) are detected in a turn with existing session context. Suppresses BM25/vector search entirely.
- `RETRIEVE`: Triggered as soon as a searchable clause stabilizes mid-stream (`provisional_search`), firing search before the user finishes speaking.

### 3.2 Multi-Intent Decomposer
Extracts distinct sub-queries from compound utterances. To prevent over-fragmentation of single-intent questions into near-duplicates (which pollutes rankers), the decomposer applies a Jaccard word-overlap similarity filter ($\text{threshold} = 0.75$) to merge near-duplicate queries.

### 3.3 Hybrid BM25 + Dense Retrieval & RRF Fusion
Sub-queries are executed in parallel against BM25 (sparse keyword index) and SentenceTransformer dense vector index. Rankings are fused using Reciprocal Rank Fusion:
$$RRF\_score(chunk) = \sum_{q \in Q} \left( \frac{1}{60 + rank_{BM25}(q, chunk)} + \frac{1}{60 + rank_{dense}(q, chunk)} \right)$$

### 3.4 Grounded Synthesizer & Verifiable Citation Validator
Synthesizes grounded answers where every assertion must cite a verifiable document section marker (e.g. `[Doc_12 §2]`). The validator parses generated text and strips any ungrounded citation tags not present in the retrieved chunk index. When corpus evidence is insufficient for a sub-intent, an explicit uncertainty statement is added.

### 3.5 Session-Aware Delta Refiner
When a user provides a late constraint (e.g., *"the trip was international"*), the delta refiner:
- Retains established session state and prior citations (`Doc_58 §1`).
- Dispatches targeted search for the delta constraint (`Doc_58 §2`).
- Mutates affected claims in-place while preserving unaffected facts.
- Increments `answer_version` from 1 to 2.

---

## 4. Technical Evaluation Gates Compliance

| Gate | Requirement | Implementation & Proof | Status |
|---|---|---|---|
| **G1 Reproducibility** | One-command CLI launch & automated tests | Clean `./run.sh all` & `docker compose up` runner | **PASS** |
| **G2 Early Retrieval** | $\ge 80\%$ early retrieval, low false triggers | Provisional clause stabilization & presentation suppression | **PASS** |
| **G3 Multi-Intent ID** | $\ge 70\%$ compound utterance decomposition | Decomposer with deduplication guard | **PASS** |
| **G4 Grounding** | $\ge 85\%$ verifiable citations, 0 hallucinations | Post-synthesis citation verification step | **PASS** |
| **G5 Session Refinement**| Late constraint update without full restart | In-place claim mutation & answer versioning ($v1 \to v2$) | **PASS** |
| **G6 Telemetry** | 100% structured JSON log schema output | TelemetryLogger capturing TTFT, latency, cost, version | **PASS** |

---

## 5. Mitigation of Failure Modes

1. **Over-Fragmentation**: Mitigated by Jaccard similarity query filtering prior to fusion.
2. **Ungrounded Citations**: Mitigated by mandatory regex index validation against active `fused_chunks`.
3. **Redundant Search on Presentation Turns**: Mitigated by heuristic keyword detector bypassing retrieval completely.
