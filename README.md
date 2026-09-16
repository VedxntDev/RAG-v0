# Streaming Live RAG Engine

A full-duplex retrieval-augmented answering engine that streams transcript chunks, begins retrieving before the user finishes speaking, decomposes compound utterances into sub-queries, fuses evidence using Reciprocal Rank Fusion (RRF), refines answers in-place upon late user details, and emits structured JSON telemetry.

---

## Technical Evaluation Gates Status

- **G1 Reproducibility**: Single-command launch (`./run.sh all` or `docker compose up`); automated tests pass 100%.
- **G2 Early Retrieval**: $\ge 80\%$ early retrieval on eligible streaming prompts; presentation-only turns suppressed with zero vector search calls.
- **G3 Multi-Intent Identification**: $\ge 70\%$ compound utterances isolated into distinct sub-queries with over-fragmentation guards.
- **G4 Factual Grounding**: $\ge 85\%$ assertions backed by verifiable corpus section citations (`[Doc_12 §2]`); 0 hallucinated document IDs.
- **G5 Session Refinement**: Late constraints update answer in-place, incrementing `answer_version` ($v1 \to v2$) without clearing state or re-searching full corpus.
- **G6 Telemetry & Observability**: $100\%$ structured JSON telemetry compliance recording TTFT, total latency, token cost, citations, and version lineage.

---

## Quick Start & One-Command Execution

```bash
# Clone and enter workspace
cd /Users/vedantbaghel/.gemini/antigravity/scratch/streaming_live_rag

# Run full test suite, benchmarks, and ablation studies in one command
./run.sh all

# Or run individual modes:
./run.sh test     # Run automated pytest test suite
./run.sh bench    # Run benchmarking evaluation vs Naive Batch RAG
./run.sh ablate   # Run ablation studies (Hybrid vs Dense, Heuristic vs Model controller)
./run.sh serve    # Launch FastAPI REST & WebSocket streaming server on port 8000
```

### Single-Command Docker Launch

```bash
docker compose up --build
```

---

## Project Structure

```
streaming_live_rag/
├── run.sh                      # CLI runner script
├── Dockerfile & docker-compose.yml # Containerization setup
├── requirements.txt            # Pinned dependencies
├── .env.example                # Config template
├── config.py                   # Configuration settings
├── data/
│   ├── sample_corpus.json      # Corpus documents with section markers
│   └── test_utterances.json    # Streaming transcript test scenarios
├── src/
│   ├── corpus/                 # Sparse BM25 + Dense indexer
│   ├── simulator/              # Timestamped transcript stream simulator
│   ├── controller/             # Zero-LLM heuristic retrieval controller
│   ├── decomposer/             # Multi-intent decomposer with deduplication
│   ├── retrieval/              # Hybrid BM25 + Dense RRF fusion
│   ├── synthesis/              # Grounded synthesizer & session delta refiner
│   ├── llm/                    # Gemini / OpenAI / Mock provider
│   ├── telemetry/              # Structured JSON event logger
│   ├── engine.py               # Streaming Live RAG orchestrator
│   └── api/server.py           # FastAPI REST & WebSocket server
├── tests/                      # Automated test suite for Gates G1-G6
├── benchmark/                  # Benchmark and ablation scripts
├── system_architecture_brief.md# Architecture Brief deliverable
├── benchmark_and_evaluation_report.md # Benchmarks & Ablation Report
├── telemetry_schema.md         # JSON Telemetry Schema deliverable
└── demo_walkthrough.md         # Demo Guide & Video Script deliverable
```
