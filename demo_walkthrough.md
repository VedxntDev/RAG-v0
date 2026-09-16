# Demo Walkthrough & Video Demonstration Guide

This guide provides a step-by-step walkthrough script, interactive UI details, and instructions for demonstrating the **Streaming Live RAG Engine**.

---

## Demo Agenda (≤5 Minutes)

### 1. Early Retrieval Triggering (0:00 - 1:00)
- **Action**: Stream transcript chunks for: *"What is the venue capacity for Executive Hall B when we host our quarterly client meeting next month?"*
- **Observed Behavior**:
  - `timestamp_s = 0.6`: As soon as *"the venue capacity for Executive Hall B"* is uttered, the Retrieval Controller fires a `provisional` retrieval event before the user finishes speaking.
  - Telemetry log records `"trigger": "provisional"`. Answer streams back immediately with citation `[Doc_12 §1]`.

### 2. Multi-Intent Decomposition & RRF Fusion (1:00 - 2:15)
- **Action**: Stream compound request: *"What is the venue capacity for a customer workshop, what is the event cancellation refund policy, and what catering options are available?"*
- **Observed Behavior**:
  - `MultiIntentDecomposer` extracts 3 discrete search-ready sub-queries.
  - Hybrid search runs in parallel across BM25 and vector index; RRF fuses rankings.
  - Output answer unifies all three answers with grounded citations: `[Doc_12 §1]`, `[Doc_31 §1]`, `[Doc_45 §1]`.

### 3. Session-Aware Delta Refinement (2:15 - 3:15)
- **Action**:
  - *Turn 1*: Ask *"What are the manager pre-approval rules for business travel?"* -> Answer cites `[Doc_58 §1]` (`answer_version: 1`).
  - *Turn 2*: Add late detail: *"Oh, I forgot to mention, the trip was international."*
- **Observed Behavior**:
  - Pipeline does NOT restart.
  - Controller identifies delta constraint (`"trip was international"`).
  - Targeted retrieval dispatches for delta context (`Doc_58 §2`).
  - In-place mutation updates answer, incrementing `answer_version` from 1 to 2, while preserving prior citations (`Doc_58 §1` and `Doc_58 §2`).

### 4. Presentation-Only Query Suppression (3:15 - 4:00)
- **Action**: Follow up with: *"Please reformat that answer as concise bullet points."*
- **Observed Behavior**:
  - Decision = `NO_RETRIEVAL` (`"trigger_reason": "presentation_only_transform"`).
  - $0$ vector/BM25 calls executed. Existing session answer is formatted into bullets without fabricating new citations.

### 5. Citation Traceability & Telemetry Output (4:00 - 5:00)
- **Action**: Inspect live structured JSON telemetry output from `/ws/stream` endpoint.
- **Observed Behavior**:
  - 100% schema compliance verifying `retrieval_events`, `sub_queries`, `answer_version`, `citations`, `uncertainty`, and `telemetry` (TTFT ms, total latency ms, token cost).

---

## Interactive Launch Instructions

```bash
# Option A: Command Line Demonstration & Full Test Suite
./run.sh all

# Option B: Launch FastAPI Real-Time WebSocket Server
./run.sh serve
# Server runs at http://localhost:8000 (WebSocket at ws://localhost:8000/ws/stream)
```
