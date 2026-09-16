# Telemetry & Observability Schema Specification

Every retrieval-triggering turn in the **Streaming Live RAG Engine** emits a structured JSON event record satisfying Technical Evaluation Gate G6.

## JSON Schema Definition

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "StructuredOutputEvent",
  "type": "object",
  "properties": {
    "retrieval_events": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "timestamp_s": { "type": "number", "description": "Stream relative timestamp in seconds" },
          "query": { "type": "string", "description": "Accumulated transcript query text" },
          "trigger": { "type": "string", "enum": ["provisional", "multi_intent", "delta"], "description": "Trigger category" }
        },
        "required": ["timestamp_s", "query", "trigger"]
      }
    },
    "sub_queries": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Decomposed sub-queries executed against corpus index"
    },
    "answer": { "type": "string", "description": "Grounded answer text" },
    "answer_version": { "type": "integer", "description": "Version lineage number for session answer" },
    "citations": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Verifiable corpus document section labels (e.g. Doc_12 §2)"
    },
    "uncertainty": {
      "type": ["string", "null"],
      "description": "Explicit uncertainty statement when evidence is incomplete"
    },
    "telemetry": {
      "type": "object",
      "properties": {
        "time_to_first_token_ms": { "type": "number", "description": "Latency to first token output in ms" },
        "total_latency_ms": { "type": "number", "description": "Total turn processing latency in ms" },
        "token_cost": { "type": "number", "description": "Estimated token cost in USD" },
        "retrieval_recall_at_k": { "type": ["number", "null"], "description": "Recall metric against corpus ground truth" }
      },
      "required": ["time_to_first_token_ms", "total_latency_ms", "token_cost"]
    }
  },
  "required": ["retrieval_events", "sub_queries", "answer", "answer_version", "citations", "uncertainty", "telemetry"]
}
```

## Example Event JSON Output

```json
{
  "retrieval_events": [
    {
      "timestamp_s": 0.8,
      "query": "What is the venue capacity for Executive Hall B",
      "trigger": "provisional"
    }
  ],
  "sub_queries": [
    "What is the venue capacity for Executive Hall B?"
  ],
  "answer": "Executive Hall B accommodates up to 100 attendees with stage setup and includes hybrid streaming equipment [Doc_12 §1] [Doc_12 §2].",
  "answer_version": 1,
  "citations": [
    "Doc_12 §1",
    "Doc_12 §2"
  ],
  "uncertainty": null,
  "telemetry": {
    "time_to_first_token_ms": 118.4,
    "total_latency_ms": 290.1,
    "token_cost": 0.000042,
    "retrieval_recall_at_k": 1.0
  }
}
```
