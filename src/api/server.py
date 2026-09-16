import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from src.engine import StreamingLiveRAGEngine
from src.synthesis.delta_refiner import SessionState
from src.corpus.schema import CorpusChunk
import config

app = FastAPI(
    title="Streaming Live RAG Engine Web Experience",
    description="Full-duplex real-time retrieval augmented generation engine web interface",
    version="1.0.0"
)

engine = StreamingLiveRAGEngine()
sessions: Dict[str, SessionState] = {}

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

class StreamingChunk(BaseModel):
    timestamp_s: float
    text: str

class QueryRequest(BaseModel):
    session_id: str = "default_session"
    chunks: List[StreamingChunk]

class CustomChunkRequest(BaseModel):
    doc_id: str
    section: str
    title: Optional[str] = ""
    content: str

@app.get("/", response_class=HTMLResponse)
def index_page():
    html_file = TEMPLATES_DIR / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h1>Streaming Live RAG Engine API</h1>"

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Streaming Live RAG Engine",
        "provider": engine.llm_provider.provider,
        "indexed_chunks": len(engine.indexer.chunks)
    }

@app.get("/api/corpus")
def get_corpus():
    return [
        {
            "doc_id": c.doc_id,
            "section": c.section,
            "title": c.title,
            "content": c.content,
            "citation_label": c.citation_label
        }
        for c in engine.indexer.chunks
    ]

@app.post("/api/corpus/add")
def add_corpus_chunk(req: CustomChunkRequest):
    chunk_id = f"{req.doc_id} {req.section}"
    new_chunk = CorpusChunk(
        chunk_id=chunk_id,
        doc_id=req.doc_id,
        section=req.section,
        title=req.title or "",
        content=f"{req.title}: {req.content}" if req.title else req.content
    )
    engine.indexer.chunks.append(new_chunk)
    # Rebuild indexes
    engine.indexer.build_index()
    return {"status": "success", "message": f"Indexed chunk {chunk_id}", "total_chunks": len(engine.indexer.chunks)}

@app.post("/api/query")
def process_query(request: QueryRequest):
    sid = request.session_id
    if sid not in sessions:
        sessions[sid] = SessionState(session_id=sid)
    session = sessions[sid]

    chunks_data = [{"timestamp_s": c.timestamp_s, "text": c.text} for c in request.chunks]
    event = engine.process_streaming_chunks(chunks_data, session=session)
    return event.to_dict()

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    session_id = f"ws_session_{id(websocket)}"
    session = SessionState(session_id=session_id)

    accumulated_chunks = []
    try:
        while True:
            data = await websocket.receive_text()
            chunk_json = json.loads(data)
            accumulated_chunks.append(chunk_json)

            event = engine.process_streaming_chunks(accumulated_chunks, session=session)
            await websocket.send_text(event.to_json())
    except WebSocketDisconnect:
        pass
