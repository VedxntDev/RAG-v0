from dataclasses import dataclass
from typing import List, Generator

@dataclass
class TranscriptChunk:
    timestamp_s: float
    text: str

class TranscriptStreamSimulator:
    def __init__(self, chunks: List[dict]):
        self.chunks = [
            TranscriptChunk(timestamp_s=c["timestamp_s"], text=c["text"])
            for c in chunks
        ]

    def stream(self) -> Generator[tuple[float, str, str], None, None]:
        """
        Yields (timestamp_s, chunk_text, accumulated_transcript)
        """
        accumulated = []
        for c in self.chunks:
            accumulated.append(c.text)
            full_text = " ".join(accumulated)
            yield c.timestamp_s, c.text, full_text
