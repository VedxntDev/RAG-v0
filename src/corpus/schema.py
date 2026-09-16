from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class CorpusChunk:
    chunk_id: str             # Full ID e.g. "Doc_12 §2"
    doc_id: str               # e.g. "Doc_12"
    section: str              # e.g. "§2"
    title: str                # Title of section
    content: str              # Full text content
    embedding: Optional[List[float]] = field(default=None, repr=False)

    @property
    def citation_label(self) -> str:
        return f"{self.doc_id} {self.section}"
