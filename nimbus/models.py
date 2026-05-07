from dataclasses import dataclass


@dataclass
class Chunk:
    id: int | str
    document_id: int
    document_name: str
    document_kind: str
    chunk_index: int
    text: str
    score: float = 0.0
