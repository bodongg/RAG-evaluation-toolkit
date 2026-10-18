"""Split documents into overlapping character chunks."""


def chunk_documents(
    docs: list[dict], chunk_size: int = 512, overlap: int = 50
) -> list[dict]:
    """Return chunks with document metadata and IDs starting at zero per document."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[dict] = []
    step = chunk_size - overlap

    for doc in docs:
        text = doc["text"]
        if not text:
            continue

        for chunk_id, start in enumerate(range(0, len(text), step)):
            chunks.append(
                {
                    "text": text[start : start + chunk_size],
                    "source": doc["source"],
                    "path": doc["path"],
                    "chunk_id": chunk_id,
                }
            )
            if start + chunk_size >= len(text):
                break

    return chunks
