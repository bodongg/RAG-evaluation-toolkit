"""Integration tests for the persistent vector store."""

from pathlib import Path

import pytest

from src.vector_store import build_vector_store, retrieve


@pytest.fixture
def chunks() -> list[dict]:
    """Return chunks about three clearly different topics."""
    return [
        {
            "text": "The Moon is Earth's only natural satellite and orbits our planet.",
            "source": "space.txt",
            "path": "docs/space.txt",
            "chunk_id": 0,
        },
        {
            "text": "Pizza dough is made with flour, water, yeast, and salt.",
            "source": "cooking.txt",
            "path": "docs/cooking.txt",
            "chunk_id": 0,
        },
        {
            "text": "Jazz musicians improvise melodies on saxophones and pianos.",
            "source": "music.txt",
            "path": "docs/music.txt",
            "chunk_id": 1,
        },
    ]


def test_build_then_retrieve_returns_k_results(tmp_path: Path, chunks: list[dict]) -> None:
    """Persist text and metadata and return the requested number of matches."""
    collection = build_vector_store(
        chunks, persist_dir=str(tmp_path), collection_name="test_documents"
    )
    results = retrieve(
        "What orbits Earth?", k=2,
        persist_dir=str(tmp_path), collection_name="test_documents"
    )

    assert len(results) == 2
    assert all(set(result) == {"text", "source", "chunk_id", "distance"} for result in results)
    assert results[0]["distance"] <= results[1]["distance"]
    stored = collection.get(ids=["music.txt-1"])
    assert stored["documents"] == [chunks[2]["text"]]
    assert stored["metadatas"] == [
        {"source": "music.txt", "path": "docs/music.txt", "chunk_id": 1}
    ]


def test_most_relevant_chunk_ranks_first(tmp_path: Path, chunks: list[dict]) -> None:
    """A question about Earth's satellite ranks the Moon passage first."""
    build_vector_store(chunks, persist_dir=str(tmp_path))
    results = retrieve("What is Earth's natural satellite?", persist_dir=str(tmp_path))

    assert results[0]["text"] == chunks[0]["text"]
    assert results[0]["source"] == "space.txt"
    assert results[0]["chunk_id"] == 0


def test_rebuild_replaces_existing_collection(tmp_path: Path, chunks: list[dict]) -> None:
    """Rebuilding replaces old records rather than retaining or duplicating them."""
    build_vector_store(chunks, persist_dir=str(tmp_path))
    collection = build_vector_store(chunks[:1], persist_dir=str(tmp_path))

    assert collection.count() == 1
    assert collection.get()["ids"] == ["space.txt-0"]


def test_empty_chunks_raise_clear_error(tmp_path: Path) -> None:
    """Empty input is rejected before attempting to build a collection."""
    with pytest.raises(ValueError, match="empty"):
        build_vector_store([], persist_dir=str(tmp_path))
