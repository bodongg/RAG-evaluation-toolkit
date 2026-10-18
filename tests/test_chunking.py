"""Tests for character-based document chunking."""

import pytest

from src.chunking import chunk_documents


def test_chunk_lengths_and_complete_text() -> None:
    """Chunks stay within the limit without dropping text."""
    docs = [{"text": "abcdefghijk", "source": "a.txt", "path": "docs/a.txt"}]
    chunks = chunk_documents(docs, chunk_size=4, overlap=0)

    assert [chunk["text"] for chunk in chunks] == ["abcd", "efgh", "ijk"]
    assert all(len(chunk["text"]) <= 4 for chunk in chunks)


def test_consecutive_chunks_overlap() -> None:
    """Adjacent chunks share exactly the requested characters."""
    docs = [{"text": "abcdefghijk", "source": "a.txt", "path": "docs/a.txt"}]
    chunks = chunk_documents(docs, chunk_size=5, overlap=2)

    assert [chunk["text"] for chunk in chunks] == ["abcde", "defgh", "ghijk"]
    for first, second in zip(chunks, chunks[1:]):
        assert first["text"][-2:] == second["text"][:2]


def test_metadata_and_ids_are_kept_per_document() -> None:
    """Metadata follows each document and IDs restart at zero."""
    docs = [
        {"text": "abcde", "source": "a.txt", "path": "docs/a.txt"},
        {"text": "xyz", "source": "b.md", "path": "docs/b.md"},
    ]

    assert chunk_documents(docs, chunk_size=3, overlap=0) == [
        {"text": "abc", "source": "a.txt", "path": "docs/a.txt", "chunk_id": 0},
        {"text": "de", "source": "a.txt", "path": "docs/a.txt", "chunk_id": 1},
        {"text": "xyz", "source": "b.md", "path": "docs/b.md", "chunk_id": 0},
    ]


@pytest.mark.parametrize("chunk_size, overlap", [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)])
def test_invalid_settings_raise_value_error(chunk_size: int, overlap: int) -> None:
    """Invalid settings are rejected even when no documents are given."""
    with pytest.raises(ValueError):
        chunk_documents([], chunk_size=chunk_size, overlap=overlap)


def test_empty_documents_are_skipped() -> None:
    """Empty text produces no chunks and does not stop later documents."""
    docs = [
        {"text": "", "source": "empty.txt", "path": "docs/empty.txt"},
        {"text": "hello", "source": "a.txt", "path": "docs/a.txt"},
    ]

    assert chunk_documents(docs) == [
        {"text": "hello", "source": "a.txt", "path": "docs/a.txt", "chunk_id": 0}
    ]
    assert chunk_documents([]) == []


def test_defaults_use_512_characters_and_50_overlap() -> None:
    """Default settings create the expected full and final chunks."""
    docs = [{"text": "a" * 512 + "b" * 88, "source": "a.txt", "path": "docs/a.txt"}]
    chunks = chunk_documents(docs)

    assert [chunk["text"] for chunk in chunks] == ["a" * 512, "a" * 50 + "b" * 88]
