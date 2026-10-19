"""Build and search a persistent ChromaDB vector store."""

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer


def build_vector_store(
    chunks: list[dict],
    persist_dir: str = "chroma_db",
    collection_name: str = "documents",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Collection:
    """Embed chunks and replace the named collection with their text and metadata."""
    if not chunks:
        raise ValueError("Cannot build a vector store from empty chunks.")

    texts = [chunk["text"] for chunk in chunks]
    ids = [f"{chunk['source']}-{chunk['chunk_id']}" for chunk in chunks]
    metadatas = [
        {"source": chunk["source"], "path": chunk["path"], "chunk_id": chunk["chunk_id"]}
        for chunk in chunks
    ]
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(texts).tolist()

    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection(name=collection_name)
    except NotFoundError:
        pass
    collection = client.create_collection(name=collection_name, embedding_function=None)
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return collection


def retrieve(
    query: str,
    k: int = 3,
    persist_dir: str = "chroma_db",
    collection_name: str = "documents",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> list[dict]:
    """Return the closest chunks using the same embedding model used to build them."""
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(name=collection_name, embedding_function=None)
    model = SentenceTransformer(embedding_model)
    results = collection.query(
        query_embeddings=model.encode([query]).tolist(),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "text": text,
            "source": metadata["source"],
            "chunk_id": metadata["chunk_id"],
            "distance": distance,
        }
        for text, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]
