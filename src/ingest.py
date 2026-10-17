"""Load local text and PDF documents for ingestion."""

from pathlib import Path

from pypdf import PdfReader


def load_documents(folder: str) -> list[dict]:
    """Return non-empty supported files from a folder and its subfolders."""
    root = Path(folder).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Document folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Document path is not a folder: {root}")

    documents = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue

        if path.suffix.lower() == ".pdf":
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = path.read_text(encoding="utf-8")

        if not text.strip():
            print(f"Warning: skipping empty file: {path}")
            continue

        documents.append({"text": text, "source": path.name, "path": str(path)})

    return documents
