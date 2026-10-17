from pathlib import Path

import pytest

from src.ingest import load_documents


def test_loads_txt_file(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("A useful document", encoding="utf-8")

    assert load_documents(str(tmp_path)) == [
        {
            "text": "A useful document",
            "source": "notes.txt",
            "path": str(file_path.resolve()),
        }
    ]


def test_loads_markdown_from_subfolder(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    file_path = nested / "notes.md"
    file_path.write_text("# Heading", encoding="utf-8")

    assert load_documents(str(tmp_path))[0]["path"] == str(file_path.resolve())


def test_skips_empty_file_with_warning(tmp_path, capsys):
    (tmp_path / "empty.txt").write_text("   \n", encoding="utf-8")

    assert load_documents(str(tmp_path)) == []
    assert "Warning" in capsys.readouterr().out


def test_missing_folder_raises_clear_error(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Document folder does not exist"):
        load_documents(str(missing))
