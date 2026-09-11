"""Unit tests for batched document ingestion without requiring a Mistral API key or backend."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from entrypoints.ingest import _TEXT_SUFFIXES, _collect_documents


def test_collect_documents_single_file(tmp_path: Path):
    f = tmp_path / "single.txt"
    f.write_text("hello")
    assert _collect_documents(f) == [f]


def test_collect_documents_directory_ignores_hidden_files(tmp_path: Path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.pdf"
    hidden = tmp_path / ".DS_Store"
    f1.write_text("a")
    f2.write_text("b")
    hidden.write_text("hidden")

    docs = _collect_documents(tmp_path)
    assert f1 in docs
    assert f2 in docs
    assert hidden not in docs


def test_collect_documents_nonexistent(tmp_path: Path):
    with pytest.raises(SystemExit):
        _collect_documents(tmp_path / "nonexistent")


def test_ingest_batches_documents_by_pipeline(tmp_path: Path):
    """Verify text and document/image files are partitioned and batched into respective pipelines."""
    f_txt = tmp_path / "doc.txt"
    f_md = tmp_path / "notes.md"
    f_pdf = tmp_path / "paper.pdf"
    f_png = tmp_path / "chart.png"
    f_doc = tmp_path / "memo.doc"

    for f in [f_txt, f_md, f_pdf, f_png, f_doc]:
        f.write_text("sample content")

    documents = [f_txt, f_md, f_pdf, f_png, f_doc]

    text_docs = [p for p in documents if p.suffix.lower() in _TEXT_SUFFIXES]
    ocr_docs = [p for p in documents if p.suffix.lower() not in _TEXT_SUFFIXES]

    assert text_docs == [f_txt, f_md]
    assert ocr_docs == [f_pdf, f_png, f_doc]

    plain_text_pipeline = MagicMock()
    plain_text_pipeline.run = AsyncMock(return_value=3)

    ocr_pipeline = MagicMock()
    ocr_pipeline.run = AsyncMock(return_value=5)

    async def run_batch():
        total = 0
        if text_docs:
            total += await plain_text_pipeline.run(documents=text_docs, use_checkpoint=False)
        if ocr_docs:
            total += await ocr_pipeline.run(documents=ocr_docs, use_checkpoint=False)
        return total

    total_chunks = asyncio.run(run_batch())

    assert total_chunks == 8
    plain_text_pipeline.run.assert_awaited_once_with(documents=text_docs, use_checkpoint=False)
    ocr_pipeline.run.assert_awaited_once_with(documents=ocr_docs, use_checkpoint=False)
