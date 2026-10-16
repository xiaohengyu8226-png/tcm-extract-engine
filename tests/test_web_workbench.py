"""Workbench downloads must respect stage validity and book asset boundaries."""
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from tcm_extract_engine.core.book import BookWorkspace
from tcm_extract_engine.core.manifest import BookManifest, BookStatus, StageRecord
from tcm_extract_engine.web import app as web


@pytest.fixture
def book(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[BookWorkspace, BookManifest]:
    monkeypatch.setattr(web, "BOOKS_ROOT", tmp_path)
    ws = BookWorkspace(tmp_path, "012345abcdef")
    ws.book_dir.mkdir(parents=True)
    manifest = BookManifest(book_id="012345abcdef", file_name="book.pdf", file_hash="sha256:test")
    manifest.save(ws.book_dir)
    return ws, manifest


async def test_static_workbench_served() -> None:
    async with AsyncClient(transport=ASGITransport(app=web.app), base_url="http://test") as client:
        assert (await client.get("/")).status_code == 200
        assert (await client.get("/static/workbench.js")).status_code == 200
        assert (await client.get("/static/workbench.css")).status_code == 200


async def test_artifacts_require_completed_stage(book: tuple[BookWorkspace, BookManifest]) -> None:
    ws, manifest = book
    path = ws.stage_artifact("extract")
    path.parent.mkdir(parents=True)
    path.write_text("{}\n")
    assert web._available_artifacts(manifest.book_id) == []
    manifest.stages["extract"] = StageRecord(stage="extract", status="DONE")
    manifest.save(ws.book_dir)
    assert web._available_artifacts(manifest.book_id)[0]["name"] == "extracted"
    response = await web.download_artifact(manifest.book_id, "extracted")
    assert Path(response.path) == path
    with pytest.raises(HTTPException) as exc:
        await web.download_artifact(manifest.book_id, "arbitrary")
    assert exc.value.status_code == 404


async def test_asset_cannot_escape_book(book: tuple[BookWorkspace, BookManifest], tmp_path: Path) -> None:
    ws, manifest = book
    inside = ws.book_dir / "image.png"
    inside.write_bytes(b"image")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    blocks = ws.stage_dir("parse") / "blocks.jsonl"
    blocks.parent.mkdir(parents=True)
    blocks.write_text("\n".join(json.dumps({"type": "content", "blockid": key, "asset_path": str(path)})
                                for key, path in (("inside", inside), ("outside", outside))))
    assert Path((await web.image_asset(manifest.book_id, "inside")).path) == inside
    with pytest.raises(HTTPException) as exc:
        await web.image_asset(manifest.book_id, "outside")
    assert exc.value.status_code == 404


def test_failed_stage_and_recovered_message(book: tuple[BookWorkspace, BookManifest]) -> None:
    ws, manifest = book
    manifest.status = BookStatus.FAILED
    manifest.error = "timeout"
    manifest.stages["extract"] = StageRecord(stage="extract", status="FAILED")
    assert web._job_view(manifest.book_id, manifest)["failed_stage"] == "extract"
    web._runtime[manifest.book_id] = {"message": "步骤失败：timeout"}
    try:
        manifest.status = BookStatus.EXTRACTED
        assert web._job_view(manifest.book_id, manifest)["message"] == ""
    finally:
        web._runtime.pop(manifest.book_id, None)


def test_stale_knowledge_hidden(book: tuple[BookWorkspace, BookManifest]) -> None:
    ws, manifest = book
    verified = ws.stage_artifact("verify")
    verified.parent.mkdir(parents=True)
    verified.write_text("{}\n")
    knowledge = ws.book_dir / "08_publish/knowledge/records.jsonl"
    knowledge.parent.mkdir(parents=True)
    knowledge.write_text("{}\n")
    manifest.stages["verify"] = StageRecord(stage="verify", status="DONE")
    manifest.save(ws.book_dir)
    assert "knowledge" in {a["name"] for a in web._available_artifacts(manifest.book_id)}
    manifest.derived_artifacts_stale = True
    manifest.save(ws.book_dir)
    assert "knowledge" not in {a["name"] for a in web._available_artifacts(manifest.book_id)}


def test_batch_progress_counts_states(book: tuple[BookWorkspace, BookManifest]) -> None:
    ws, manifest = book
    manifest.status = BookStatus.EXTRACTING
    ledger = ws.stage_dir("extract") / "batch_ledger.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps([{"status": "DONE"}, {"status": "FAILED"}, {"status": "PENDING"}]))
    assert web._job_view(manifest.book_id, manifest)["batch_progress"] == {
        "total": 3, "completed": 1, "failed": 1,
    }
