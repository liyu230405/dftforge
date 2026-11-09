"""Public artifacts must be downloadable without exposing session internals."""

import asyncio

import httpx
import pytest

import dft_forge.web as web


@pytest.fixture
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "WEB_WORKDIR", tmp_path / "sessions")
    monkeypatch.setattr(web, "WEB_SESSION_CTX", {})
    root = tmp_path / "sessions" / "sample"
    root.mkdir(parents=True)
    return root


def test_artifact_url_roundtrips_reserved_characters(artifact_root):
    directory = artifact_root / "计算输入"
    directory.mkdir()
    path = directory / "结构 #1?value%done.cif"
    path.write_text("offline structure evidence", encoding="utf-8")

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            items = (await client.get("/api/sessions/sample/artifacts")).json()["artifacts"]
            assert len(items) == 1
            item = items[0]
            assert "%23" in item["download_url"] and "%3F" in item["download_url"]
            assert item["sha256"] and item["kind"] == "structure"
            downloaded = await client.get(item["download_url"])
            assert downloaded.status_code == 200
            assert downloaded.content == path.read_bytes()
    asyncio.run(scenario())


def test_internal_files_are_hidden_from_listing_and_direct_download(artifact_root):
    hidden = ["session.json", ".env", ".cache/result.out", "graphs/graph_state.db",
              "graphs/graph_state.db-wal", "graphs/graph_state.db-shm",
              "ledger.sqlite-journal", "ledger.sqlite-wal", "ledger.sqlite3-shm",
              "evidence.db", "manifest.sqlite-wal"]
    for name in hidden:
        path = artifact_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("isolated test fixture")
    (artifact_root / "result.out").write_text("scientific output")
    (artifact_root / "linked.out").symlink_to(artifact_root / "result.out")
    (artifact_root / "linked_dir").symlink_to(artifact_root / "graphs", target_is_directory=True)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            items = (await client.get("/api/sessions/sample/artifacts")).json()["artifacts"]
            assert [item["path"] for item in items] == ["result.out"]
            for path in hidden + ["linked.out", "linked_dir/graph_state.db"]:
                assert (await client.get(f"/api/sessions/sample/artifacts/{path}")).status_code == 404
    asyncio.run(scenario())


def test_external_link_is_neither_listed_nor_downloaded(artifact_root, tmp_path):
    outside = tmp_path / "outside.out"
    outside.write_text("not in the session")
    (artifact_root / "outside.out").symlink_to(outside)
    assert web._session_artifacts("sample") == []
    with pytest.raises(web.HTTPException) as exc:
        web.download_session_artifact("sample", "outside.out")
    assert exc.value.status_code == 404
