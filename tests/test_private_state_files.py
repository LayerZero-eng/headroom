"""Every state file that holds conversation-derived content is created owner-only.

The CCR store and the runtime log already were; the memory databases (facts
extracted from conversations plus the user id they belong to), the vector and
graph stores, the HNSW metadata dump, the licence validation cache and the
native memory directory were created at the process umask — ``0644`` on a
stock host, readable by any local account.

The fix routes them all through ``headroom.fileperms``, so these tests pin
both the primitives and every call site, under a permissive umask so the
result is known to be umask-independent.
"""

from __future__ import annotations

import os
import sqlite3
import stat

import pytest

from headroom import fileperms

pytestmark = pytest.mark.skipif(
    not fileperms.OWNER_ONLY_SUPPORTED, reason="POSIX permission bits only"
)


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture(autouse=True)
def _permissive_umask():
    old = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(old)


# ---- primitives -------------------------------------------------------------


class TestEnsurePrivateFile:
    def test_new_file_is_private_from_birth(self, tmp_path):
        p = tmp_path / "store.db"
        fileperms.ensure_private_file(p)
        assert _mode(p) == 0o600

    def test_existing_wide_file_is_narrowed(self, tmp_path):
        p = tmp_path / "store.db"
        p.write_text("x")
        assert _mode(p) == 0o644  # precondition under the 022 umask
        fileperms.ensure_private_file(p)
        assert _mode(p) == 0o600
        assert p.read_text() == "x"

    def test_symlink_is_refused_and_target_untouched(self, tmp_path):
        target = tmp_path / "victim"
        target.write_text("SECRET")
        before = _mode(target)
        link = tmp_path / "store.db"
        link.symlink_to(target)
        with pytest.raises(PermissionError):
            fileperms.ensure_private_file(link, what="memory store")
        assert _mode(target) == before
        assert target.read_text() == "SECRET"

    def test_directory_is_refused(self, tmp_path):
        d = tmp_path / "store.db"
        d.mkdir()
        with pytest.raises(PermissionError):
            fileperms.ensure_private_file(d)

    def test_narrowing_failure_fails_closed(self, tmp_path, monkeypatch):
        p = tmp_path / "store.db"
        p.write_text("")

        def boom(fd, mode):
            raise PermissionError("cannot fchmod")

        monkeypatch.setattr(os, "fchmod", boom)
        with pytest.raises(PermissionError):
            fileperms.ensure_private_file(p)


class TestConnectPrivateSqlite:
    def test_creates_private_db_and_sidecars(self, tmp_path):
        p = tmp_path / "m.db"
        conn = fileperms.connect_private_sqlite(p)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t(x)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
        assert _mode(p) == 0o600
        # sqlite gives the -wal/-shm sidecars the main file's mode.
        for sidecar in ("m.db-wal", "m.db-shm"):
            if (tmp_path / sidecar).exists():
                assert _mode(tmp_path / sidecar) == 0o600
        conn.close()

    def test_memory_and_uri_pass_through_without_creating_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fileperms.connect_private_sqlite(":memory:").close()
        fileperms.connect_private_sqlite("file:mem?mode=memory&cache=shared", uri=True).close()
        # No file literally named ":memory:" (a bug another store has shipped).
        assert sorted(p.name for p in tmp_path.iterdir()) == []

    def test_refuses_symlinked_db(self, tmp_path):
        target = tmp_path / "victim.db"
        sqlite3.connect(target).close()
        link = tmp_path / "m.db"
        link.symlink_to(target)
        with pytest.raises(PermissionError):
            fileperms.connect_private_sqlite(link)


class TestPrivateDir:
    def test_new_dir_and_missing_parents_are_0700(self, tmp_path):
        d = tmp_path / "a" / "b" / "memories"
        fileperms.private_dir(d)
        assert _mode(d) == 0o700
        assert _mode(tmp_path / "a") == 0o700
        assert _mode(tmp_path / "a" / "b") == 0o700

    def test_existing_dir_left_alone_unless_tighten(self, tmp_path):
        d = tmp_path / "shared"
        d.mkdir()
        assert _mode(d) == 0o755
        fileperms.private_dir(d)
        assert _mode(d) == 0o755
        fileperms.private_dir(d, tighten=True)
        assert _mode(d) == 0o700

    def test_symlinked_dir_is_not_re_permissioned(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "memories"
        link.symlink_to(target)
        fileperms.private_dir(link, tighten=True)
        assert _mode(target) == 0o755


class TestOpenOwnerOnlyWriteMode:
    def test_write_mode_truncates_and_is_private(self, tmp_path):
        p = tmp_path / "cache.json"
        p.write_text("old-and-long")
        assert _mode(p) == 0o644
        fileperms.write_private_text(p, "new")
        assert p.read_text() == "new"
        assert _mode(p) == 0o600

    def test_append_mode_still_appends(self, tmp_path):
        p = tmp_path / "log"
        with fileperms.open_owner_only(p, "a") as fh:
            fh.write("a")
        with fileperms.open_owner_only(p, "a") as fh:
            fh.write("b")
        assert p.read_text() == "ab"

    def test_read_modes_are_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            fileperms.open_owner_only(tmp_path / "x", "r")


# ---- call sites -------------------------------------------------------------


class TestMemoryStoresArePrivate:
    def test_sqlite_memory_store(self, tmp_path):
        from headroom.memory.adapters.sqlite import SQLiteMemoryStore

        p = tmp_path / "headroom_memory.db"
        SQLiteMemoryStore(p)
        assert _mode(p) == 0o600

    def test_sqlite_memory_store_narrows_existing_db(self, tmp_path):
        from headroom.memory.adapters.sqlite import SQLiteMemoryStore

        p = tmp_path / "headroom_memory.db"
        sqlite3.connect(p).close()  # an older, unhardened run left it 0644
        assert _mode(p) == 0o644
        SQLiteMemoryStore(p)
        assert _mode(p) == 0o600

    def test_fts5_index(self, tmp_path):
        from headroom.memory.adapters.fts5 import FTS5TextIndex

        p = tmp_path / "headroom_memory.db"
        try:
            FTS5TextIndex(p)
        except sqlite3.OperationalError as exc:  # sqlite built without FTS5
            pytest.skip(f"FTS5 unavailable: {exc}")
        assert _mode(p) == 0o600

    def test_graph_store(self, tmp_path):
        from headroom.memory.adapters.sqlite_graph import SQLiteGraphStore

        p = tmp_path / "headroom_graph.db"
        SQLiteGraphStore(p)
        assert _mode(p) == 0o600

    def test_vector_index(self, tmp_path):
        pytest.importorskip("sqlite_vec")
        from headroom.memory.adapters.sqlite_vector import SQLiteVectorIndex

        p = tmp_path / "vectors.db"
        try:
            SQLiteVectorIndex(dimension=8, db_path=p)
        except (ImportError, RuntimeError) as exc:
            pytest.skip(f"sqlite-vec unavailable here: {exc}")
        assert _mode(p) == 0o600

    def test_hnsw_metadata_dump(self, tmp_path):
        # Never ``importorskip("hnswlib")`` here: on runners without AVX the
        # native import dies with SIGILL and takes the whole pytest shard with
        # it. The adapter's probe runs the import in a subprocess.
        import asyncio

        from headroom.memory.adapters.hnsw import _check_hnswlib_available

        if not _check_hnswlib_available():
            pytest.skip("hnswlib not available (or not usable on this CPU)")

        import numpy as np

        from headroom.memory.adapters.hnsw import HNSWVectorIndex
        from headroom.memory.models import Memory

        index = HNSWVectorIndex(dimension=4, max_elements=16)
        mem = Memory(content="secret fact", user_id="alice", embedding=np.ones(4, dtype=np.float32))
        asyncio.run(index.index(mem))
        base = tmp_path / "idx"
        index.save_index(base)
        assert _mode(base.with_suffix(".meta")) == 0o600


def test_native_memory_dir_is_0700(tmp_path):
    from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler

    d = tmp_path / "memories"
    d.mkdir()
    assert _mode(d) == 0o755
    handler = MemoryHandler(MemoryConfig(native_memory_dir=str(d)))
    handler._init_native_memory_dir()  # what initialize() runs for the native tool
    assert _mode(d) == 0o700


def test_license_cache_is_private(tmp_path):
    from headroom.telemetry.reporter import LicenseInfo, UsageReporter

    cache = tmp_path / "license.json"
    cache.write_text("{}")
    assert _mode(cache) == 0o644
    reporter = UsageReporter("hlk_test", cache_path=cache)
    reporter._license_info = LicenseInfo(status="active")
    reporter._save_cache()
    assert _mode(cache) == 0o600
    assert "active" in cache.read_text()


def test_ccr_backend_still_private_via_shared_primitive(tmp_path):
    from headroom.cache.backends.sqlite import SQLiteBackend

    p = tmp_path / "ccr_store.db"
    SQLiteBackend(p)
    assert _mode(p) == 0o600
