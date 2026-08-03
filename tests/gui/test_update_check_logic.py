"""Pure update-check logic (services/update_check.py) — no Qt."""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import urllib.error

import pytest

from linac_gen_gui.interphase.services import update_check as uc

git_missing = shutil.which("git") is None


class TestVersions:
    @pytest.mark.parametrize("s,expect", [
        ("v1.4", (1, 4)), ("1.4", (1, 4)), ("v1.4.2", (1, 4, 2)),
        ("2", (2,)), ("", None), ("junk", None), ("v1.4-rc1", None),
    ])
    def test_parse(self, s, expect):
        assert uc.parse_version(s) == expect

    @pytest.mark.parametrize("latest,installed,newer", [
        ("v1.5", "1.4", True), ("v1.4", "1.4", False),
        ("1.10", "1.9", True), ("v1.4", "1.4.1", False),
        ("v1.4.1", "1.4", True), ("v2", "1.99", True),
        ("garbage", "1.4", False), ("v1.5", "unknown", False),
    ])
    def test_is_newer(self, latest, installed, newer):
        assert uc.is_newer(latest, installed) is newer


class TestFetch:
    def _opener_returning(self, payload: bytes):
        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def opener(req, timeout):
            assert timeout == 3.0
            assert req.get_header("User-agent") == "HELIX-update-check"
            return _Resp(payload)
        return opener

    def test_ok(self):
        body = json.dumps({"tag_name": "v1.4",
                           "html_url": "https://x/rel"}).encode()
        got = uc.fetch_latest_release(opener=self._opener_returning(body))
        assert got == ("v1.4", "https://x/rel")

    def test_missing_html_url_falls_back(self):
        body = json.dumps({"tag_name": "v1.4"}).encode()
        got = uc.fetch_latest_release(opener=self._opener_returning(body))
        assert got == ("v1.4", uc.RELEASES_PAGE)

    def test_junk_tag_is_none(self):
        body = json.dumps({"tag_name": "not-a-version"}).encode()
        assert uc.fetch_latest_release(
            opener=self._opener_returning(body)) is None

    @pytest.mark.parametrize("exc", [
        urllib.error.URLError("dns"),
        urllib.error.HTTPError("u", 403, "rate limited", {}, None),
        TimeoutError(),
        ValueError("bad json"),
    ])
    def test_every_failure_is_silent_none(self, exc):
        def opener(req, timeout):
            raise exc
        assert uc.fetch_latest_release(opener=opener) is None


class TestGitProgress:
    @pytest.mark.parametrize("chunk,expect", [
        ("Receiving objects:  42% (10/24)", ("Receiving objects", 42)),
        ("remote: Counting objects: 45%", ("Counting objects", 45)),
        ("Resolving deltas: 100% (5/5), done.",
         ("Resolving deltas", 100)),
        ("Receiving objects: 10%\rReceiving objects: 67%",
         ("Receiving objects", 67)),
        ("From github.com:Accel-Toolkit/HELIX", None),
        ("", None),
    ])
    def test_parse(self, chunk, expect):
        assert uc.parse_git_progress(chunk) == expect


def _run(args, cwd):
    subprocess.run(args, cwd=str(cwd), check=True,
                   capture_output=True, text=True)


@pytest.mark.skipif(git_missing, reason="git not available")
class TestClassifyInstall:
    @pytest.fixture()
    def clone(self, tmp_path):
        origin = tmp_path / "origin.git"
        _run(["git", "init", "--bare", "-b", "main", str(origin)],
             tmp_path)
        work = tmp_path / "clone"
        _run(["git", "clone", str(origin), str(work)], tmp_path)
        _run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
              "commit", "--allow-empty", "-m", "init"], work)
        _run(["git", "push", "origin", "HEAD:main"], work)
        _run(["git", "checkout", "-B", "main", "origin/main"], work)
        return work

    def _with_origin(self, clone, url):
        _run(["git", "remote", "set-url", "origin", url], clone)
        return uc.classify_install(clone)

    @pytest.mark.parametrize("url", [
        "https://github.com/Accel-Toolkit/HELIX",
        "https://github.com/Accel-Toolkit/HELIX.git",
        "git@github.com:Accel-Toolkit/HELIX.git",
        "https://github.com/accel-toolkit/helix.git",
    ])
    def test_public_spellings(self, clone, url):
        assert self._with_origin(clone, url) \
            is uc.InstallState.PUBLIC_CLEAN

    def test_dev_clone_even_when_dirty(self, clone):
        (clone / "junk.txt").write_text("x")
        got = self._with_origin(
            clone, "https://github.com/Abhishek-Pathak-90/HELIX-dev.git")
        assert got is uc.InstallState.DEV

    def test_foreign_remote_is_not_git(self, clone):
        got = self._with_origin(
            clone, "https://github.com/Someone/Else.git")
        assert got is uc.InstallState.NOT_GIT

    def test_dirty_public(self, clone):
        _run(["git", "remote", "set-url", "origin",
              "https://github.com/Accel-Toolkit/HELIX.git"], clone)
        (clone / "edited.py").write_text("x")
        assert uc.classify_install(clone) \
            is uc.InstallState.PUBLIC_DIRTY

    def test_non_main_branch_is_dirty(self, clone):
        _run(["git", "remote", "set-url", "origin",
              "https://github.com/Accel-Toolkit/HELIX.git"], clone)
        _run(["git", "checkout", "-b", "side"], clone)
        assert uc.classify_install(clone) \
            is uc.InstallState.PUBLIC_DIRTY

    def test_plain_directory(self, tmp_path):
        d = tmp_path / "plain"
        d.mkdir()
        assert uc.classify_install(d) is uc.InstallState.NOT_GIT
