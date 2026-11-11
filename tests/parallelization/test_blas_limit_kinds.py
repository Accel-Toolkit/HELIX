"""BLAS on one thread while runs last (linac_gen.parallel.threads), for each
way a library's thread limit can act: per thread (threadpoolctl 3.7 and
later drive MKL and OpenBLAS built on OpenMP so), one pool for the whole
process, and one pool that only one thread at a time may set (how
threadpoolctl 3.8 drives OpenBLAS built with pthreads, the pip wheels'
build: the first thread that sets it, until it lets go or ends; the others
are ignored).  The libraries are simulated, so every kind runs on every
platform, whatever BLAS numpy has."""
from __future__ import annotations

import logging
import threading
import time
import types

import pytest

from tests.helpers import kept_thread_setting


class _ProcessLib:
    """One count for the whole process; any thread sets it."""

    def __init__(self, n, path="fake/libprocess.so"):
        self.filepath = path
        self.count = n
        self.sets = 0

    def get_num_threads(self):
        return self.count

    def set_num_threads(self, n):
        self.sets += 1
        self.count = n


class _DeafLib(_ProcessLib):
    """Takes no limit."""

    def set_num_threads(self, n):
        self.sets += 1


class _ThreadLib:
    """A count per thread: a thread that never set it sees the default."""

    def __init__(self, default, path="fake/libthread.so"):
        self.filepath = path
        self.default = default
        self._local = threading.local()

    def get_num_threads(self):
        return getattr(self._local, "n", self.default)

    def set_num_threads(self, n):
        self._local.n = n


class _OwnedLib:
    """One count for the process that one thread at a time may set: the
    first that sets it, while it lives and until it lets go."""

    owner = None                    # shared by every instance, as in 3.8
    _lock = threading.Lock()

    def __init__(self, n, path="fake/libowned.so"):
        self.filepath = path
        self.count = n
        self.ignored = 0

    @classmethod
    def clear_controlling_thread(cls, force=False):
        with cls._lock:
            if force or cls.owner is threading.current_thread():
                cls.owner = None

    def get_num_threads(self):
        return self.count

    def set_num_threads(self, n):
        me = threading.current_thread()
        with self._lock:
            if _OwnedLib.owner is None or not _OwnedLib.owner.is_alive():
                _OwnedLib.owner = me
            mine = _OwnedLib.owner is me
        if mine:
            self.count = n
        else:
            self.ignored += 1


class _Controller:
    def __init__(self, libs):
        self.lib_controllers = list(libs)

    def select(self, user_api=None):
        assert user_api == "blas"
        return self

    def limit(self, limits=None):
        """threadpoolctl's limiter, which HELIX before 1.24.1 used: the
        counts as the calling thread sees them, set back on restore."""
        return _Limiter(self.lib_controllers, limits)


class _Limiter:
    def __init__(self, libs, n):
        self._before = [(lib, lib.get_num_threads()) for lib in libs]
        for lib in libs:
            lib.set_num_threads(n)

    def restore_original_limits(self):
        for lib, n in self._before:
            lib.set_num_threads(n)


@pytest.fixture
def blas(monkeypatch):
    """HELIX's BLAS limit pointed at simulated libraries (``blas(*libs)``
    installs them), its bookkeeping fresh, the thread setting kept."""
    with kept_thread_setting() as T:
        libs = []
        monkeypatch.setattr(T, "_pool_controller", lambda: _Controller(libs))
        monkeypatch.setattr(T, "_blas_scope", {}, raising=False)
        monkeypatch.setattr(T, "_blas_shared", [], raising=False)
        monkeypatch.setattr(T, "_blas_depth", 0)
        monkeypatch.setattr(T, "LIMIT_BLAS_IN_RUNS", True)
        _OwnedLib.owner = None

        def use(*new):
            libs[:] = new
            return new
        use.T = T
        yield use
        _OwnedLib.owner = None


def _counts(libs):
    return [lib.get_num_threads() for lib in libs]


def _in_thread(fn):
    """``fn()`` in a new thread; its value (its exception re-raised)."""
    out = {}

    def body():
        try:
            out["value"] = fn()
        except BaseException as exc:               # noqa: BLE001
            out["error"] = exc
    t = threading.Thread(target=body)
    t.start()
    t.join(30)
    assert not t.is_alive()
    if "error" in out:
        raise out["error"]
    return out.get("value")


def test_one_run_holds_each_kind_and_restores_it(blas):
    T = blas.T
    libs = blas(_ProcessLib(8), _ThreadLib(8), _OwnedLib(8))
    with T._blas_one_thread() as held:
        assert held == 1
        assert _counts(libs) == [1, 1, 1]
    assert _counts(libs) == [8, 8, 8]
    assert T._blas_scope == {"fake/libprocess.so": "process",
                             "fake/libthread.so": "thread",
                             "fake/libowned.so": "process"}
    assert T._blas_depth == 0 and T._blas_shared == []


@pytest.mark.parametrize("first_ends_first", [True, False])
def test_overlapping_runs_in_two_threads(blas, first_ends_first):
    """Each run holds every kind in its own thread, whichever ends first; a
    per-thread count comes back in its own thread when its run ends, a pool
    the process shares when the last run ends."""
    T = blas.T
    p, t, o = libs = blas(_ProcessLib(8), _ThreadLib(8), _OwnedLib(8))
    inside = {k: threading.Event() for k in "ab"}
    release = {k: threading.Event() for k in "ab"}
    ended = {k: threading.Event() for k in "ab"}
    done = threading.Event()
    seen = {}

    def run(k):
        def body():
            if k == "b":
                inside["a"].wait(30)
            with T._blas_one_thread() as held:
                seen[k + "_held"] = held
                inside[k].set()
                release[k].wait(30)
                seen[k + "_inside"] = _counts(libs)
            seen[k + "_after"] = t.get_num_threads()   # its own thread's
            ended[k].set()
            done.wait(30)                   # lives on until the test ends
        return threading.Thread(target=body)
    ta, tb = run("a"), run("b")
    ta.start()
    tb.start()
    try:
        assert inside["a"].wait(30) and inside["b"].wait(30)
        first, last = ("a", "b") if first_ends_first else ("b", "a")
        release[first].set()
        assert ended[first].wait(30)
        assert seen[first + "_after"] == 8          # back in its own thread
        assert (p.count, o.count) == (1, 1)         # the other run holds them
        release[last].set()
        assert ended[last].wait(30)
    finally:
        done.set()
        for k in "ab":
            release[k].set()
        ta.join(30)
        tb.join(30)
    for k in "ab":
        assert seen[k + "_held"] == 1
        assert seen[k + "_inside"] == [1, 1, 1]
        assert seen[k + "_after"] == 8
    assert _counts(libs) == [8, 8, 8] and o.ignored == 0
    assert T._blas_depth == 0 and T._blas_shared == []


def test_a_run_after_another_threads_run_holds_the_pool(blas):
    """A thread that ran a run lives on (a server or GUI thread): a run in
    another thread still holds the pool one thread may set."""
    T = blas.T
    (o,) = blas(_OwnedLib(8))
    ran, release = threading.Event(), threading.Event()

    def x():
        with T._blas_one_thread():
            pass
        ran.set()
        release.wait(30)
    tx = threading.Thread(target=x)
    tx.start()
    try:
        assert ran.wait(30)
        assert _OwnedLib.owner is None              # the run let go of it

        def y():
            with T._blas_one_thread() as held:
                return held, o.get_num_threads()
        assert _in_thread(y) == (1, 1)
    finally:
        release.set()
        tx.join(30)
    assert o.get_num_threads() == 8 and o.ignored == 0


def test_a_pool_another_thread_holds(blas):
    """This thread holds the pool (as ``threadpool_limits`` in a script's
    main thread does): a run elsewhere still limits it and restores it, and
    the holder can set it afterwards."""
    T = blas.T
    (o,) = blas(_OwnedLib(8))
    o.set_num_threads(2)
    assert _OwnedLib.owner is threading.current_thread()

    def run():
        with T._blas_one_thread() as held:
            return held, o.get_num_threads()
    assert _in_thread(run) == (1, 1)
    assert o.get_num_threads() == 2
    o.set_num_threads(5)
    assert o.get_num_threads() == 5


def test_helix_never_keeps_the_pool(blas):
    T = blas.T
    blas(_OwnedLib(8))
    with T._blas_one_thread():
        assert _OwnedLib.owner is None
    assert _OwnedLib.owner is None


def test_nested_runs_in_one_thread(blas, monkeypatch):
    """The inner run takes the outer one's limit and lifts nothing."""
    T = blas.T
    libs = blas(_ProcessLib(8), _ThreadLib(8))
    limits = []
    real = T._limit_blas

    def counted():
        limits.append(1)
        return real()
    monkeypatch.setattr(T, "_limit_blas", counted)
    with T._blas_one_thread() as outer:
        with T._blas_one_thread() as inner:
            assert inner == outer == 1
        assert _counts(libs) == [1, 1]
        assert T._blas_depth == 1 and len(limits) == 1
    assert _counts(libs) == [8, 8]
    assert T._blas_depth == 0 and T._blas_shared == []


def test_a_pool_raised_during_a_run_gets_its_first_count_back(blas):
    """Someone raises a held pool while a run lasts; a second run holds it
    again; when both end it has its count from before any run."""
    T = blas.T
    (p,) = blas(_ProcessLib(8))
    ran, release = threading.Event(), threading.Event()

    def first():
        with T._blas_one_thread():
            ran.set()
            release.wait(30)
    t = threading.Thread(target=first)
    t.start()
    try:
        assert ran.wait(30) and p.get_num_threads() == 1
        p.set_num_threads(5)                        # not a HELIX run
        with T._blas_one_thread() as held:
            assert held == 1 and p.get_num_threads() == 1
    finally:
        release.set()
        t.join(30)
    assert p.get_num_threads() == 8


def test_restored_after_a_failing_run(blas):
    T = blas.T
    libs = blas(_ProcessLib(8), _ThreadLib(8), _OwnedLib(8))
    with pytest.raises(RuntimeError, match="a failing run"), \
            T._blas_one_thread():
        raise RuntimeError("a failing run")
    assert _counts(libs) == [8, 8, 8]
    with T._blas_one_thread() as held:              # and the next run holds
        assert held == 1 and _counts(libs) == [1, 1, 1]


def test_each_library_is_classified_once(blas, monkeypatch):
    """Two reads from a new thread (before and after) on a library's first
    limit; none afterwards."""
    T = blas.T
    blas(_ProcessLib(8), _ThreadLib(8))
    probes = []
    real = T._count_elsewhere

    def counted(lib):
        probes.append(lib.filepath)
        return real(lib)
    monkeypatch.setattr(T, "_count_elsewhere", counted)
    for _ in range(3):
        with T._blas_one_thread():
            pass
    assert sorted(probes) == ["fake/libprocess.so"] * 2 + \
        ["fake/libthread.so"] * 2


def test_a_per_thread_library_raised_in_this_thread_only(blas):
    """Other threads see one thread, this one four: still per thread, and
    this thread gets its four back."""
    T = blas.T
    (t,) = blas(_ThreadLib(1))
    t.set_num_threads(4)
    with T._blas_one_thread() as held:
        assert held == 1 and t.get_num_threads() == 1
    assert t.get_num_threads() == 4
    assert T._blas_scope["fake/libthread.so"] == "thread"


def test_an_unread_library_is_shared_and_not_classified(blas, monkeypatch):
    T = blas.T
    (p,) = blas(_ProcessLib(8))
    monkeypatch.setattr(T, "_count_elsewhere", lambda lib: None)
    with T._blas_one_thread() as held:
        assert held == 1 and [c[2] for c in T._blas_shared] == ["process"]
    assert T._blas_scope == {} and p.get_num_threads() == 8


def test_a_library_already_on_one_thread_is_left_alone(blas):
    T = blas.T
    (p,) = blas(_ProcessLib(1))
    with T._blas_one_thread() as held:
        assert held == 1
    assert p.sets == 0 and T._blas_scope == {}


def test_a_library_that_takes_no_limit(blas, monkeypatch, caplog):
    """The run records None (not 1), warns once, and still holds the
    others."""
    T = blas.T
    d, p = blas(_DeafLib(8, path="fake/libdeaf.so"), _ProcessLib(8))
    monkeypatch.setattr(T, "_warned", set())
    with caplog.at_level(logging.WARNING, logger=T.__name__):
        with T._blas_one_thread() as held:
            assert held is None
            assert p.get_num_threads() == 1
        with T._blas_one_thread():
            pass
    assert _counts((d, p)) == [8, 8]
    assert caplog.text.count("could not hold fake/libdeaf.so") == 1


class _ClampLib(_ProcessLib):
    """Keeps at least two threads."""

    def set_num_threads(self, n):
        self.sets += 1
        self.count = max(n, 2)


def test_a_library_that_keeps_two_threads_gets_its_count_back(blas):
    """A limit that does not reach one thread is undone at once, and the
    run records None."""
    T = blas.T
    (c,) = blas(_ClampLib(8, path="fake/libclamp.so"))
    with T._blas_one_thread() as held:
        assert held is None and c.get_num_threads() == 8
    assert c.get_num_threads() == 8


def test_switched_off(blas, monkeypatch):
    T = blas.T
    (p,) = blas(_ProcessLib(8))
    monkeypatch.setattr(T, "LIMIT_BLAS_IN_RUNS", False)
    with T._blas_one_thread() as held:
        assert held is None and p.get_num_threads() == 8
    assert p.sets == 0


def test_without_threadpoolctl(blas, monkeypatch):
    T = blas.T
    monkeypatch.setattr(T, "_pool_controller", lambda: None)
    with T._blas_one_thread() as held:
        assert held is None


def test_no_blas_library_seen(blas):
    """threadpoolctl sees no BLAS library (numpy on Apple's Accelerate):
    nothing is held, so the run does not record 1."""
    T = blas.T
    blas()
    with T._blas_one_thread() as held:
        assert held is None


def test_run_scope_records_what_its_thread_holds(blas):
    """Through run_scope, the entry every run takes: ``blas`` is 1 only when
    BLAS runs one thread in the run's thread."""
    T = blas.T
    (p,) = blas(_ProcessLib(8))
    with T.run_scope() as info:
        assert info["blas"] == 1 and p.get_num_threads() == 1
    blas(_DeafLib(8, path="fake/libdeaf.so"))
    with T.run_scope() as info:
        assert info["blas"] is None


def test_a_nested_run_records_what_the_outer_one_holds(blas):
    """A matcher's run nests the envelope's: when the outer run could not
    hold BLAS, the inner one does not record 1 either."""
    T = blas.T
    blas(_DeafLib(8, path="fake/libdeaf.so"))
    with T._blas_one_thread() as outer, T._blas_one_thread() as inner:
        assert outer is None and inner is None
    blas()
    with T.run_scope() as outer, T.run_scope() as inner:
        assert outer["blas"] is None and inner["blas"] is None


class _UnreadLib(_ProcessLib):
    """Its count cannot be read."""

    def get_num_threads(self):
        return None


def test_an_unreadable_library_is_not_held(blas, monkeypatch, caplog):
    T = blas.T
    u, p = blas(_UnreadLib(8, path="fake/libunread.so"), _ProcessLib(8))
    monkeypatch.setattr(T, "_warned", set())
    with caplog.at_level(logging.WARNING, logger=T.__name__), \
            T._blas_one_thread() as held:
        assert held is None and p.get_num_threads() == 1
    assert u.sets == 0 and p.get_num_threads() == 8
    assert "could not read the thread count of fake/libunread.so" in caplog.text


class _FailingLib(_ProcessLib):
    """Raises on a read (``fail_read``), on a read once it has been set
    (``fail_read_after_set``), on a set (``fail_set``) or on a set that
    took effect (``set_then_fail``)."""

    def __init__(self, n, how, path="fake/libfailing.so"):
        super().__init__(n, path)
        self.how = how

    def get_num_threads(self):
        if self.how == "fail_read" or (self.how == "fail_read_after_set"
                                       and self.sets):
            raise OSError("read failed")
        return self.count

    def set_num_threads(self, n):
        if self.how == "fail_set":
            self.sets += 1
            raise OSError("set failed")
        super().set_num_threads(n)
        if self.how == "set_then_fail" and self.sets == 1:
            raise OSError("set failed after taking effect")


@pytest.mark.parametrize("how", ["fail_read", "fail_read_after_set",
                                 "fail_set", "set_then_fail"])
def test_a_library_that_raises(blas, how):
    """The run records None, still holds the others, and the failing
    library has its count back when the run ends."""
    T = blas.T
    f, p = blas(_FailingLib(8, how), _ProcessLib(8))
    with T._blas_one_thread() as held:
        assert held is None and p.get_num_threads() == 1
    assert (f.count, p.get_num_threads()) == (8, 8)
    assert T._blas_shared == [] and T._blas_depth == 0


class _NoThread:
    """A thread that cannot start (as at interpreter shutdown)."""

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        raise RuntimeError("can't create new thread at interpreter shutdown")


def test_a_probe_that_cannot_start(blas, monkeypatch):
    """Without a thread to read from, the library is held as a pool the
    process shares, put back when the run ends, and classified later."""
    T = blas.T
    (t,) = blas(_ThreadLib(8))
    monkeypatch.setattr(T, "threading", types.SimpleNamespace(Thread=_NoThread))
    with T._blas_one_thread() as held:
        assert held == 1 and t.get_num_threads() == 1
        assert [c[2] for c in T._blas_shared] == ["process"]
    assert t.get_num_threads() == 8 and T._blas_scope == {}


class _SlowLib(_ProcessLib):
    """Records how many threads are inside its read at once."""

    def __init__(self, n, path="fake/libslow.so"):
        super().__init__(n, path)
        self._lock = threading.Lock()
        self.inside = self.most = 0

    def get_num_threads(self):
        with self._lock:
            self.inside += 1
            self.most = max(self.most, self.inside)
        time.sleep(0.02)
        with self._lock:
            self.inside -= 1
        return self.count


def test_two_runs_entering_together(blas):
    """Two runs start at the same moment (two assistant jobs): one at a time
    limits BLAS, both hold it, and it comes back only when both have
    ended."""
    T = blas.T
    (s,) = blas(_SlowLib(8))
    start = threading.Barrier(2)
    inside = threading.Barrier(3)
    release = threading.Event()
    seen = []

    def run():
        start.wait(30)
        with T._blas_one_thread() as held:
            seen.append(held)
            inside.wait(30)
            release.wait(30)
    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    try:
        inside.wait(30)
        assert T._blas_depth == 2 and s.count == 1
    finally:
        release.set()
        for t in threads:
            t.join(30)
    assert seen == [1, 1] and s.most == 1
    assert s.count == 8 and T._blas_depth == 0


def test_runs_in_one_thread_that_end_out_of_order(blas):
    """Two runs open in one thread (held by generators), the first ends
    first: the limit stays until the second ends."""
    T = blas.T
    libs = blas(_ProcessLib(8), _ThreadLib(8))
    first, second = T._blas_one_thread(), T._blas_one_thread()
    assert first.__enter__() == 1 and second.__enter__() == 1
    first.__exit__(None, None, None)
    assert _counts(libs) == [1, 1]
    second.__exit__(None, None, None)
    assert _counts(libs) == [8, 8] and T._blas_depth == 0


def test_a_run_ended_from_another_thread(blas):
    """A run opened in one thread and ended from another (a generator
    closed elsewhere): the opening thread's next run limits BLAS again."""
    T = blas.T
    (p,) = blas(_ProcessLib(8))
    box = {}
    _in_thread(lambda: box.setdefault("run", T._blas_one_thread()).__enter__())
    assert p.get_num_threads() == 1
    box["run"].__exit__(None, None, None)           # ended here
    assert p.get_num_threads() == 8 and T._blas_depth == 0

    def next_run():
        with T._blas_one_thread() as held:
            return held, p.get_num_threads()
    assert _in_thread(next_run) == (1, 1)


class _InterruptedLib(_ProcessLib):
    """A Ctrl-C arrives while its count is read, after another library was
    held."""

    def get_num_threads(self):
        raise KeyboardInterrupt


def test_ctrl_c_while_limiting(blas):
    T = blas.T
    p, k = blas(_ProcessLib(8), _InterruptedLib(8, path="fake/libctrlc.so"))
    with pytest.raises(KeyboardInterrupt), T._blas_one_thread():
        pass
    assert p.get_num_threads() == 8
    assert T._blas_depth == 0 and T._blas_shared == []
    blas(p)
    with T._blas_one_thread() as held:              # the thread runs again
        assert held == 1 and p.get_num_threads() == 1
    assert p.get_num_threads() == 8
