#!/usr/bin/env python3
"""Release sweep helper for ``scripts/cut_public_release.sh``.

The token lists it reads live in a dev-only folder that every public stage
leaves out, so neither this script nor the release script carries them.

    python3 scripts/release_scan.py check-tokens DEV_ONLY_DIR
    python3 scripts/release_scan.py deep STAGE DEV_ONLY_DIR

``check-tokens``: every pattern (one per line, an extended regular expression,
matched case-insensitively) must compile, must match its canary line (the
same line of ``<list>.canary``) and must match none of the public example
lines (``public_examples.txt``) — a broken or too-broad pattern refuses the
release instead of silently matching nothing or everything.  No blank lines,
no CR line ends.

``deep``: what a plain ``grep -a`` cannot read is decoded and searched for the
content tokens — members of ``.npz``/``.zip`` files (numpy string arrays are
stored as UTF-32), gzip files, PNG text chunks, UTF-16/UTF-32 text, the text
datasets and attributes of HDF5 files (compressed ones included, through
h5py; an HDF5 file it cannot read is refused), the compressed (FlateDecode)
streams of PDF files; a symbolic link that leaves the stage is refused; and,
for every directory listed in ``private_roots.txt``, any staged file whose
bytes (as they are, or with LF line ends) equal a file there (of 256 bytes or
more) is refused.  Absolute home-directory paths found inside files are
reported (not refused).  Plain text is the release script's ``grep`` step.

Exit status: 0 clean, 1 something found, 2 a broken token list.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import os
import re
import sys
import zipfile
import zlib
from pathlib import Path

LISTS = ("private_name_tokens.txt", "private_content_tokens.txt")
_MIN_PRIVATE_BYTES = 256         # smaller files carry nothing worth fingerprinting
HOME_PATH = re.compile(rb"(/Users/[A-Za-z0-9._-]+/|/home/[A-Za-z0-9._-]+/"
                       rb"|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)")


def _lines(path: Path) -> list[str]:
    data = path.read_bytes()
    if b"\r" in data:
        raise ValueError(f"{path.name}: CR line ends (save it with LF)")
    lines = data.decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if any(not ln.strip() for ln in lines):
        raise ValueError(f"{path.name}: a blank line (grep -f would match everything)")
    return lines


def _patterns(dev: Path, name: str) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in _lines(dev / name)]


def check_tokens(dev: Path) -> int:
    try:
        public = _lines(dev / "public_examples.txt")
        for name in LISTS:
            pats = _lines(dev / name)
            canaries = _lines(dev / (name + ".canary"))
            if len(pats) != len(canaries):
                raise ValueError(f"{name}: {len(pats)} patterns but {len(canaries)} canaries")
            for pat, canary in zip(pats, canaries, strict=True):
                rx = re.compile(pat, re.IGNORECASE)
                if not rx.search(canary):
                    raise ValueError(f"{name}: {pat!r} does not match its canary {canary!r}")
                hit = next((p for p in public if rx.search(p)), None)
                if hit is not None:
                    raise ValueError(f"{name}: {pat!r} matches the public example {hit!r}")
    except (OSError, ValueError, re.error) as exc:
        print(f"broken private token list: {exc}")
        return 2
    print("token lists checked")
    return 0


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------
def _texts_of_bytes(data: bytes) -> list[str]:
    return [data.decode("latin-1")] + [data.decode(enc, errors="ignore") for enc in
                                       ("utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be")]


def _npy_texts(data: bytes) -> list[str]:
    import numpy as np
    try:
        arr = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception:            # noqa: BLE001 - object arrays, odd files: raw text below
        return _texts_of_bytes(data)
    if arr.dtype.kind in "US":
        return [" ".join(str(x) for x in arr.ravel().tolist())]
    return [data.decode("latin-1")]


def _png_texts(data: bytes) -> list[str]:
    out, i = [], 8
    while i + 8 <= len(data):
        n = int.from_bytes(data[i:i + 4], "big")
        kind = data[i + 4:i + 8]
        body = data[i + 8:i + 8 + n]
        try:
            if kind == b"tEXt":
                out.append(body.decode("latin-1"))
            elif kind == b"zTXt":
                key, _, rest = body.partition(b"\0")
                out.append(key.decode("latin-1") + " " + zlib.decompress(rest[1:]).decode("latin-1"))
            elif kind == b"iTXt":
                key, _, rest = body.partition(b"\0")
                compressed, rest = rest[0], rest[2:]
                _lang, _, rest = rest.partition(b"\0")
                _tkey, _, text = rest.partition(b"\0")
                text = zlib.decompress(text) if compressed else text
                out.append(key.decode("latin-1") + " " + text.decode("utf-8", errors="replace"))
        except (zlib.error, IndexError):
            out.append(body.decode("latin-1"))
        i += 12 + n
    return out


class _Unreadable(Exception):
    """A file whose content the scan cannot decode (refused)."""


_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
# a PDF stream's data (not the "stream" ending "endstream")
_PDF_STREAM = re.compile(rb"(?<!end)stream\r?\n(.*?)endstream", re.DOTALL)


def _is_hdf5(data: bytes) -> bool:
    off = 0                      # the superblock: at 0, or after a 512·2^n-byte user block
    while off + 8 <= len(data):
        if data[off:off + 8] == _HDF5_MAGIC:
            return True
        off = 512 if off == 0 else 2 * off
    return False


def _h5_texts(path: Path) -> list[str]:
    """Names, attributes and text datasets of an HDF5 file, through h5py (a
    compressed dataset is invisible to the raw-byte search)."""
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise _Unreadable(f"an HDF5 file, and h5py is missing to read it ({exc})") from exc

    def text(value) -> str:
        return " ".join(x.decode("latin-1") if isinstance(x, bytes) else str(x)
                        for x in np.asarray(value).ravel().tolist())

    out: list[str] = []

    def visit(name, obj) -> None:
        out.append(name)
        out.extend(f"{k} {text(v)}" for k, v in obj.attrs.items())
        if isinstance(obj, h5py.Dataset) and (obj.dtype.kind in "SUO" or obj.dtype.names):
            out.append(text(obj[()]))

    try:
        with h5py.File(path, "r") as f:
            out.extend(f"{k} {text(v)}" for k, v in f.attrs.items())
            f.visititems(visit)
    except Exception as exc:     # noqa: BLE001 - any failure leaves text unread: refuse
        raise _Unreadable(f"an HDF5 file h5py cannot read ({type(exc).__name__}: {exc})") \
            from exc
    return out


def _pdf_texts(data: bytes) -> list[str]:
    """The FlateDecode (zlib) streams of a PDF, decompressed — page text and
    metadata are usually compressed there."""
    out = []
    for m in _PDF_STREAM.finditer(data):
        try:
            out.append(zlib.decompressobj().decompress(m.group(1)).decode("latin-1"))
        except zlib.error:
            continue             # (another filter: an image, a font)
    return out


def _file_texts(path: Path) -> list[str]:
    data = path.read_bytes()
    suffix = path.suffix.lower()
    texts: list[str] = []
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                member = zf.read(info)
                texts.append(info.filename)
                texts += _npy_texts(member) if info.filename.endswith(".npy") else \
                    _texts_of_bytes(member)
    elif data[:2] == b"\x1f\x8b":
        try:
            texts += _texts_of_bytes(gzip.decompress(data))
        except (OSError, EOFError):
            texts += _texts_of_bytes(data)
    elif data[:8] == b"\x89PNG\r\n\x1a\n":
        texts += _png_texts(data)
    elif data[:5] == b"%PDF-":
        texts += _texts_of_bytes(data) + _pdf_texts(data)
    elif _is_hdf5(data):
        texts += _texts_of_bytes(data) + _h5_texts(path)
    elif suffix == ".npy":
        texts += _npy_texts(data)
    elif b"\0" in data[:4096] or data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        texts += _texts_of_bytes(data)            # UTF-16/32 text, other binaries
    return texts


def _hashes(data: bytes) -> set[str]:
    return {hashlib.sha256(data).hexdigest(),
            hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()}


def deep(stage: Path, dev: Path) -> int:
    content = _patterns(dev, "private_content_tokens.txt")
    stage = stage.resolve()
    found: list[str] = []
    notes: list[str] = []
    private: dict[str, str] = {}
    roots_file = dev / "private_roots.txt"
    for root in (_lines(roots_file) if roots_file.exists() else []):
        r = Path(os.path.expanduser(root))
        if not r.is_dir():
            notes.append(f"private root not found, its files are not fingerprinted: {root}")
            continue
        for f in r.rglob("*"):
            if f.is_file() and not f.is_symlink():
                data = f.read_bytes()
                if len(data) < _MIN_PRIVATE_BYTES:      # (empty .gitkeep, a blank __init__)
                    continue
                for h in _hashes(data):
                    private[h] = str(f.relative_to(r))
    for dirpath, dirnames, filenames in os.walk(stage):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in filenames + [d for d in dirnames if (Path(dirpath) / d).is_symlink()]:
            f = Path(dirpath) / name
            rel = f.relative_to(stage)
            if f.is_symlink():
                target = f.resolve()
                if stage not in target.parents and target != stage:
                    found.append(f"{rel}: a link that leaves the stage")
                continue
            data = f.read_bytes()
            hit = next((private[h] for h in _hashes(data) if h in private), None)
            if hit is not None:
                found.append(f"{rel}: a copy of the private file {hit}")
                continue
            try:
                texts = _file_texts(f)
            except _Unreadable as exc:
                found.append(f"{rel}: {exc}")
                continue
            for text in texts:
                rx = next((rx for rx in content if rx.search(text)), None)
                if rx is not None:
                    found.append(f"{rel}: private token {rx.pattern!r} inside")
                    break
            m = HOME_PATH.search(data)
            if m:
                notes.append(f"{rel}: an absolute home path ({m.group(0)[:40].decode('latin-1')})")
    for n in notes:
        print(f"note: {n}")
    if found:
        print("PRIVATE CONTENT FOUND:")
        for x in found:
            print(f"  {x}")
        return 1
    print("deep scan clean")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "check-tokens":
        return check_tokens(Path(argv[1]))
    if len(argv) == 3 and argv[0] == "deep":
        return deep(Path(argv[1]), Path(argv[2]))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
