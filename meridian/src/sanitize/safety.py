"""Input validation (A10) and resource hygiene (A11).

* detect_type()      content sniffing - never trusts the file extension
* validate_input()   size / archive-expansion / page / slide / image / embedded-object limits
* Deadline           wall-clock budget checked between work units
* WorkDir            tracked temp directory with Windows-tolerant cleanup
"""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import stat
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .policy import Limits

MIME = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "png": "image/png", "jpeg": "image/jpeg", "tiff": "image/tiff", "bmp": "image/bmp",
    "gif": "image/gif", "webp": "image/webp",
}
EXTENSIONS = {
    "pdf": {".pdf"}, "docx": {".docx"}, "pptx": {".pptx"}, "xlsx": {".xlsx"},
    "png": {".png"}, "jpeg": {".jpg", ".jpeg"}, "tiff": {".tif", ".tiff"}, "bmp": {".bmp"},
    "gif": {".gif"}, "webp": {".webp"},
}
IMAGE_KINDS = {"png", "jpeg", "tiff", "bmp", "gif", "webp"}
SUPPORTED = {"pdf", "docx", "pptx"} | IMAGE_KINDS

OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


class InputRejected(Exception):
    """The input cannot be processed. ``code`` is a stable machine-readable reason."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class DeadlineExceeded(RuntimeError):
    pass


class Deadline:
    def __init__(self, seconds: float, clock=time.monotonic):
        self.clock = clock
        self.end = clock() + seconds
        self.seconds = seconds

    def remaining(self) -> float:
        return max(0.0, self.end - self.clock())

    def check(self, what: str = "") -> None:
        if self.clock() > self.end:
            raise DeadlineExceeded(f"processing duration limit ({self.seconds:.0f}s) exceeded {what}".strip())


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------- content sniffing
def detect_type(path: str | os.PathLike) -> tuple[str | None, str]:
    """-> (kind, detail). kind in MIME keys, or None when unknown / unsupported. Never uses the extension."""
    with open(path, "rb") as fh:
        head = fh.read(4096)
    if not head:
        return None, "empty file"
    if head.startswith(b"%PDF-"):
        return "pdf", "pdf header"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png", "png signature"
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg", "jpeg signature"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff", "tiff signature"
    if head[:2] == b"BM":
        return "bmp", "bmp signature"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif", "gif signature"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp", "webp signature"
    if head.startswith(OLE_MAGIC):
        return None, "ole-compound (legacy Office or password-encrypted OOXML)"
    if head[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(path) as z:
                names = set(z.namelist())
        except zipfile.BadZipFile:
            return None, "corrupt zip container"
        if "word/document.xml" in names:
            return "docx", "ooxml word"
        if "ppt/presentation.xml" in names:
            return "pptx", "ooxml presentation"
        if "xl/workbook.xml" in names:
            return "xlsx", "ooxml workbook"
        return None, "zip container of unknown type"
    return None, "unrecognised content"


def validate_input(path: str | os.PathLike, limits: Limits, *, allow_extension_mismatch: bool = False) -> dict:
    """Validate content and limits. Returns {'kind', 'mime', 'detail', 'bytes', 'archive': {...}}."""
    p = Path(path)
    if not p.is_file():
        raise InputRejected("not_found", f"input file not found: {p.name}")
    size = p.stat().st_size
    if size == 0:
        raise InputRejected("empty", "input file is empty")
    if size > limits.max_file_bytes:
        raise InputRejected("file_too_large", f"{size} bytes exceeds limit {limits.max_file_bytes}")
    kind, detail = detect_type(p)
    if kind is None:
        if "ole-compound" in detail:
            raise InputRejected("encrypted_or_legacy", f"{detail}: encrypted or legacy Office files are not supported")
        if "corrupt" in detail:
            raise InputRejected("malformed", detail)
        raise InputRejected("unsupported_type", detail)
    if kind == "xlsx":
        raise InputRejected("unsupported_type", "spreadsheets are not supported by this engine")
    if kind not in SUPPORTED:
        raise InputRejected("unsupported_type", f"{kind} is not supported")
    ext = p.suffix.lower()
    if ext not in EXTENSIONS[kind] and not allow_extension_mismatch:
        raise InputRejected("extension_mismatch", f"content is {kind} but the extension is '{ext or '(none)'}'")
    info = {"kind": kind, "mime": MIME[kind], "detail": detail, "bytes": size,
            "extension_mismatch": ext not in EXTENSIONS[kind], "archive": {}}
    if kind in ("docx", "pptx"):
        info["archive"] = _check_archive(p, limits)
    return info


def _check_archive(path: Path, limits: Limits) -> dict:
    try:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            if len(infos) > limits.max_archive_entries:
                raise InputRejected("archive_too_many_entries", f"{len(infos)} entries > {limits.max_archive_entries}")
            total, worst = 0, 0.0
            embedded = 0
            for zi in infos:
                name = zi.filename
                if name.startswith(("/", "\\")) or ".." in Path(name).parts or ":" in name.split("/")[0]:
                    raise InputRejected("archive_unsafe_path", f"unsafe member path: {name[:60]}")
                if zi.flag_bits & 0x1:
                    raise InputRejected("encrypted", "encrypted archive member")
                total += zi.file_size
                ratio = zi.file_size / max(1, zi.compress_size)
                worst = max(worst, ratio)
                if ratio > limits.max_archive_ratio and zi.file_size > 1_000_000:
                    raise InputRejected("archive_expansion", f"{name[:60]} expands {ratio:.0f}x")
                if "/embeddings/" in name or name.endswith(".bin") and "vbaProject" in name:
                    embedded += 1
            if total > limits.max_archive_uncompressed_bytes:
                raise InputRejected("archive_expansion", f"uncompressed size {total} > {limits.max_archive_uncompressed_bytes}")
            if embedded > limits.max_embedded_objects:
                raise InputRejected("too_many_embedded_objects", f"{embedded} > {limits.max_embedded_objects}")
            bad = z.testzip()
            if bad:
                raise InputRejected("malformed", f"corrupt archive member {bad}")
    except zipfile.BadZipFile as exc:
        raise InputRejected("malformed", f"corrupt archive: {exc}") from exc
    return {"entries": len(infos), "uncompressed_bytes": total, "max_ratio": round(worst, 1), "embedded_objects": embedded}


# --------------------------------------------------------------------------- temp-file hygiene
def _rmtree_retry(path: str, attempts: int = 6, delay: float = 0.1) -> list[str]:
    """Remove a tree, retrying for Windows file-lock / read-only quirks. Returns leftovers."""
    def onerror(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass
    for i in range(attempts):
        shutil.rmtree(path, onerror=onerror)
        if not os.path.exists(path):
            return []
        time.sleep(delay * (i + 1))
    left = []
    for root, _dirs, files in os.walk(path):
        left.extend(os.path.join(root, f) for f in files)
    return left or [path]


@dataclass
class WorkDir:
    """Private scratch directory. Files written here are removed (with retries) on exit."""
    parent: str | None = None
    path: str = ""
    created: list[str] = field(default_factory=list)
    leftover: list[str] = field(default_factory=list)

    def __enter__(self) -> "WorkDir":
        self.path = tempfile.mkdtemp(prefix="meridian-san-", dir=self.parent)
        return self

    def new_path(self, suffix: str = "") -> str:
        p = os.path.join(self.path, uuid.uuid4().hex + suffix)
        self.created.append(p)
        return p

    def __exit__(self, *exc) -> None:
        self.leftover = _rmtree_retry(self.path) if self.path else []

    def report(self) -> dict:
        return {"temp_files_created": len(self.created), "leftover": list(self.leftover),
                "clean": not self.leftover}


def atomic_publish(src: str, dst: str) -> None:
    """Move a finished artifact into place (os.replace is atomic and works on Windows)."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(0.1 * (i + 1))
    shutil.copyfile(src, dst)
    os.remove(src)
