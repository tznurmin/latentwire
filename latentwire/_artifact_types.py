"""Immutable file-verification values and fixed public failures."""

from dataclasses import dataclass
from enum import Enum
import math


class VerificationErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    BUDGET_EXCEEDED = "budget_exceeded"
    UNSUPPORTED_PLATFORM = "unsupported_platform"
    ROOT_UNAVAILABLE = "root_unavailable"
    ROOT_CHANGED = "root_changed"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    CANCELLED = "cancelled"
    WORKER_FAILED = "worker_failed"
    MANIFEST_UNAVAILABLE = "manifest_unavailable"


class VerificationError(ValueError):
    """A public failure containing no local path or upstream diagnostic."""

    def __init__(self, code: VerificationErrorCode | str):
        self.code = VerificationErrorCode(code)
        super().__init__(f"Artifact verification failed: {self.code.value}.")


@dataclass(frozen=True, slots=True)
class VerificationLimits:
    workers: int
    chunk_bytes: int
    max_payload_bytes: int
    timeout_seconds: float

    def __post_init__(self):
        validate_limits(self)


def validate_limits(value):
    valid = type(value) is VerificationLimits
    for name, low, high in (("workers", 1, 256), ("chunk_bytes", 4096, 8388608),
                            ("max_payload_bytes", 0, 2**42)):
        number = getattr(value, name, None) if valid else None
        valid = valid and type(number) is int and low <= number <= high
    timeout = getattr(value, "timeout_seconds", None) if valid else None
    if not (valid and type(timeout) in (int, float) and 0 < timeout <= 86400
            and math.isfinite(timeout)):
        raise VerificationError(VerificationErrorCode.INVALID_REQUEST)


def valid_absolute_path(path):
    if type(path) is not str or not 1 <= len(path) <= 4096 or "\0" in path:
        return False
    if not path.startswith("/") or (path != "/" and any(
            part in ("", ".", "..") for part in path[1:].split("/"))):
        return False
    try:
        return len(path.encode("utf-8")) <= 4096
    except UnicodeError:
        return False


class FileStatus(str, Enum):
    VERIFIED = "verified"
    MISSING = "missing"
    UNSAFE_PATH = "unsafe_path"
    UNREADABLE = "unreadable"
    SIZE_MISMATCH = "size_mismatch"
    DIGEST_MISMATCH = "digest_mismatch"
    CHANGED = "changed"
    IO_ERROR = "io_error"


@dataclass(frozen=True, slots=True)
class FileVerification:
    path: str
    status: FileStatus
    observed_size_bytes: int | None
    bytes_read: int
    sha256: str | None


@dataclass(frozen=True, slots=True)
class VerificationReport:
    manifest_sha256: str
    files: tuple[FileVerification, ...]
    requested_workers: int
    resolved_workers: int
    peak_workers: int
    max_pending: int

    @property
    def verified(self):
        return all(file.status is FileStatus.VERIFIED for file in self.files)
