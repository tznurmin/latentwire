"""Verify declared local artifact bytes with explicit, bounded concurrency."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import os
import sys
import threading
import time

from . import _artifact_io as file_io
from ._artifact_types import (
    FileStatus, FileVerification, VerificationError, VerificationErrorCode,
    VerificationLimits, VerificationReport, valid_absolute_path, validate_limits,
)
from .manifest import load_manifest

__all__ = ["FileStatus", "FileVerification", "VerificationError", "VerificationErrorCode",
           "VerificationLimits", "VerificationReport", "verify_files"]


class _Stopped(Exception):
    def __init__(self, code):
        self.code = code


class _Control:
    def __init__(self, timeout, external):
        self.deadline = time.monotonic() + timeout
        self.external = external
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.active = self.peak = 0

    def check(self):
        if self.external is not None and self.external.is_set():
            raise _Stopped("cancelled")
        if time.monotonic() >= self.deadline:
            raise _Stopped("deadline_exceeded")
        if self.stop.is_set():
            raise _Stopped("cancelled")

    def run(self, root_fd, entry, limits):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            self.check()
            return file_io.verify_one(root_fd, entry, limits, self.check)
        finally:
            with self.lock:
                self.active -= 1


def _parallel(root_fd, entries, limits, control):
    workers = min(limits.workers, len(entries))
    results, pending, next_index, max_pending = [None] * len(entries), {}, 0, 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="artifact-read") as pool:
        try:
            while next_index < len(entries) or pending:
                control.check()
                while next_index < len(entries) and len(pending) < workers:
                    future = pool.submit(control.run, root_fd, entries[next_index], limits)
                    pending[future] = next_index
                    next_index += 1
                max_pending = max(max_pending, len(pending))
                done, _ = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=pending.__getitem__):
                    index = pending.pop(future)
                    results[index] = future.result()
            control.check()
        except BaseException:
            control.stop.set()
            for future in pending:
                future.cancel()
            raise
    return tuple(results), workers, max_pending


def _execute(root, manifest, limits, control, wire_sha):
    control.check()
    if not os.path.isdir("/proc/self/fd"):
        raise _Stopped("unsupported_platform")
    root_fd = None
    unavailable = False
    try:
        root_fd = file_io.open_root(root, control.check)
    except OSError:
        unavailable = True
    if unavailable:
        raise _Stopped("root_unavailable")
    try:
        initial = os.fstat(root_fd)
        results, workers, max_pending = _parallel(root_fd, manifest.files, limits, control)
        if not file_io.root_matches(root, initial, control.check):
            raise _Stopped("root_changed")
        control.check()
        return VerificationReport(wire_sha, results, limits.workers, workers, control.peak, max_pending)
    finally:
        control.stop.set()
        os.close(root_fd)


def verify_files(root: str, manifest_data: bytes, limits: VerificationLimits,
                 *, cancel: threading.Event | None = None) -> VerificationReport:
    """Return complete file observations, or a fixed operation error after cleanup.

    Deadlines are cooperative between I/O calls. Use an outer process deadline
    for filesystems where an individual kernel call may block indefinitely.
    """
    validate_limits(limits)
    if not valid_absolute_path(root) or (cancel is not None and type(cancel) is not threading.Event):
        raise VerificationError("invalid_request")
    manifest = load_manifest(manifest_data)
    if sum(entry.size_bytes for entry in manifest.files) > limits.max_payload_bytes:
        raise VerificationError("budget_exceeded")
    if sys.platform != "linux" or not hasattr(os, "O_PATH"):
        raise VerificationError("unsupported_platform")
    control = _Control(limits.timeout_seconds, cancel)
    failure = None
    try:
        return _execute(root, manifest, limits, control, hashlib.sha256(manifest_data).hexdigest())
    except _Stopped as error:
        failure = error.code
    except (KeyboardInterrupt, MemoryError):
        raise
    except Exception:
        failure = "worker_failed"
    # Raising outside the handler avoids retaining private exception context.
    raise VerificationError(failure)
