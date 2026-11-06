"""Exclusive, verified local Qwen checkpoint leases; no eager native imports."""

import hashlib
import os
import threading

from .artifact_files import VerificationError, verify_files
from .manifest import ManifestError
from ._qwen_checkpoint_data import (
    CHECKPOINT_FILES, CHECKPOINT_MANIFEST, MODEL_REVISION,
    CheckpointError, CheckpointErrorCode, CheckpointIdentity,
)

__all__ = ["CHECKPOINT_MANIFEST", "MODEL_REVISION", "CheckpointError",
           "CheckpointErrorCode", "CheckpointIdentity", "QwenCheckpoint", "open_qwen_checkpoint"]
_LEASE_LOCK = threading.Lock()


def _load_native(root, manifest_sha256):
    from ._qwen_native import load_native

    return load_native(root, manifest_sha256)


def _check_directory(root):
    invalid = False
    names = set()
    allowed = CHECKPOINT_FILES | {"SHA256MANIFEST.txt"}
    try:
        with os.scandir(root) as entries:
            for index, entry in enumerate(entries):
                if index >= 13 or entry.name not in allowed:
                    invalid = True
                    break
                names.add(entry.name)
        invalid = invalid or not CHECKPOINT_FILES <= names
    except OSError:
        invalid = True
    if invalid:
        raise CheckpointError("unexpected_files")


class QwenCheckpoint:
    """Single-use context; borrowed native references must not outlive it."""

    def __init__(self, root, verification_limits):
        self._root = root
        self._limits = verification_limits
        self._used = self._locked = False
        self._native = self._identity = self._verification = None
        self.cleanup_error = None

    @property
    def model(self):
        if self._native is None:
            raise CheckpointError("closed")
        return self._native.model

    @property
    def tokenizer(self):
        if self._native is None:
            raise CheckpointError("closed")
        return self._native.tokenizer

    @property
    def identity(self):
        if self._identity is None:
            raise CheckpointError("closed")
        return self._identity

    @property
    def verification(self):
        return self._verification

    def _release(self):
        failed = False
        try:
            if self._native is not None:
                self._native.close()
        except Exception:
            failed = True
        finally:
            self._native = None
            if self._locked:
                self._locked = False
                _LEASE_LOCK.release()
        if failed:
            self.cleanup_error = CheckpointError("cleanup_failed")

    def __enter__(self):
        if self._used:
            raise CheckpointError("closed")
        self._used = True
        if not _LEASE_LOCK.acquire(blocking=False):
            raise CheckpointError("busy")
        self._locked = True
        failure = None
        try:
            self._verification = verify_files(self._root, CHECKPOINT_MANIFEST, self._limits)
            if not self._verification.verified:
                raise CheckpointError("checkpoint_unverified", verification=self._verification)
            _check_directory(self._root)
            self._native = _load_native(self._root, hashlib.sha256(CHECKPOINT_MANIFEST).hexdigest())
            self._identity = self._native.identity
        except (VerificationError, ManifestError, CheckpointError, KeyboardInterrupt, SystemExit) as error:
            failure = error
        except Exception:
            failure = CheckpointError("load_failed")
        if failure is not None:
            self._release()
            raise failure
        return self

    def __exit__(self, exception_type, exception, traceback):
        self._release()
        if self.cleanup_error is not None and exception_type is None:
            raise self.cleanup_error
        return False


def open_qwen_checkpoint(root, verification_limits):
    """Verify the supported checkpoint, then lend one CUDA-only native model.

    The caller must prevent writes to the checkpoint throughout the lease and
    impose hard process/resource limits around native work. No CPU fallback.
    """
    return QwenCheckpoint(root, verification_limits)
