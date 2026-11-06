"""Admit the exact retained arrays from bounded immutable file buffers."""
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import math
import os
import stat

from . import _artifact_io as io
from . import _reader_identity as identity
from ._artifact_types import valid_absolute_path
from .artifact_files import VerificationError, VerificationLimits, verify_files
from .manifest import ManifestError, load_manifest
from ._reader_errors import ReaderError


@dataclass(frozen=True, slots=True)
class ReaderInputs:
    reader: object
    codewords: object
    verification: object


def read_array(data, *, shape, raw_sha256, row_sha256=()):
    """The caller authenticates this immutable buffer before format admission."""
    failed = False
    try:
        if (type(data) is not bytes or type(shape) is not tuple or len(shape) != 2
                or any(type(v) is not int or v < 1 for v in shape)
                or math.prod(shape) > 4096 * 128
                or len(data) != 128 + math.prod(shape) * 4):
            raise ReaderError("invalid_inputs")
        import numpy as np
        stream = BytesIO(data)
        if np.lib.format.read_magic(stream) != (1, 0):
            raise ReaderError("invalid_inputs")
        actual, order, dtype = np.lib.format.read_array_header_1_0(
            stream, max_header_size=118)
        if (actual != shape or order is not False or dtype.str != "<f4"
                or dtype.fields is not None or dtype.subdtype is not None
                or stream.tell() != 128):
            raise ReaderError("invalid_inputs")
        stream.seek(0)
        value = np.lib.format.read_array(stream, allow_pickle=False,
                                         max_header_size=118)
        if (not bool(np.isfinite(value).all())
                or sha256(value.tobytes(order="C")).hexdigest() != raw_sha256):
            raise ReaderError("invalid_inputs")
        if row_sha256 and (len(row_sha256) != shape[0] or any(
                sha256(value[i].tobytes(order="C")).hexdigest() != expected
                for i, expected in enumerate(row_sha256))):
            raise ReaderError("invalid_inputs")
        return value
    except ReaderError:
        raise
    except (ImportError, OSError, ValueError, TypeError, EOFError, OverflowError):
        failed = True
    if failed:
        raise ReaderError("invalid_inputs")


def _check():
    pass


def _read(root_fd, name, limit, *, exact=None):
    with ExitStack() as stack:
        path_fd = io.open_relative(root_fd, name, _check)
        stack.callback(os.close, path_fd)
        before = os.fstat(path_fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_size > limit
                or (exact is not None and before.st_size != exact)):
            raise ReaderError("invalid_inputs")
        data_fd = io.open_data(path_fd)
        stack.callback(os.close, data_fd)
        chunks, count = [], 0
        while count <= limit:
            block = os.read(data_fd, min(1048576, limit + 1 - count))
            if not block:
                break
            count += len(block)
            chunks.append(block)
        if (count > limit or count != before.st_size
                or io.signature(before) != io.signature(os.fstat(data_fd))
                or not io.path_matches(root_fd, name, before, _check)):
            raise ReaderError("invalid_inputs")
        return b"".join(chunks)


def _layout(root_fd):
    directory_fd = io.open_data(root_fd)
    try:
        with os.scandir(directory_fd) as entries:
            names = set()
            for number, entry in enumerate(entries):
                if (number >= 3 or not entry.is_file(follow_symlinks=False)
                        or entry.name not in {"manifest.json","reader.npy","codewords.npy"}):
                    raise ReaderError("invalid_inputs")
                names.add(entry.name)
        if names != {"manifest.json","reader.npy","codewords.npy"}:
            raise ReaderError("invalid_inputs")
    finally:
        os.close(directory_fd)


def _load(root):
    if not valid_absolute_path(root):
        raise ReaderError("invalid_inputs")
    specs = identity.ARRAYS
    if (type(identity.MANIFEST_SHA256) is not str or len(identity.MANIFEST_SHA256) != 64
            or len(specs) != 2
            or tuple(s[0] for s in specs) != ("reader.npy","codewords.npy")):
        raise ReaderError("invalid_inputs")
    with ExitStack() as stack:
        root_fd = io.open_root(root, _check)
        stack.callback(os.close, root_fd)
        before = os.fstat(root_fd)
        _layout(root_fd)
        manifest_data = _read(root_fd, "manifest.json", 4096)
        if sha256(manifest_data).hexdigest() != identity.MANIFEST_SHA256:
            raise ReaderError("invalid_inputs")
        manifest = load_manifest(manifest_data)
        expected = sorted((s[0], s[2], s[3]) for s in specs)
        if [(f.path,f.size_bytes,f.sha256) for f in manifest.files] != expected:
            raise ReaderError("invalid_inputs")
        report = verify_files(root, manifest_data,
            VerificationLimits(2, 1048576, sum(s[2] for s in specs), 120))
        if not report.verified:
            raise ReaderError("invalid_inputs", verification=report)
        buffers = []
        for name, shape, size, file_sha, raw_sha, row_shas in specs:
            data = _read(root_fd, name, size, exact=size)
            if sha256(data).hexdigest() != file_sha:
                raise ReaderError("invalid_inputs")
            buffers.append(data)
        _layout(root_fd)
        if not io.root_matches(root, before, _check):
            raise ReaderError("invalid_inputs")
        arrays = [read_array(data, shape=spec[1], raw_sha256=spec[4],
                             row_sha256=spec[5])
                  for data, spec in zip(buffers, specs)]
    return ReaderInputs(arrays[0], arrays[1], report)


def load_inputs(root):
    """Read only the fixed, authenticated three-file input directory."""
    failed = False
    try:
        return _load(root)
    except (ReaderError, ManifestError, VerificationError):
        raise
    except (OSError, TypeError, ValueError, OverflowError):
        failed = True
    if failed:
        raise ReaderError("invalid_inputs")
