"""Linux descriptor traversal and bounded ordinary-file byte observations."""

from contextlib import ExitStack
import errno
import hashlib
import os
import stat

from ._artifact_types import FileStatus, FileVerification


def _walk(fd, components, directory_last, check):
    """Consume an owned descriptor and return another owned descriptor."""
    try:
        for index, component in enumerate(components):
            check()
            flags = os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC
            if directory_last or index < len(components) - 1:
                flags |= os.O_DIRECTORY
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def open_root(path, check):
    check()
    fd = os.open("/", os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    return _walk(fd, path[1:].split("/") if path != "/" else [], True, check)


def open_relative(root_fd, path, check):
    check()
    return _walk(os.dup(root_fd), path.split("/"), False, check)


def open_data(path_fd):
    # The held O_PATH descriptor pins the ordinary inode checked by the caller.
    return os.open(f"/proc/self/fd/{path_fd}", os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)


def read_chunk(fd, size):
    return os.read(fd, size)


def signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def identity(info):
    return info.st_dev, info.st_ino


def path_matches(root_fd, path, before, check):
    try:
        fd = open_relative(root_fd, path, check)
        try:
            current = os.fstat(fd)
            return stat.S_ISREG(current.st_mode) and signature(before) == signature(current)
        finally:
            os.close(fd)
    except OSError:
        return False


def root_matches(path, before, check):
    try:
        fd = open_root(path, check)
        try:
            return identity(before) == identity(os.fstat(fd))
        finally:
            os.close(fd)
    except OSError:
        return False


def verify_one(root_fd, entry, limits, check):
    observed_size, count = None, 0

    def result(status, digest=None):
        return FileVerification(entry.path, status, observed_size, count, digest)

    try:
        with ExitStack() as owned:
            path_fd = open_relative(root_fd, entry.path, check)
            owned.callback(os.close, path_fd)
            before = os.fstat(path_fd)
            if not stat.S_ISREG(before.st_mode):
                return result(FileStatus.UNSAFE_PATH)
            observed_size = before.st_size
            if observed_size != entry.size_bytes:
                return result(FileStatus.SIZE_MISMATCH)
            check()
            data_fd = open_data(path_fd)
            owned.callback(os.close, data_fd)
            if signature(before) != signature(os.fstat(data_fd)):
                return result(FileStatus.CHANGED)
            digest = hashlib.sha256()
            while count <= entry.size_bytes:
                check()
                chunk = read_chunk(data_fd, min(limits.chunk_bytes, entry.size_bytes + 1 - count))
                count += len(chunk)
                check()
                if not chunk or count > entry.size_bytes:
                    break
                digest.update(chunk)
            check()
            if (count != entry.size_bytes or signature(before) != signature(os.fstat(data_fd))
                    or not path_matches(root_fd, entry.path, before, check)):
                return result(FileStatus.CHANGED)
            actual = digest.hexdigest()
            status = FileStatus.VERIFIED if actual == entry.sha256 else FileStatus.DIGEST_MISMATCH
            return result(status, actual)
    except OSError as error:
        if error.errno == errno.ENOENT:
            status = FileStatus.MISSING
        elif error.errno in (errno.ENOTDIR, errno.ELOOP):
            status = FileStatus.UNSAFE_PATH
        elif error.errno in (errno.EACCES, errno.EPERM):
            status = FileStatus.UNREADABLE
        else:
            status = FileStatus.IO_ERROR
    return result(status)
