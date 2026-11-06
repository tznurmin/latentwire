"""Bounded declarations of expected artifact bytes, without filesystem access."""

from dataclasses import dataclass
from enum import Enum
import json
import re


SCHEMA = "latentwire.artifact-manifest.v1"
MAX_MANIFEST_BYTES = 131072
MAX_FILES = 256
MAX_FILE_BYTES = 2**40
MAX_TOTAL_BYTES = 2**42
_DIGEST = re.compile(r"[0-9a-f]{64}")
_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}
_FILE_FIELDS = {"path", "size_bytes", "sha256"}


class ManifestErrorCode(str, Enum):
    INPUT_TYPE = "input_type"
    MANIFEST_TOO_LARGE = "manifest_too_large"
    MALFORMED_JSON = "malformed_json"
    INVALID_MANIFEST = "invalid_manifest"


class ManifestError(ValueError):
    """A fixed public failure without artifact names or parser diagnostics."""

    def __init__(self, code: ManifestErrorCode | str):
        self.code = ManifestErrorCode(code)
        super().__init__(f"Invalid artifact manifest: {self.code.value}.")


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self):
        _validate_file(self)


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    files: tuple[ArtifactFile, ...]

    def __post_init__(self):
        object.__setattr__(self, "files", _validated_files(self.files))


def _path_valid(path):
    if type(path) is not str or not 1 <= len(path) <= 240:
        return False
    components = path.split("/")
    return len(components) <= 8 and all(
        _COMPONENT.fullmatch(part) is not None
        and not part.endswith(".")
        and part.split(".", 1)[0].upper() not in _RESERVED
        for part in components
    )


def _validate_file(value):
    if not (_path_valid(value.path)
            and type(value.size_bytes) is int and 0 <= value.size_bytes <= MAX_FILE_BYTES
            and type(value.sha256) is str and _DIGEST.fullmatch(value.sha256) is not None):
        raise ManifestError(ManifestErrorCode.INVALID_MANIFEST)


def _validate_addresses(files):
    paths = [file.path.lower() for file in files]
    addresses = set(paths)
    if len(addresses) != len(paths):
        raise ManifestError(ManifestErrorCode.INVALID_MANIFEST)
    for path in paths:
        components = path.split("/")
        if any("/".join(components[:i]) in addresses for i in range(1, len(components))):
            raise ManifestError(ManifestErrorCode.INVALID_MANIFEST)


def _validated_files(files):
    if type(files) is not tuple or not 1 <= len(files) <= MAX_FILES:
        raise ManifestError(ManifestErrorCode.INVALID_MANIFEST)
    if any(type(file) is not ArtifactFile for file in files):
        raise ManifestError(ManifestErrorCode.INVALID_MANIFEST)
    for file in files:
        _validate_file(file)
    if sum(file.size_bytes for file in files) > MAX_TOTAL_BYTES:
        raise ManifestError(ManifestErrorCode.INVALID_MANIFEST)
    _validate_addresses(files)
    return tuple(sorted(files, key=lambda file: file.path))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("non-finite constant")


def _read_json(data):
    failed = False
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
    except (UnicodeError, ValueError, RecursionError):
        failed = True
    if failed:
        raise ManifestError(ManifestErrorCode.MALFORMED_JSON)
    return value


def load_manifest(data: bytes) -> ArtifactManifest:
    """Admit expected file identities; this does not verify any actual file."""
    if type(data) is not bytes:
        raise ManifestError(ManifestErrorCode.INPUT_TYPE)
    if len(data) > MAX_MANIFEST_BYTES:
        raise ManifestError(ManifestErrorCode.MANIFEST_TOO_LARGE)
    value = _read_json(data)
    if type(value) is not dict or set(value) != {"schema", "files"} or value["schema"] != SCHEMA:
        raise ManifestError(ManifestErrorCode.INVALID_MANIFEST)
    files = value["files"]
    if type(files) is not list or not 1 <= len(files) <= MAX_FILES:
        raise ManifestError(ManifestErrorCode.INVALID_MANIFEST)
    if any(type(file) is not dict or set(file) != _FILE_FIELDS for file in files):
        raise ManifestError(ManifestErrorCode.INVALID_MANIFEST)
    return ArtifactManifest(tuple(ArtifactFile(**file) for file in files))


def dump_manifest(value: ArtifactManifest) -> bytes:
    """Encode canonical expected identities after revalidating the complete value."""
    if type(value) is not ArtifactManifest:
        raise ManifestError(ManifestErrorCode.INPUT_TYPE)
    files = _validated_files(value.files)
    document = {"schema": SCHEMA, "files": [
        {"path": file.path, "size_bytes": file.size_bytes, "sha256": file.sha256} for file in files
    ]}
    result = json.dumps(document, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=True, allow_nan=False).encode("utf-8")
    if len(result) > MAX_MANIFEST_BYTES:
        raise ManifestError(ManifestErrorCode.MANIFEST_TOO_LARGE)
    return result
