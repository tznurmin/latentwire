"""Portable finite state rows, preserving numeric bits and explicit geometry."""

import base64
import binascii
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
import struct


SCHEMA = "latentwire.state-row.v1"
MAX_PAYLOAD_BYTES = 32768
MAX_WIDTH = 4096
_DIGEST = re.compile(r"[0-9a-f]{64}")
_FORMATS = {"bfloat16": (2, "<H", 0x7F80), "float32": (4, "<I", 0x7F800000)}
_FIELDS = {"schema", "model_sha256", "layer", "head", "row", "dtype", "width", "data_base64"}


class StateRowErrorCode(str, Enum):
    INPUT_TYPE = "input_type"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    INVALID_DIGEST = "invalid_digest"
    DIGEST_MISMATCH = "digest_mismatch"
    MALFORMED_JSON = "malformed_json"
    INVALID_ROW = "invalid_row"


class StateRowError(ValueError):
    """A typed failure without input-bearing diagnostic details."""

    def __init__(self, code: StateRowErrorCode | str):
        self.code = StateRowErrorCode(code)
        super().__init__(f"Invalid state row: {self.code.value}.")


@dataclass(frozen=True, slots=True)
class StateRow:
    """One little-endian row; the model manifest is an external identity."""

    model_sha256: str
    layer: int
    head: int
    row: int
    dtype: str
    width: int
    data: bytes

    def __post_init__(self):
        _validate(self)


def _digest_valid(value):
    return type(value) is str and _DIGEST.fullmatch(value) is not None


def _validate(value):
    valid = _digest_valid(value.model_sha256)
    valid = valid and all(type(index) is int and 0 <= index <= 65535
                          for index in (value.layer, value.head, value.row))
    valid = valid and type(value.dtype) is str and value.dtype in _FORMATS
    valid = valid and type(value.width) is int and 1 <= value.width <= MAX_WIDTH
    valid = valid and type(value.data) is bytes
    if not valid:
        raise StateRowError(StateRowErrorCode.INVALID_ROW)
    size, layout, exponent = _FORMATS[value.dtype]
    if len(value.data) != value.width * size:
        raise StateRowError(StateRowErrorCode.INVALID_ROW)
    if any(bits & exponent == exponent for (bits,) in struct.iter_unpack(layout, value.data)):
        raise StateRowError(StateRowErrorCode.INVALID_ROW)


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
        raise StateRowError(StateRowErrorCode.MALFORMED_JSON)
    return value


def _read_base64(text):
    if type(text) is not str or len(text) > 4 * ((MAX_WIDTH * 4 + 2) // 3):
        raise StateRowError(StateRowErrorCode.INVALID_ROW)
    failed = False
    try:
        raw = base64.b64decode(text, validate=True)
    except (ValueError, binascii.Error):
        failed = True
    if failed:
        raise StateRowError(StateRowErrorCode.INVALID_ROW)
    if base64.b64encode(raw).decode("ascii") != text:
        raise StateRowError(StateRowErrorCode.INVALID_ROW)
    return raw


def load_state_row(data: bytes, expected_sha256: str) -> StateRow:
    """Verify the caller's wire digest, then admit one bounded state row."""
    if type(data) is not bytes:
        raise StateRowError(StateRowErrorCode.INPUT_TYPE)
    if len(data) > MAX_PAYLOAD_BYTES:
        raise StateRowError(StateRowErrorCode.PAYLOAD_TOO_LARGE)
    if not _digest_valid(expected_sha256):
        raise StateRowError(StateRowErrorCode.INVALID_DIGEST)
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise StateRowError(StateRowErrorCode.DIGEST_MISMATCH)
    value = _read_json(data)
    if type(value) is not dict or set(value) != _FIELDS or value["schema"] != SCHEMA:
        raise StateRowError(StateRowErrorCode.INVALID_ROW)
    return StateRow(
        model_sha256=value["model_sha256"], layer=value["layer"], head=value["head"],
        row=value["row"], dtype=value["dtype"], width=value["width"],
        data=_read_base64(value["data_base64"]),
    )


def dump_state_row(value: StateRow) -> bytes:
    """Encode exact finite scalar bits with canonical public metadata."""
    if type(value) is not StateRow:
        raise StateRowError(StateRowErrorCode.INPUT_TYPE)
    _validate(value)
    document = {
        "schema": SCHEMA, "model_sha256": value.model_sha256,
        "layer": value.layer, "head": value.head, "row": value.row,
        "dtype": value.dtype, "width": value.width,
        "data_base64": base64.b64encode(value.data).decode("ascii"),
    }
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_PAYLOAD_BYTES:
        raise StateRowError(StateRowErrorCode.PAYLOAD_TOO_LARGE)
    return encoded
