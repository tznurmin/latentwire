"""Fixed diagnostics for the reference reader."""
from enum import Enum


class ReaderErrorCode(str, Enum):
    INVALID_INPUTS = "invalid_inputs"
    UNSUPPORTED_RUNTIME = "unsupported_runtime"
    INVALID_OBSERVATION = "invalid_observation"
    EXECUTION_FAILED = "execution_failed"
    CLEANUP_FAILED = "cleanup_failed"


class ReaderError(ValueError):
    def __init__(self, code, *, verification=None):
        self.code = ReaderErrorCode(code)
        self.verification = verification
        super().__init__(f"Reader operation failed: {self.code.value}.")
