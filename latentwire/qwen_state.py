"""One explicit Qwen kernel-input intervention and portable raw observation."""

from dataclasses import dataclass
from enum import Enum
import hashlib
import threading

from .qwen_checkpoint import CheckpointError, QwenCheckpoint
from .state_row import StateRow, StateRowError, dump_state_row

_LOCK = threading.Lock()
_SOURCE = "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"
_KERNEL = "transformers-5.17.0-torch-reference-aten-cuda"


class StateInterventionErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_RUNTIME = "unsupported_runtime"
    BUSY = "busy"
    INVALID_CALL = "invalid_call"
    INVALID_RESULT = "invalid_result"
    EXECUTION_FAILED = "execution_failed"
    INCOMPLETE_CAPTURE = "incomplete_capture"
    CLOSED = "closed"
    CLEANUP_FAILED = "cleanup_failed"


class StateInterventionError(ValueError):
    """A fixed diagnostic without supplied content or native failure details."""

    def __init__(self, code):
        self.code = StateInterventionErrorCode(code)
        super().__init__(f"State intervention failed: {self.code.value}.")


@dataclass(frozen=True, slots=True)
class StateObservation:
    final_row: StateRow
    raw_readout: tuple[float, ...]
    raw_query: tuple[float, ...]
    kernel: str
    token_count: int
    initial_state_present: bool
    payload_sha256: str | None
    suppress_first_decay: bool
    native_state_dtype: str


class QwenStateCapture:
    """Single-use instrumentation; execute one forward inside its context."""

    def __init__(self, checkpoint, *, layer, head, row, payload=None, suppress_first_decay=False):
        if (type(checkpoint) is not QwenCheckpoint or type(suppress_first_decay) is not bool
                or any(type(v) is not int or not 0 <= v < bound
                       for v, bound in ((layer, 32), (head, 32), (row, 128)))):
            raise StateInterventionError("invalid_request")
        payload_hash = None
        if payload is not None:
            wire = dump_state_row(payload)
            if (payload.layer, payload.head, payload.row, payload.width) != (layer, head, row, 128):
                raise StateInterventionError("invalid_request")
            payload_hash = hashlib.sha256(wire).hexdigest()
        self._checkpoint, self._site = checkpoint, (layer, head, row)
        self._payload, self._suppress = payload, suppress_first_decay
        self._payload_sha256 = payload_hash
        self._used = self._locked = False
        self._hook = self._pending = self._observation = None
        self.cleanup_error = None

    @property
    def layer(self):
        return self._site[0]

    @property
    def head(self):
        return self._site[1]

    @property
    def row(self):
        return self._site[2]

    @property
    def payload(self):
        return self._payload

    @property
    def suppress_first_decay(self):
        return self._suppress

    @property
    def observation(self):
        if self._observation is None:
            raise StateInterventionError("closed")
        return self._observation

    def _release(self):
        interrupted = None
        try:
            if self._hook is not None:
                self._hook.close()
        except (KeyboardInterrupt, SystemExit, MemoryError) as error:
            interrupted = error
            self.cleanup_error = StateInterventionError("cleanup_failed")
        except Exception:
            self.cleanup_error = StateInterventionError("cleanup_failed")
        finally:
            self._hook = self._checkpoint = None
            if self._locked:
                self._locked = False
                _LOCK.release()
        return interrupted

    def _make_hook(self, model):
        from ._qwen_state_hook import StateHook

        return StateHook(self, model)

    def __enter__(self):
        if self._used:
            raise StateInterventionError("closed")
        self._used = True
        if not _LOCK.acquire(blocking=False):
            self._checkpoint = None
            raise StateInterventionError("busy")
        self._locked = True
        failure = None
        try:
            model = self._checkpoint.model
            identity = self._checkpoint.identity
            if self.payload is not None and self.payload.model_sha256 != identity.manifest_sha256:
                raise StateInterventionError("invalid_request")
            if identity.model_source_sha256 != _SOURCE or identity.kernel != _KERNEL:
                raise StateInterventionError("unsupported_runtime")
            self._hook = self._make_hook(model)
            self._hook.install()
        except (StateInterventionError, StateRowError, CheckpointError,
                KeyboardInterrupt, SystemExit, MemoryError) as error:
            failure = error
        except Exception:
            failure = StateInterventionError("execution_failed")
        if failure is not None:
            self._release()
            raise failure
        return self

    def __exit__(self, exception_type, exception, traceback):
        complete = self._hook is not None and self._hook.complete and self._pending is not None
        interrupted = self._release()
        pending, self._pending = self._pending, None
        if exception_type is not None:
            return False
        if interrupted is not None:
            raise interrupted
        if self.cleanup_error is not None:
            raise self.cleanup_error
        if not complete:
            raise StateInterventionError("incomplete_capture")
        self._observation = pending
        return False


def capture_qwen_state(checkpoint, *, layer, head, row, payload=None, suppress_first_decay=False):
    """Scope a single selected native call; the caller owns model execution."""
    return QwenStateCapture(checkpoint, layer=layer, head=head, row=row,
                            payload=payload, suppress_first_decay=suppress_first_decay)
