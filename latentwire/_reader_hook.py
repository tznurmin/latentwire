"""A single measured-state reader invocation and its scoped restoration."""
from types import SimpleNamespace

from . import _qwen_state_tensor as tensor_owner
from .qwen_state import QwenStateCapture
from ._reader_errors import ReaderError


class MeasuredCapture(QwenStateCapture):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output = None

    def _make_hook(self, model):
        from ._qwen_state_hook import StateHook

        def observe(capture, torch, args, kwargs, output, kind, metadata):
            result = tensor_owner.observe_call(
                capture, torch, args, kwargs, output, kind, metadata)
            self.output = output
            return result

        operations = SimpleNamespace(
            prepare_call=lambda *a, **kw: tensor_owner.prepare_call(*a, **kw),
            observe_call=observe)
        return StateHook(self, model, tensor_ops=operations)

    def release_measurements(self):
        self.output = None

    def _release(self):
        try:
            return super()._release()
        finally:
            self.release_measurements()


def _updated_args(args, output, matrix, *, enabled):
    import torch
    failed = False
    try:
        signal_output = output[0] if type(output) is tuple and len(output) == 2 else output
        x = args[0] if type(args) is tuple and args else None

        def valid(value, shape, dtype):
            return (type(value) is torch.Tensor and tuple(value.shape) == shape
                    and value.device.type == "cuda" and value.device.index == 0
                    and value.dtype == dtype and bool(torch.isfinite(value).all()))

        if (not valid(x, (1,1,4096), torch.bfloat16)
                or not valid(signal_output, (1,1,32,128), torch.bfloat16)
                or not valid(matrix, (4096,128), torch.float32)
                or type(enabled) is not bool):
            raise ReaderError("invalid_observation")
        if not enabled:
            return args
        signal = signal_output[0,-1,0,:].float()
        unit = signal / torch.linalg.vector_norm(signal).clamp_min(1e-12)
        delta = matrix @ unit
        if not bool(torch.isfinite(delta).all()):
            raise ReaderError("invalid_observation")
        return (x + delta.to(x.dtype), *args[1:])
    except ReaderError:
        raise
    except Exception as error:
        if isinstance(error, MemoryError):
            raise
        failed = True
    if failed:
        raise ReaderError("execution_failed")


class ReaderHook:
    """Attach only for the selected forward; hold no writer payload or label."""
    def __init__(self, layer, capture, matrix, *, enabled):
        self.layer, self.capture, self.matrix = layer, capture, matrix
        self.enabled = enabled
        self.calls = 0
        self.cleanup_error = None
        self._handle = self._originals = None
        self._callback = self._before
        self._used = False

    def _before(self, module, args):
        self.calls += 1
        if self.calls != 1 or self.capture is None:
            raise ReaderError("invalid_observation")
        return _updated_args(args, self.capture.output, self.matrix,
                             enabled=self.enabled)

    def _close(self):
        failed, fatal = False, None
        try:
            if self._handle is not None:
                self._handle.remove()
        except (KeyboardInterrupt, SystemExit, MemoryError) as error:
            fatal, failed = error, True
        except Exception:
            failed = True
        try:
            # A failed registration may install our callback before returning
            # its handle. Remove only our own object, never another hook.
            if self.layer is not None and self._originals is not None:
                registry = self.layer._forward_pre_hooks
                for key, value in tuple(registry.items()):
                    if value is self._callback:
                        registry.pop(key)
                if dict(registry) != self._originals:
                    failed = True
        except (KeyboardInterrupt, SystemExit, MemoryError) as error:
            fatal, failed = fatal or error, True
        except Exception:
            failed = True
        finally:
            self.layer = self.capture = self.matrix = None
            self._handle = self._originals = self._callback = None
        if failed:
            self.cleanup_error = ReaderError("cleanup_failed")
        return fatal

    def __enter__(self):
        if self._used:
            raise ReaderError("invalid_observation")
        self._used = True
        failed, fatal = False, None
        try:
            self._originals = dict(self.layer._forward_pre_hooks)
            self._handle = self.layer.register_forward_pre_hook(self._callback)
        except (KeyboardInterrupt, SystemExit, MemoryError) as error:
            fatal = error
        except Exception:
            failed = True
        if failed or fatal is not None:
            cleanup_fatal = self._close()
            if fatal is not None:
                raise fatal
            if cleanup_fatal is not None:
                raise cleanup_fatal
            raise ReaderError("execution_failed")
        return self

    def __exit__(self, error_type, error, traceback):
        fatal = self._close()
        if error_type is not None:
            return False
        if fatal is not None:
            raise fatal
        if self.cleanup_error is not None:
            raise self.cleanup_error
        if self.calls != 1:
            raise ReaderError("invalid_observation")
        return False
