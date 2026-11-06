"""Exact-module identification and reversible kernel instrumentation."""

import importlib
import threading

from .qwen_state import StateInterventionError
from . import _qwen_state_tensor as tensors

_NAMES = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")


class StateHook:
    def __init__(self, capture, model, *, tensor_ops=tensors):
        self.capture = capture
        self.tensor_ops = tensor_ops
        self.thread = threading.get_ident()
        self.torch = importlib.import_module("torch")
        self.module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        valid = False
        try:
            self.target = model.model.language_model.layers[capture.layer].linear_attn
            valid = (type(self.target) is self.module.Qwen3_5GatedDeltaNet
                     and self.target.layer_idx == capture.layer
                     and (self.target.num_v_heads, self.target.head_k_dim, self.target.head_v_dim) == (32, 128, 128))
            self.originals = [getattr(self.module, name) for name in _NAMES]
            valid = valid and all(callable(value) for value in self.originals)
        except (AttributeError, IndexError, TypeError):
            pass
        if not valid:
            raise StateInterventionError("unsupported_runtime")
        self.handles, self.wrappers = [], []
        self.active = self.failed = False
        self.target_calls = self.kernel_calls = 0

    @property
    def complete(self):
        return not self.active and not self.failed and self.target_calls == self.kernel_calls == 1

    def _thread_check(self):
        if threading.get_ident() != self.thread:
            raise StateInterventionError("busy")

    def _before(self, module, args):
        self._thread_check()
        if self.active or self.target_calls:
            self.failed = True
            raise StateInterventionError("invalid_call")
        self.target_calls += 1
        self.active = True

    def _after(self, module, args, output):
        if threading.get_ident() != self.thread:
            return
        if output is None:
            self.failed = True
        self.active = False

    def _wrapper(self, original, kind):
        def dispatch(*args, **kwargs):
            self._thread_check()
            if not self.active:
                return original(*args, **kwargs)
            if self.kernel_calls:
                self.failed = True
                raise StateInterventionError("invalid_call")
            self.kernel_calls += 1
            failure = None
            try:
                args, kwargs, metadata = self.tensor_ops.prepare_call(self.capture, self.torch, args, kwargs)
                output = original(*args, **kwargs)
                self.capture._pending = self.tensor_ops.observe_call(
                    self.capture, self.torch, args, kwargs, output, kind, metadata)
            except (StateInterventionError, KeyboardInterrupt, SystemExit, MemoryError) as error:
                failure = error
            except Exception:
                failure = StateInterventionError("execution_failed")
            if failure is not None:
                self.failed = True
                raise failure
            return output
        return dispatch

    def install(self):
        self.handles.append(self.target.register_forward_pre_hook(self._before))
        self.handles.append(self.target.register_forward_hook(self._after, always_call=True))
        for name, original, kind in zip(_NAMES, self.originals, ("chunk", "recurrent")):
            wrapper = self._wrapper(original, kind)
            self.wrappers.append((name, original, wrapper))
            setattr(self.module, name, wrapper)

    def close(self):
        failed = False
        interrupted = None
        for name, original, wrapper in reversed(self.wrappers):
            try:
                if getattr(self.module, name) is not wrapper:
                    failed = True
                else:
                    setattr(self.module, name, original)
            except (KeyboardInterrupt, SystemExit, MemoryError) as error:
                interrupted = interrupted or error
            except Exception:
                failed = True
        for handle in reversed(self.handles):
            try:
                handle.remove()
            except (KeyboardInterrupt, SystemExit, MemoryError) as error:
                interrupted = interrupted or error
            except Exception:
                failed = True
        self.wrappers.clear()
        self.handles.clear()
        self.capture = self.target = self.module = self.torch = self.originals = None
        self.tensor_ops = None
        if interrupted is not None:
            raise interrupted
        if failed:
            raise StateInterventionError("cleanup_failed")
