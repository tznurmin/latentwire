"""Concrete optional provider for the pinned reference-kernel CUDA checkpoint."""

import gc
from functools import partial
import hashlib
import importlib
import importlib.util
from importlib import metadata
import os
import platform
import sys

from ._qwen_checkpoint_data import CheckpointError, CheckpointIdentity, MODEL_ID, MODEL_REVISION

_VERSIONS = {"torch": "2.13.0", "transformers": "5.17.0", "tokenizers": "0.23.2", "safetensors": "0.8.0"}
_FLAGS = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
          "USE_HUB_KERNELS": "0", "TOKENIZERS_PARALLELISM": "false"}


def _environment():
    valid = all(os.environ.get(key) == value for key, value in _FLAGS.items())
    try:
        valid = valid and all(metadata.version(key) == value for key, value in _VERSIONS.items())
        valid = valid and all(name not in sys.modules and importlib.util.find_spec(name) is None
                              for name in ("fla", "causal_conv1d", "kernels"))
    except (ValueError, ImportError):
        valid = False
    if not valid:
        raise CheckpointError("unsupported_environment")


def _source_digest(module):
    failed = False
    try:
        with open(module.__file__, "rb") as source:
            data = source.read(262145)
        failed = len(data) > 262144 or not data
    except (OSError, TypeError, AttributeError):
        failed = True
    if failed:
        raise CheckpointError("unsupported_environment")
    return hashlib.sha256(data).hexdigest()


def _placement(model, torch):
    count = 0
    for parameter in model.parameters():
        count += 1
        if (parameter.device.type != "cuda" or parameter.device.index != 0
                or parameter.dtype != torch.bfloat16):
            raise CheckpointError("placement_mismatch")
    if not count:
        raise CheckpointError("placement_mismatch")


def _select_aten(state):
    valid = False
    try:
        registry = state.torch._native.registry
        disabled = registry._filter_state._dispatch_keys
        valid = (type(disabled) is set and all(type(key) is str for key in disabled)
                 and callable(registry.deregister_op_overrides)
                 and callable(registry.reenable_op_overrides))
    except (AttributeError, TypeError):
        pass
    if not valid:
        raise CheckpointError("unsupported_environment")
    if "CUDA" not in disabled:
        state._restore_dispatch = partial(registry.reenable_op_overrides, enable_dispatch_keys=["CUDA"])
        registry.deregister_op_overrides(disable_dispatch_keys=["CUDA"])
        if "CUDA" not in registry._filter_state._dispatch_keys:
            raise CheckpointError("unsupported_environment")


class _NativeCheckpoint:
    def __init__(self):
        self.model = self.tokenizer = self.identity = self.torch = None
        self._cuda_owned = self._closed = False
        self._restore_dispatch = None

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.model = self.tokenizer = None
        failed = False
        try:
            gc.collect()
            if self._cuda_owned:
                self.torch.cuda.empty_cache()
        except Exception:
            failed = True
        restore, self._restore_dispatch = self._restore_dispatch, None
        try:
            if restore is not None:
                restore()
        except Exception:
            failed = True
        if failed:
            raise CheckpointError("cleanup_failed")


def _populate(state, root, manifest_sha256):
    _environment()
    state.torch = torch = importlib.import_module("torch")
    _select_aten(state)
    transformers = importlib.import_module("transformers")
    module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    source_digest = _source_digest(module)
    if not (torch.cuda.is_available() and torch.cuda.device_count() == 1
            and torch.cuda.is_bf16_supported(including_emulation=False)):
        raise CheckpointError("cuda_unavailable")
    state._cuda_owned = True
    if torch.cuda.mem_get_info(0)[0] < 22 * 2**30:
        raise CheckpointError("insufficient_vram")
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    state.model, info = module.Qwen3_5ForConditionalGeneration.from_pretrained(
        root, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="eager",
        use_kernels=False, output_loading_info=True)
    fields = {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
    if type(info) is not dict or set(info) != fields or any(info.values()):
        raise CheckpointError("load_failed")
    _placement(state.model, torch)
    state.model.eval()
    state.model.requires_grad_(False)
    state.tokenizer = transformers.AutoTokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=False)
    state.identity = CheckpointIdentity(
        MODEL_ID, MODEL_REVISION, manifest_sha256, platform.python_version(),
        _VERSIONS["torch"], _VERSIONS["transformers"], _VERSIONS["tokenizers"],
        _VERSIONS["safetensors"], source_digest, "transformers-5.17.0-torch-reference-aten-cuda",
        "bfloat16", torch.cuda.get_device_name(0))


def load_native(root, manifest_sha256):
    state = _NativeCheckpoint()
    code = interrupt = None
    try:
        _populate(state, root, manifest_sha256)
    except CheckpointError as error:
        code = error.code
    except (KeyboardInterrupt, SystemExit) as error:
        interrupt = error.with_traceback(None)
    except Exception:
        code = "load_failed"
    if code is not None or interrupt is not None:
        # Foreign traceback frames no longer retain partially loaded model objects.
        state.close()
        if interrupt is not None:
            raise interrupt
        raise CheckpointError(code)
    return state
