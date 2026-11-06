"""The fixed retained-reader example; no fitting or output-score modification."""
import os
import sys

from . import _reader_identity as identity
from ._reader_errors import ReaderError
from ._reader_inputs import load_inputs
from ._reader_hook import MeasuredCapture, ReaderHook
from .artifact_files import VerificationError, VerificationLimits
from .manifest import ManifestError, load_manifest
from .qwen_checkpoint import CHECKPOINT_MANIFEST, CheckpointError, open_qwen_checkpoint
from .qwen_state import StateInterventionError
from .state_row import StateRow, StateRowError

_OWNER_ERRORS = (ReaderError, ManifestError, VerificationError,
                 CheckpointError, StateInterventionError, StateRowError)


def _admit_runtime(lease):
    import torch
    value = lease.identity
    if (value.model_id != identity.MODEL_ID or value.revision != identity.REVISION
            or value.manifest_sha256 != identity.MODEL_SHA256
            or value.model_source_sha256 != identity.MODEL_SOURCE_SHA256
            or value.kernel != identity.KERNEL or value.parameter_dtype != "bfloat16"
            or sys.byteorder != "little"
            or torch.get_float32_matmul_precision() != "highest"
            or torch.backends.cuda.matmul.allow_tf32
            or os.environ.get("NVIDIA_TF32_OVERRIDE") == "1"):
        raise ReaderError("unsupported_runtime")


def _matrix_on_cuda(array):
    import torch
    return torch.tensor(array, device="cuda:0", dtype=torch.float32)


def _prompt_ids(tokenizer):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": identity.PROMPT}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if ids != list(identity.PROMPT_IDS):
        raise ReaderError("unsupported_runtime")
    for label, token in zip((*identity.LABELS, "unknown"),
                            (*identity.TOKEN_IDS, 9496)):
        if tokenizer(" " + label, add_special_tokens=False)["input_ids"] != [token]:
            raise ReaderError("unsupported_runtime")
    return ids


def generate(lease, prompt_ids, payload, matrix, *, enabled):
    import torch
    from transformers.cache_utils import DynamicCache

    with torch.no_grad():
        cache = DynamicCache(config=lease.model.config)
        ids = torch.tensor([prompt_ids], device="cuda:0")
        lease.model(input_ids=ids[:, :-1], past_key_values=cache,
                    use_cache=True, logits_to_keep=1)
        layer = lease.model.model.language_model.layers[27]
        with MeasuredCapture(lease, layer=16, head=0, row=7, payload=payload,
                             suppress_first_decay=False) as state:
            with ReaderHook(layer, state, matrix, enabled=enabled):
                output = lease.model(input_ids=ids[:, -1:], past_key_values=cache,
                                     use_cache=True, logits_to_keep=1)
        logits = output.logits[0, -1].float()
        emitted = []
        for index in range(16):
            token = int(logits.argmax())
            emitted.append(token)
            if token in (248044, 248046) or index == 15:
                break
            output = lease.model(
                input_ids=torch.tensor([[token]], device="cuda:0"),
                past_key_values=cache, use_cache=True, logits_to_keep=1)
            logits = output.logits[0, -1].float()
    return {"token_ids": emitted,
            "text": lease.tokenizer.decode(emitted, skip_special_tokens=False)}


def _run(model_root, artifact_root):
    inputs = load_inputs(artifact_root)
    checkpoint_bytes = sum(f.size_bytes for f in load_manifest(CHECKPOINT_MANIFEST).files)
    limits = VerificationLimits(4, 1048576, checkpoint_bytes, 600)
    rows = []
    with open_qwen_checkpoint(model_root, limits) as lease:
        _admit_runtime(lease)
        prompt_ids = _prompt_ids(lease.tokenizer)
        matrix = _matrix_on_cuda(inputs.reader)
        for index, enabled in ((0, True), (1, True), (0, False), (None, True)):
            payload = None if index is None else StateRow(
                lease.identity.manifest_sha256, 16, 0, 7, "float32", 128,
                inputs.codewords[index].tobytes())
            generated = generate(lease, prompt_ids, payload, matrix, enabled=enabled)
            rows.append({"write": None if index is None else identity.LABELS[index],
                         "reader": enabled, **generated})
        del matrix
    return {"prompt": identity.PROMPT, "conditions": rows}


def run_example(model_root, artifact_root):
    failed = False
    try:
        return _run(model_root, artifact_root)
    except _OWNER_ERRORS:
        raise
    except MemoryError:
        raise
    except Exception:
        failed = True
    if failed:
        raise ReaderError("execution_failed")
