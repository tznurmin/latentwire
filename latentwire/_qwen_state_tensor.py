"""Bounded CUDA edits and portable capture; no model loading or CPU forwards."""

import struct

from .qwen_state import StateInterventionError, StateObservation
from .state_row import StateRow


def _tensor(torch, value, shape, dtypes):
    return (type(value) is torch.Tensor and tuple(value.shape) == shape
            and value.device.type == "cuda" and value.device.index == 0
            and value.dtype in dtypes and bool(torch.isfinite(value).all()))


def prepare_call(capture, torch, args, kwargs):
    valid = len(args) == 3
    if not valid or type(args[0]) is not torch.Tensor or args[0].ndim != 4:
        raise StateInterventionError("invalid_call")
    tokens = args[0].shape[1]
    shape, gate_shape, state_shape = (1, tokens, 32, 128), (1, tokens, 32), (1, 32, 128, 128)
    valid = (1 <= tokens <= 256 and all(_tensor(torch, x, shape, (torch.bfloat16,)) for x in args)
             and _tensor(torch, kwargs.get("g"), gate_shape, (torch.float32,))
             and _tensor(torch, kwargs.get("beta"), gate_shape, (torch.bfloat16, torch.float32))
             and kwargs.get("output_final_state") is True
             and kwargs.get("use_qk_l2norm_in_kernel") is True
             and kwargs.get("cu_seqlens") is None)
    initial = kwargs.get("initial_state")
    valid = valid and (initial is None or _tensor(torch, initial, state_shape, (torch.bfloat16, torch.float32)))
    if not valid:
        raise StateInterventionError("invalid_call")
    edited = dict(kwargs)
    with torch.no_grad():
        if capture.payload is not None:
            state = (torch.zeros(state_shape, device=args[0].device, dtype=torch.float32)
                     if initial is None else initial.to(dtype=torch.float32, copy=True))
            dtype = torch.bfloat16 if capture.payload.dtype == "bfloat16" else torch.float32
            payload = torch.frombuffer(bytearray(capture.payload.data), dtype=dtype)
            state[0, capture.head, capture.row, :] = payload.to(device=state.device, dtype=state.dtype)
            edited["initial_state"] = state
        if capture.suppress_first_decay:
            decay = kwargs["g"].clone()
            decay[0, 0, capture.head] = 0
            edited["g"] = decay
    metadata = {"manifest_sha256": capture._checkpoint.identity.manifest_sha256,
                "initial_state_present": initial is not None}
    return args, edited, metadata


def observe_call(capture, torch, args, kwargs, output, kind, metadata):
    tokens = args[0].shape[1]
    valid = (type(output) is tuple and len(output) == 2
             and _tensor(torch, output[0], (1, tokens, 32, 128), (torch.bfloat16,))
             and _tensor(torch, output[1], (1, 32, 128, 128), (torch.bfloat16, torch.float32)))
    if not valid:
        raise StateInterventionError("invalid_result")
    with torch.no_grad():
        row = output[1][0, capture.head, capture.row, :].detach().to(torch.bfloat16)
        if not bool(torch.isfinite(row).all()):
            raise StateInterventionError("invalid_result")
        data = struct.pack("<128H", *row.contiguous().view(torch.uint16).cpu().tolist())
        readout = tuple(output[0][0, -1, capture.head].detach().float().cpu().tolist())
        query = tuple(args[0][0, -1, capture.head].detach().float().cpu().tolist())
    final_row = StateRow(metadata["manifest_sha256"], capture.layer, capture.head,
                         capture.row, "bfloat16", 128, data)
    return StateObservation(final_row, readout, query, kind, tokens,
        metadata["initial_state_present"], capture._payload_sha256,
        capture.suppress_first_decay, "float32" if output[1].dtype == torch.float32 else "bfloat16")
