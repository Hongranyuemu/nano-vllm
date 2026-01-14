import os
from contextlib import contextmanager
from threading import local

try:
    import torch
    from torch.cuda import nvtx as torch_nvtx
except Exception:    # pragma: no cover - torch import varies by env
    torch = None
    torch_nvtx = None


def layer_nvtx_name(name: str, layer_id: int | None) -> str:
    """Insert the decoder layer identifier into an NVTX label."""
    if layer_id is None:
        return name
    layer_tag = f"decoder_layer_{layer_id}"
    if layer_tag in name:
        return name
    if "::" in name:
        prefix, rest = name.split("::", 1)
        return f"{prefix}::{layer_tag}::{rest}"
    return f"{layer_tag}::{name}"


def _env_enabled() -> bool:
    flag = os.getenv("ENABLE_NVTX", "").strip().lower()
    return flag not in ("", "0", "false", "off")


_ENABLE_NVTX = bool(torch) and bool(torch_nvtx) and _env_enabled() and torch.cuda.is_available()
_meta_ctx = local()


def nvtx_enabled() -> bool:
    return _ENABLE_NVTX


@contextmanager
def nvtx_meta(step_id=None, bs=None, seqlen=None, mode=None, rank=None):
    # Metadata hook reserved for future structured tagging; currently just a guard.
    if not _ENABLE_NVTX:
        yield
        return
    yield


@contextmanager
def nvtx_range(msg: str):
    if not _ENABLE_NVTX or not msg:
        yield
        return
    torch_nvtx.range_push(msg)
    try:
        yield
    finally:
        torch_nvtx.range_pop()


@contextmanager
def nvtx_step(step_id, bs, seqlen, mode, rank=None, emit_range: bool = True):
    if not _ENABLE_NVTX:
        yield
        return
    with nvtx_meta(step_id=step_id, bs=bs, seqlen=seqlen, mode=mode, rank=rank):
        if emit_range:
            with nvtx_range(step_tag(step_id, bs, seqlen, mode, rank)):
                yield
        else:
            yield


def run_tag(rank=None) -> str:
    return "run" if rank is None else f"run rank:{rank}"


def step_tag(step_id, bs, seqlen, mode, rank=None) -> str:
    tag = f"step:{step_id} bs:{bs} seq:{seqlen} mode:{mode}"
    if rank is not None:
        tag += f" rank:{rank}"
    return tag


def layer_tag(layer_id, step_id=None, bs=None, seqlen=None, mode=None, rank=None) -> str:
    # LAYER tag must stay concise: L{layer_id}
    return f"L{layer_id}"


def op_tag(name, step_id=None, bs=None, seqlen=None, mode=None, rank=None) -> str:
    return f"op:{name}"
