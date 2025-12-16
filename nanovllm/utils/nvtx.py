import contextlib

import torch


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


@contextlib.contextmanager
def nvtx_range(name: str):
    """Push/pop an NVTX range when CUDA is available."""
    pushed = False
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        pushed = True
    try:
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()
