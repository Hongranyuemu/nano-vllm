import contextlib

import torch


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
