"""NamedShadow for cgf680.gemm_probe.

Kept in a file whose name ends with `shadow.py` on purpose: shadow._caller_frame skips frames of files ending with
"shadow.py" when it looks for the model source line of a dtype cast (explicit_fp32_at_call). A wrapper frame from any
other file would be taken as the caller and change FP16 shadow tags (seen on T5LayerNorm: upcast missed).
"""
from cgf680.shadow import ShadowMode


GEMM_NAMES = {"linear", "matmul", "__matmul__", "__rmatmul__", "addmm", "mm", "bmm", "baddbmm", "addbmm", "einsum",
              "mv", "addmv", "dot", "tensordot", "inner", "outer"}


def func_name(func):
    return getattr(func, "__name__", None) or str(func)


class FirstGemmReached(Exception):
    pass


class NamedShadow(ShadowMode):
    """ShadowMode that also remembers the torch function name of every counted call (self.calls numbering).
    abort=True: raise FirstGemmReached right after the first GEMM-type call returned (its output is recorded)."""

    def __init__(self, abort=False, abort_after=8):
        super().__init__()
        self.names = {}
        self.abort = abort
        self.abort_after = abort_after   # stop after this many GEMM calls: requests just before the stop point take
        self.n_gemm = 0                  # inferred tags from later calls (shadow.match_records), so stop well after

    def __torch_function__(self, func, types, args=(), kwargs=None):
        before = self.calls
        res = super().__torch_function__(func, types, args, kwargs)
        if self.calls > before:
            n = func_name(func)
            self.names[self.calls] = n
            if self.abort and n in GEMM_NAMES:
                self.n_gemm += 1
                if self.n_gemm >= self.abort_after:
                    raise FirstGemmReached(n)
        return res


class FirstGemmRecorder:
    """fp16_real worker mode: a minimal TorchFunctionMode (no dtype emulation) that records the output storage of the
    FIRST GEMM-type torch function, so the cuBLAS workspace position (right after that output request) is known from the
    same trace. Afterwards it only forwards calls."""

    def __new__(cls):
        from torch.overrides import TorchFunctionMode

        class _Rec(TorchFunctionMode):
            def __init__(self):
                super().__init__()
                self.records = []
                self.first_name = None
                self.calls = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                res = func(*args, **(kwargs or {}))
                if self.first_name is None:
                    self.calls += 1
                    n = func_name(func)
                    if n in GEMM_NAMES:
                        import torch
                        outs = [res] if isinstance(res, torch.Tensor) else [r for r in (res or ()) if isinstance(r, torch.Tensor)]
                        for o in outs:
                            st = o.untyped_storage()
                            self.records.append(("out", st.data_ptr(), st.nbytes(), "h", self.calls))
                        self.first_name = n
                return res
        return _Rec()
