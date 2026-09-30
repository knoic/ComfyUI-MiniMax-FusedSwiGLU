"""Complete verification test suite for ComfyUI-MiniMax-FusedSwiGLU."""

import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

# Mock ComfyUI environment
comfy_mod = types.ModuleType("comfy")
ops_mod = types.ModuleType("comfy.ops")
quant_mod = types.ModuleType("comfy.quant_ops")


class _QuantizedTensor:
    pass


class CastBiasWeightContext:
    def __init__(self, slf, input=None, offloadable=False, want_requant=False):
        w = getattr(slf, "weight", None)
        b = getattr(slf, "bias", None)
        if w is not None and input is not None and hasattr(w, "dtype") and w.dtype != input.dtype:
            w = w.to(input.dtype)
        self.state = (w, b)

    def __enter__(self):
        return self.state

    def __exit__(self, *_):
        return False


def linear_input_act(linear, x, input_act, *args, **kwargs):
    if input_act == "swiglu":
        gate, up = x.chunk(2, dim=-1)
        x = F.silu(gate).mul_(up)
    return linear(x)


class _TWI:
    @staticmethod
    def get_plain_tensors(w):
        raise RuntimeError("INT8 mocked")


class _CK:
    @staticmethod
    def int8_linear(*a, **k):
        raise RuntimeError("INT8 mocked")


ops_mod.CastBiasWeightContext = CastBiasWeightContext
ops_mod.linear_input_act = linear_input_act
quant_mod.QuantizedTensor = _QuantizedTensor
quant_mod.TensorWiseINT8Layout = _TWI
quant_mod.ck = _CK
comfy_mod.ops = ops_mod
comfy_mod.quant_ops = quant_mod
sys.modules["comfy"] = comfy_mod
sys.modules["comfy.ops"] = ops_mod
sys.modules["comfy.quant_ops"] = quant_mod

from fused_swiglu import (
    triton_swiglu,
    triton_swiglu_split,
    triton_addcmul_,
    triton_modulate_,
    HAS_TRITON,
)
from nodes import (
    token_stream_mlp_forward,
    channel_fused_mlp_forward,
    _e2e_dit_mlp_pipeline,
)


class FakeMLP(nn.Module):
    def __init__(self, H, Fd, dtype, device):
        super().__init__()
        self.fc1 = nn.Linear(H, 2 * Fd, bias=True, dtype=dtype, device=device)
        self.fc2 = nn.Linear(Fd, H, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        h = self.fc1(x)
        gate, up = h.chunk(2, dim=-1)
        return self.fc2(F.silu(gate).mul_(up))


class FakeNorm(nn.Module):
    def __init__(self, H, dtype, device):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))

    def forward(self, x):
        return x * self.weight


class FakeBlock(nn.Module):
    def __init__(self, H, Fd, dtype, device):
        super().__init__()
        self.norm2 = FakeNorm(H, dtype, device)
        self.mlp = FakeMLP(H, Fd, dtype, device)


def run_tests():
    print("=" * 60)
    print("Running ComfyUI-MiniMax-FusedSwiGLU Verification Test Suite")
    print("=" * 60)

    PASS, FAIL = 0, 0

    def check(name, cond, detail=""):
        nonlocal PASS, FAIL
        if cond:
            PASS += 1
            print(f"  [PASS] {name}")
        else:
            FAIL += 1
            print(f"  [FAIL] {name}  {detail}")

    # 1. GPU Triton Micro-Kernels
    if torch.cuda.is_available() and HAS_TRITON:
        dev = torch.device("cuda")
        dtype = torch.bfloat16
        print("\n== 1. GPU Triton Kernel Correctness & Bit-Exactness ==")

        # SwiGLU inplace
        x = torch.randn(1024, 2 * 14336, device=dev, dtype=dtype)
        gate, up = x.chunk(2, dim=-1)
        ref_swiglu = F.silu(gate).mul_(up)
        out_inplace = triton_swiglu(x.clone(), inplace=True, bit_exact=True)
        check("triton_swiglu inplace bit-exact == eager", torch.equal(out_inplace, ref_swiglu))

        # SwiGLU split
        g = torch.randn(1024, 14336, device=dev, dtype=dtype)
        u = torch.randn(1024, 14336, device=dev, dtype=dtype)
        ref_split = F.silu(g).mul_(u)
        out_split = triton_swiglu_split(g, u, bit_exact=True)
        check("triton_swiglu_split bit-exact == eager", torch.equal(out_split, ref_split))

        # Fused Modulate
        h = torch.randn(1024, 6144, device=dev, dtype=dtype)
        s = torch.randn(6144, device=dev, dtype=dtype)
        sh = torch.randn(6144, device=dev, dtype=dtype)
        ref_mod = h.clone().mul_(s).add_(sh)
        out_mod = triton_modulate_(h.clone(), s, sh)
        mod_diff = (out_mod - ref_mod).abs().max().item()
        check("triton_modulate_ within 1 ULP of eager", mod_diff <= 0.0625, f"maxdiff={mod_diff}")

        # Addcmul
        xx = torch.randn(1024, 6144, device=dev, dtype=dtype)
        rr = torch.randn(1024, 6144, device=dev, dtype=dtype)
        gg = torch.randn(6144, device=dev, dtype=dtype)
        ref_add = xx.clone().addcmul_(rr, gg)
        out_add = triton_addcmul_(xx.clone(), rr, gg)
        check("triton_addcmul_ == eager", torch.equal(out_add, ref_add))

    # 2. Pipeline Equivalence on CUDA
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        dtype = torch.bfloat16
        H, Fd, S = 6144, 14336, 4000
        print(f"\n== 2. Pipeline Equivalence on CUDA (S={S}, H={H}, F={Fd}) ==")

        mlp = FakeMLP(H, Fd, dtype, dev).eval()
        for p in mlp.parameters():
            p.requires_grad = False
        x = torch.randn(S, H, device=dev, dtype=dtype)

        with torch.no_grad():
            ref_mlp = mlp(x)
            out_tok = token_stream_mlp_forward(mlp, x, tile_size=1024, use_triton=True)
            diff_tok = (out_tok - ref_mlp).abs().max().item()
            check("token_stream_mlp_forward == eager", diff_tok <= 0.01, f"maxdiff={diff_tok:.3e}")

            out_ch = channel_fused_mlp_forward(mlp, x, num_chunks=4, tile_size=1024, use_triton=True)
            diff_ch = (out_ch - ref_mlp).abs().max().item()
            check("channel_fused_mlp_forward == eager", diff_ch <= 0.01, f"maxdiff={diff_ch:.3e}")

        # 3. End-to-End DiT Block Pipeline
        print(f"\n== 3. End-to-End DiT Block Pipeline ==")
        block = FakeBlock(H, Fd, dtype, dev).eval()
        for p in block.parameters():
            p.requires_grad = False

        shift = torch.randn(6, H, device=dev, dtype=dtype)
        scale = torch.randn(6, H, device=dev, dtype=dtype)
        gate = torch.randn(6, H, device=dev, dtype=dtype)
        segments = [(0, 2500, 0), (2500, 4000, 1)]

        def eager_block_ref(b_mod, x_in, s_mlp, sc_mlp, g_mlp, segs):
            x_res = x_in.clone()
            for a, b, row in segs:
                ch = b_mod.norm2(x_res[a:b])
                s_v = (1.0 + sc_mlp[row]).to(dtype)
                sh_v = s_mlp[row].to(dtype)
                ch = ch * s_v + sh_v
                m_out = b_mod.mlp(ch)
                x_res[a:b].addcmul_(m_out, g_mlp[row].to(dtype))
            return x_res

        with torch.no_grad():
            ref_e2e = eager_block_ref(block, x, shift, scale, gate, segments)
            out_e2e = _e2e_dit_mlp_pipeline(
                block, x.clone(), shift, scale, gate, segments, tile_size=1024, use_triton=True
            )
            diff_e2e = (out_e2e - ref_e2e).abs().max().item()
            check("End-to-End DiT Pipeline == eager block", diff_e2e <= 0.05, f"maxdiff={diff_e2e:.3e}")

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
