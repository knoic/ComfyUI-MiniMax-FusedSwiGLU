import logging
import torch

log = logging.getLogger("MiniMax-FusedSwiGLU")

HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _swiglu_kernel(
        in_ptr, out_ptr,
        stride_in_row, stride_in_col,
        stride_out_row, stride_out_col,
        d,
        BLOCK_SIZE: tl.constexpr,
        CONTIG: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_block_idx = tl.program_id(1)

        col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < d

        if CONTIG:
            gate_ptrs = in_ptr + row_idx * stride_in_row + col_offsets
            up_ptrs = in_ptr + row_idx * stride_in_row + d + col_offsets
            out_ptrs = out_ptr + row_idx * stride_out_row + col_offsets
        else:
            gate_ptrs = in_ptr + row_idx * stride_in_row + col_offsets * stride_in_col
            up_ptrs = in_ptr + row_idx * stride_in_row + (d + col_offsets) * stride_in_col
            out_ptrs = out_ptr + row_idx * stride_out_row + col_offsets * stride_out_col

        # Load gate and up into registers/SRAM, computing SiLU in FP32 for numerical stability
        gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)

        silu_gate = gate * tl.sigmoid(gate)
        res = silu_gate * up

        tl.store(out_ptrs, res.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _swiglu_split_kernel(
        g_ptr, u_ptr, out_ptr,
        stride_gm, stride_gn,
        stride_um, stride_un,
        stride_om, stride_on,
        d,
        BLOCK_SIZE: tl.constexpr,
        CONTIG: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_block_idx = tl.program_id(1)

        col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < d

        if CONTIG:
            g_ptrs = g_ptr + row_idx * stride_gm + col_offsets
            u_ptrs = u_ptr + row_idx * stride_um + col_offsets
            out_ptrs = out_ptr + row_idx * stride_om + col_offsets
        else:
            g_ptrs = g_ptr + row_idx * stride_gm + col_offsets * stride_gn
            u_ptrs = u_ptr + row_idx * stride_um + col_offsets * stride_un
            out_ptrs = out_ptr + row_idx * stride_om + col_offsets * stride_on

        g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_ptrs, mask=mask, other=0.0).to(tl.float32)

        res = (g * tl.sigmoid(g)) * u
        tl.store(out_ptrs, res.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _addcmul_inplace_kernel(
        x_ptr, res_ptr, g_ptr,
        stride_xm, stride_xn,
        stride_rm, stride_rn,
        stride_gm, stride_gn,
        N,
        BLOCK_N: tl.constexpr,
        CONTIG: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = col_offsets < N

        if CONTIG:
            x_off = pid_m * stride_xm + col_offsets
            r_off = pid_m * stride_rm + col_offsets
            g_off = pid_m * stride_gm + col_offsets
        else:
            x_off = pid_m * stride_xm + col_offsets * stride_xn
            r_off = pid_m * stride_rm + col_offsets * stride_rn
            g_off = pid_m * stride_gm + col_offsets * stride_gn

        g = tl.load(g_ptr + g_off, mask=mask, other=0.0).to(tl.float32)
        x_val = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
        res_val = tl.load(res_ptr + r_off, mask=mask, other=0.0).to(tl.float32)

        out = x_val + res_val * g
        tl.store(x_ptr + x_off, out.to(x_ptr.dtype.element_ty), mask=mask)

    HAS_TRITON = True
except Exception as e:
    log.warning(f"Triton is not available for Fused SwiGLU ({e}). Falling back to PyTorch implementation.")


def triton_swiglu(x: torch.Tensor) -> torch.Tensor:
    """Computes Fused SwiGLU: SiLU(gate) * up along the last dimension.
    
    x: [..., 2 * d]
    returns: [..., d]
    """
    orig_shape = x.shape
    total_d = orig_shape[-1]
    if total_d % 2 != 0:
        log.warning(f"triton_swiglu: expected last dimension to be even, got {total_d}. Falling back to standard PyTorch.")
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate).mul_(up)

    d = total_d // 2

    if not x.is_cuda or not HAS_TRITON:
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate).mul_(up)

    if not x.is_contiguous():
        x = x.contiguous()

    x_2d = x.view(-1, total_d)
    rows = x_2d.shape[0]

    out_2d = torch.empty((rows, d), device=x.device, dtype=x.dtype)
    BLOCK_SIZE = 1024
    grid = (rows, triton.cdiv(d, BLOCK_SIZE))
    contig = (x_2d.stride(1) == 1 and out_2d.stride(1) == 1)

    _swiglu_kernel[grid](
        x_2d, out_2d,
        x_2d.stride(0), x_2d.stride(1),
        out_2d.stride(0), out_2d.stride(1),
        d,
        BLOCK_SIZE=BLOCK_SIZE,
        CONTIG=contig,
        num_warps=4
    )

    if len(orig_shape) != 2:
        return out_2d.view(*orig_shape[:-1], d)
    return out_2d


def triton_swiglu_split(g: torch.Tensor, u: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Computes Fused SwiGLU from separate gate and up tensors: SiLU(g) * u.
    
    Eliminates intermediate tensor allocations and executes in SRAM/registers.
    """
    if not g.is_cuda or not HAS_TRITON:
        res = torch.nn.functional.silu(g).mul_(u)
        if out is not None:
            out.copy_(res)
            return out
        return res

    orig_shape = g.shape
    d = orig_shape[-1]
    g_2d = g.view(-1, d)
    u_2d = u.view(-1, d)
    rows = g_2d.shape[0]

    if out is None:
        out_2d = torch.empty((rows, d), device=g.device, dtype=g.dtype)
    else:
        out_2d = out.view(-1, d)

    BLOCK_SIZE = 1024
    grid = (rows, triton.cdiv(d, BLOCK_SIZE))
    contig = (g_2d.stride(1) == 1 and u_2d.stride(1) == 1 and out_2d.stride(1) == 1)

    _swiglu_split_kernel[grid](
        g_2d, u_2d, out_2d,
        g_2d.stride(0), g_2d.stride(1),
        u_2d.stride(0), u_2d.stride(1),
        out_2d.stride(0), out_2d.stride(1),
        d,
        BLOCK_SIZE=BLOCK_SIZE,
        CONTIG=contig,
        num_warps=4
    )

    if out is None and len(orig_shape) != 2:
        return out_2d.view(orig_shape)
    return out if out is not None else out_2d


def triton_addcmul_(x: torch.Tensor, res: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Computes in-place gated residual accumulation: x += res * g.
    
    Bypasses PyTorch TensorIterator launch overhead and performs in-place
    accumulation directly in registers/SRAM with zero VRAM allocation.
    """
    if not x.is_cuda or not HAS_TRITON:
        return x.addcmul_(res, g)

    orig_shape = x.shape
    N = orig_shape[-1]
    M = x.numel() // N

    if x.ndim == 2 and res.ndim == 2:
        stride_xm, stride_xn = x.stride(0), x.stride(1)
        stride_rm, stride_rn = res.stride(0), res.stride(1)
    elif x.is_contiguous() and res.is_contiguous():
        stride_xm, stride_xn = N, 1
        stride_rm, stride_rn = N, 1
    else:
        return x.addcmul_(res, g)

    if g.numel() == N:
        stride_gm = 0
        stride_gn = g.stride(-1) if g.ndim > 0 else 0
    elif g.ndim == 2 and g.shape[0] == M and g.shape[1] == N:
        stride_gm, stride_gn = g.stride(0), g.stride(1)
    elif g.numel() == M * N and g.is_contiguous():
        stride_gm = N
        stride_gn = 1
    else:
        return x.addcmul_(res, g)

    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    contig = (stride_xn == 1 and stride_rn == 1 and (stride_gn == 1 or stride_gn == 0))

    _addcmul_inplace_kernel[grid](
        x, res, g,
        stride_xm, stride_xn,
        stride_rm, stride_rn,
        stride_gm, stride_gn,
        N,
        BLOCK_N=BLOCK_N,
        CONTIG=contig,
        num_warps=4
    )
    return x


__all__ = ["triton_swiglu", "triton_swiglu_split", "triton_addcmul_", "HAS_TRITON"]
