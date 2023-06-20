import torch
import triton
import triton.language as tl

from einops import repeat


@triton.jit
def ball_query_kernel(
        x_ptr, y_ptr, out_ptr, temp_ptr,
        radius,
        N, M, K, C,
        stride_xb, stride_xn, stride_xc,
        stride_yb, stride_ym, stride_yc,
        stride_out_b, stride_out_m, stride_out_k,
        stride_temp_b, stride_temp_m, stride_temp_k,
        BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
    # x: src, (b, n, c)
    # y: query, (b, m, c)
    # out: (b, m, k)
    # temp: (b, m, BLOCK_K)
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    # start pointer in the block
    x_start_ptr = x_ptr + pid_b * stride_xb
    y_start_ptr = y_ptr + pid_b * stride_yb + pid_m * stride_ym
    out_start_ptr = out_ptr + pid_b * stride_out_b + pid_m * stride_out_m
    temp_start_ptr = temp_ptr + pid_b * stride_temp_b + pid_m * stride_temp_m

    # compute k indices in the block for each query point
    mask_c = tl.arange(0, BLOCK_C) < C
    x_ptrs = x_start_ptr + (tl.arange(0, BLOCK_N)[:, None] * stride_xn + tl.arange(0, BLOCK_C)[None, :] * stride_xc)
    y_ptrs = y_start_ptr + tl.arange(0, BLOCK_C) * stride_yc
    temp_ptrs = temp_start_ptr + tl.arange(0, BLOCK_N) * stride_temp_k

    count = 0
    for n in range(tl.cdiv(N, BLOCK_N)):
        # load x: (BLOCK_K, BLOCK_C)
        mask_n = tl.arange(0, BLOCK_N) < N - n * BLOCK_N
        x = tl.load(x_ptrs, mask=mask_n[:, None] & mask_c[None, :], other=0.)  # (BLOCK_K, BLOCK_C)

        # load y: (BLOCK_C, )
        y = tl.load(y_ptrs, mask=mask_c, other=0.)  # (BLOCK_M, BLOCK_C)

        # compute distances
        dists = y - x  # (BLOCK_K, BLOCK_C)
        dists = tl.sum(dists * dists, axis=1)  # (BLOCK_K, )

        # store idx, -1 indicates invalid
        idx = tl.where(dists <= radius * radius, tl.arange(0, BLOCK_N) + n * BLOCK_N, -1)  # (BLOCK_K, )
        tl.store(temp_ptrs, idx)
        _idx_ptr = temp_start_ptr
        _out_ptr = out_start_ptr + count * stride_out_k
        for i in range(BLOCK_N):
            if count < K:
                _idx = tl.load(_idx_ptr)
                if _idx >= 0:
                    tl.store(_out_ptr, _idx)
                    _out_ptr += stride_out_k
                    count += 1
            _idx_ptr += stride_temp_k

        # advance
        x_ptrs += BLOCK_N * stride_xn


def ball_query_triton(src, query, radius, k):
    # src: (b, n, 3)
    # query: (b, m, 3)
    assert k < 64, "k should be less than 64"
    assert src.shape[-1] < 64, "c should be less than 64"
    BLOCK_C = triton.next_power_of_2(src.shape[-1])
    if BLOCK_C <= 8:
        BLOCK_N = 512
    else:
        BLOCK_N = 256
    b, n = src.shape[:2]
    m = query.shape[1]
    device = src.device

    # allocate memory
    out = torch.full((b, m, k), n, dtype=torch.int32, device=device)
    temp = torch.full((b, m, BLOCK_N), -1, dtype=torch.int32, device=device)

    # launch kernel
    ball_query_kernel[(m, b)](
        src, query, out, temp,
        radius,
        n, m, k, src.shape[-1],
        src.stride(0), src.stride(1), src.stride(2),
        query.stride(0), query.stride(1), query.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        temp.stride(0), temp.stride(1), temp.stride(2),
        BLOCK_N, BLOCK_C)

    # out = out.sort(dim=-1).values[:, :, :k]
    out = torch.where(out >= n, out[:, :, [0]], out)

    return out.long()


def ball_query_pytorch(src, query, radius, k):
    # src: (b, n, 3)
    # query: (b, m, 3)
    b, n = src.shape[:2]
    m = query.shape[1]
    dists = torch.cdist(query, src)  # (b, m, n)
    idx = repeat(torch.arange(n, device=src.device), 'n -> b m n', b=b, m=m)
    idx = torch.where(dists > radius, n, idx)
    idx = idx.sort(dim=-1).values[:, :, :k]  # (b, m, k)
    idx = torch.where(idx == n, idx[:, :, [0]], idx)
    return idx
