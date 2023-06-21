import torch
import triton
import triton.language as tl

from einops import repeat


def exists(val):
    return val is not None


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def fps_kernel(x_ptr, dist_ptr, out_ptr,
               N, M, C,
               stride_xb, stride_xn, stride_xc,
               stride_dist_b, stride_dist_n,
               stride_out_b, stride_out_m,
               BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
    # x_ptr: (b, n, 3)
    # dist_ptr: (b, n)
    # out_ptr: (b, m) store index of sampled points
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_b = pid // num_pid_n

    start_ptr_xb = x_ptr + pid_b * stride_xb
    start_ptr_dist_b = dist_ptr + pid_b * stride_dist_b
    offs_n = tl.arange(0, BLOCK_N)
    offs_c = tl.arange(0, BLOCK_C)

    idx = tl.load(out_ptr + pid_b * stride_out_b)  # current idx
    for i in range(1, M):
        mask_c = offs_c < C
        xi = tl.load(start_ptr_xb + idx * stride_xn + offs_c[None, :] * stride_xc,
                     mask=mask_c[None, :], other=0.)  # (1, 4)

        x_ptrs = start_ptr_xb + (offs_n[:, None] * stride_xn + tl.arange(0, BLOCK_C)[None, :] * stride_xc)  # (n, 4)
        dist_ptrs = start_ptr_dist_b + offs_n * stride_dist_n
        # initialize memory to store max dist and idx in each block
        max_dist_n = tl.zeros((BLOCK_N,), dtype=tl.float32)
        idx_n = tl.zeros((BLOCK_N,), dtype=tl.int32)
        for n in range(0, tl.cdiv(N, BLOCK_N)):
            # inner loop: given idx, compute dists
            mask = offs_n < N - n * BLOCK_N
            x = tl.load(x_ptrs, mask=mask[:, None] & mask_c[None, :], other=0.)  # (n, 4)
            # compute distance
            dist = tl.sqrt(tl.sum((x - xi) * (x - xi), axis=1))

            # tl.atomic_min(dist_ptrs, dist, mask=mask)  # save to DRAM
            dists = tl.load(dist_ptrs, mask=mask, other=-float('inf'))
            dists = tl.minimum(dist, dists)
            tl.store(dist_ptrs, dists, mask=mask)  # save to DRAM

            # find max dist and index: get the lowest index if multiple maximum values
            _max_dist = tl.max(dists, axis=0)
            _idx = tl.where(dists == _max_dist, tl.arange(0, BLOCK_N), BLOCK_N)
            _idx = tl.min(_idx, axis=0) + n * BLOCK_N
            # _idx = tl.argmax(dists, axis=0) + n * BLOCK_N

            # reserve max dist and index
            masked_dist_n = tl.where(tl.arange(0, BLOCK_N) == n, _max_dist, 0.)
            masked_idx_n = tl.where(tl.arange(0, BLOCK_N) == n, _idx, 0)

            # update max dist
            max_dist_n += masked_dist_n
            idx_n += masked_idx_n

            # advance pointers
            x_ptrs += BLOCK_N * stride_xn
            dist_ptrs += BLOCK_N * stride_dist_n

        # update idx
        # i_max_dist_n = tl.argmax(max_dist_n, axis=0)
        idx_n = tl.where(max_dist_n == tl.max(max_dist_n, axis=0), idx_n, N)
        idx = tl.min(idx_n, axis=0)

        # update idx
        tl.store(out_ptr + pid_b * stride_out_b + i * stride_out_m, idx)


def fps_triton(x, n_sample, start_idx: int = None):
    # x: (b, n, 3)
    # n_sample: number of points to sample
    # return: (b, n_sample) store index of sampled points
    b, n, c = x.shape
    if exists(start_idx):
        out = torch.full((b, n_sample), start_idx, dtype=torch.int32, device=x.device)
    else:
        out = torch.randint(0, n, (b, n_sample), dtype=torch.int32, device=x.device)

    dists = torch.full((b, n), float('inf'), dtype=torch.float32, device=x.device)
    stride_xb, stride_xn, stride_xc = x.stride()
    stride_out_b, stride_out_m = out.stride()
    stride_dist_b, stride_dist_n = dists.stride()

    grid = lambda meta: (b * triton.cdiv(n, meta['BLOCK_N']),)
    fps_kernel[grid](x, dists, out,
                     n, n_sample, c,
                     stride_xb, stride_xn, stride_xc,
                     stride_dist_b, stride_dist_n,
                     stride_out_b, stride_out_m,
                     BLOCK_C=triton.next_power_of_2(c))

    return out.long()


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
