"""Matrix transpose - 4 optimization stages + cutile.

Usage:
    python -m kernels.transpose                                   # verify + benchmark
    ncu --set full python -m kernels.transpose --profile          # ncu profile (skip 2 for cutile, 3-6 for v1-v4)
"""

import sys
import numpy as np
import cupy as cp
import cuda.tile as ct

M, N = 4096, 4096
TILE = 32

# =============================================================================
# v1: naive — no shared memory, uncoalesced writes
# =============================================================================
v1_src = r"""
extern "C" __global__
void transpose_v1(const float* __restrict__ a, float* __restrict__ out, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
        out[col * M + row] = a[row * N + col];
}
"""

# =============================================================================
# v2: shared memory — bank conflicts on transposed read
# =============================================================================
v2_src = r"""
extern "C" __global__
void transpose_v2(const float* __restrict__ a, float* __restrict__ out, int M, int N) {
    __shared__ float tile[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int src_row = blockIdx.y * blockDim.y + ty, src_col = blockIdx.x * blockDim.x + tx;
    if (src_row < M && src_col < N) tile[ty][tx] = a[src_row * N + src_col];
    __syncthreads();
    int dst_row = blockIdx.x * blockDim.x + ty, dst_col = blockIdx.y * blockDim.y + tx;
    if (dst_row < N && dst_col < M) out[dst_row * M + dst_col] = tile[tx][ty];
}
"""

# =============================================================================
# v3: + padding — eliminates bank conflicts
# =============================================================================
v3_src = r"""
extern "C" __global__
void transpose_v3(const float* __restrict__ a, float* __restrict__ out, int M, int N) {
    __shared__ float tile[TILE][TILE + 1];  // +1 padding avoids bank conflicts
    int tx = threadIdx.x, ty = threadIdx.y;
    int src_row = blockIdx.y * blockDim.y + ty, src_col = blockIdx.x * blockDim.x + tx;
    if (src_row < M && src_col < N) tile[ty][tx] = a[src_row * N + src_col];
    __syncthreads();
    int dst_row = blockIdx.x * blockDim.x + ty, dst_col = blockIdx.y * blockDim.y + tx;
    if (dst_row < N && dst_col < M) out[dst_row * M + dst_col] = tile[tx][ty];
}
"""

# =============================================================================
# v4: 128 threads — improve theoretical occupancy and thread coarsening
# =============================================================================
v4_src = r"""
extern "C" __global__
void transpose_v4(const float* __restrict__ a, float* __restrict__ out, int M, int N) {
    __shared__ float tile[TILE][TILE + 1];
    int tx = threadIdx.x, ty = threadIdx.y;
    for (int row = ty; row < 32; row += 4) {
        int src_row = blockIdx.y * 32 + row, src_col = blockIdx.x * 32 + tx;
        if (src_row < M && src_col < N) tile[row][tx] = a[src_row * N + src_col];
    }
    __syncthreads();
    for (int row = ty; row < 32; row += 4) {
        int dst_row = blockIdx.x * 32 + row, dst_col = blockIdx.y * 32 + tx;
        if (dst_row < N && dst_col < M) out[dst_row * M + dst_col] = tile[tx][row];
    }
}
"""

# =============================================================================
# cutile
# =============================================================================
@ct.kernel
def _cutile_kernel(a, out, tile_m: ct.Constant[int], tile_n: ct.Constant[int]):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    a_tile = ct.load(a, index=(pid_m, pid_n), shape=(tile_m, tile_n))
    out_tile = ct.transpose(a_tile)
    ct.store(out, index=(pid_n, pid_m), tile=out_tile)


# =============================================================================
# Registry: name → (kernel, block)
# =============================================================================
def _compile(src, name):
    return cp.RawModule(code=src, options=(f"-DTILE={TILE}",)).get_function(name)

KERNELS = {
    "v1:naive":   {"kernel": _compile(v1_src, "transpose_v1"), "block": (TILE, TILE), "tile": (TILE, TILE)},
    "v2:shmem":   {"kernel": _compile(v2_src, "transpose_v2"), "block": (TILE, TILE), "tile": (TILE, TILE)},
    "v3:padding": {"kernel": _compile(v3_src, "transpose_v3"), "block": (TILE, TILE), "tile": (TILE, TILE)},
    "v4:128t":    {"kernel": _compile(v4_src, "transpose_v4"), "block": (TILE, 4),    "tile": (TILE, TILE)},
}


def transpose(a_dev, out_dev, name):
    m, n = a_dev.shape
    if name == "cutile":
        grid = (ct.cdiv(m, TILE), ct.cdiv(n, TILE), 1)
        ct.launch(cp.cuda.get_current_stream(), grid,
                  _cutile_kernel, (a_dev, out_dev, TILE, TILE))
    else:
        cfg = KERNELS[name]
        tile_m, tile_n = cfg["tile"]
        grid = ((n + tile_n - 1) // tile_n, (m + tile_m - 1) // tile_m, 1)
        cfg["kernel"](grid, cfg["block"], (a_dev, out_dev, np.int32(m), np.int32(n)))


# =============================================================================
# CLI
# =============================================================================
def verify():
    a = cp.random.randn(M, N, dtype=cp.float32)
    expected = a.T
    for name in ["cutile"] + list(KERNELS):
        out = cp.empty((N, M), dtype=cp.float32)
        transpose(a, out, name)
        diff = float(cp.abs(out - expected).max())
        if diff >= 1e-5:
            print(f"  FAIL  {name}  max_diff={diff:.2e}")
            return
    print("Pass")



if __name__ == "__main__":
    if "--profile" in sys.argv:
        a = cp.array(np.random.randn(M, N).astype(cp.float32))
        out = cp.empty((N, M), dtype=cp.float32)
        for name in ["cutile"] + list(KERNELS):
            transpose(a, out, name)
        cp.cuda.Stream.null.synchronize()
    else:
        verify()
