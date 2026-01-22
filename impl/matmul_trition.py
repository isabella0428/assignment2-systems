import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
    ROWS_TILE_SIZE: tl.constexpr,
    COLS_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    n_row_tiles = tl.num_programs(0)

    a_block_ptr = tl.make_block_ptr(                # ROWS_TILE_SIZE, COLS_TILE_SIZE
        a,
        shape=(M, N),
        strides=(stride_am, stride_an),
        offsets=(pid_m * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, COLS_TILE_SIZE),
        order=(1, 0)
    )

    b_block_ptr = tl.make_block_ptr(
        b,
        shape=(N, K),
        strides=(stride_bn, stride_bk),
        offsets=(0, pid_k * K_TILE_SIZE),
        block_shape=(COLS_TILE_SIZE, K_TILE_SIZE),
        order=(1, 0)
    )

    c_block_ptr = tl.make_block_ptr(
        c,
        shape=(M, K),
        strides=(stride_cm, stride_ck),
        offsets=(pid_m * ROWS_TILE_SIZE, pid_k * K_TILE_SIZE),
        block_shape=(ROWS_TILE_SIZE, K_TILE_SIZE),
        order=(1, 0)
    )

    acc = tl.zeros((ROWS_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
    for n in range(0, N, COLS_TILE_SIZE):
        a_tile = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b_tile = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")   

        # acc += tl.sum(a_tile[:, :, None] * b_tile[None, :, :], axis=1)
        acc += tl.dot(a_tile, b_tile, allow_tf32=False)
        a_block_ptr = a_block_ptr.advance((0, COLS_TILE_SIZE))
        b_block_ptr = b_block_ptr.advance((COLS_TILE_SIZE, 0))
        
    tl.store(c_block_ptr, acc, boundary_check=(0, 1))


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    ROWS_TILE_SIZE = 16
    COLS_TILE_SIZE = 16
    K_TILE_SIZE = 16

    grid = (
        triton.cdiv(M, ROWS_TILE_SIZE),
        triton.cdiv(K, K_TILE_SIZE),
    )
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
        ROWS_TILE_SIZE=ROWS_TILE_SIZE,
        COLS_TILE_SIZE=COLS_TILE_SIZE,
        K_TILE_SIZE=K_TILE_SIZE
    )
