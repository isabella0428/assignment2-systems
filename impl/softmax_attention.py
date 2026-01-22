import torch
import triton
import triton.language as tl
import math

@triton.jit
def matrix_mutmul(
    A, B, C,
    M, N, K,
    stride_ar, stride_ac,
    stride_br, stride_bc,
    stride_cr, stride_cc,
    ROW_TILE_SIZE: tl.constexpr, COL_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr):

    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    a_block_ptr = tl.make_block_ptr(
        A,
        shape=(M, N),
        strides=(stride_ar, stride_ac),
        order=(0, 1),
        offsets=(pid_m * ROW_TILE_SIZE, 0),
        block_shape=(ROW_TILE_SIZE, COL_TILE_SIZE)
    )

    b_block_ptr = tl.make_block_ptr(
        B,
        shape=(N, K),
        strides=(stride_br, stride_bc),
        order=(0, 1),
        offsets=(0, pid_k * K_TILE_SIZE),
        block_shape = (COL_TILE_SIZE, K_TILE_SIZE)
    )

    acc = tl.zeros((ROW_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
    for i in range(0, N, COL_TILE_SIZE):
        a_tile = tl.load(a_block_ptr, boundary_check=(0, 1))
        b_tile = tl.load(b_block_ptr, boundary_check=(0, 1))

        acc = tl.dot(a_tile, b_tile, acc, out_dtype=tl.float32, allow_tf32=False)
        a_block_ptr = a_block_ptr.advance((0, COL_TILE_SIZE))
        b_block_ptr = b_block_ptr.advance((COL_TILE_SIZE, 0))

    c_block_ptr = tl.make_block_ptr(
        C,
        shape=(M, K),
        strides=(stride_cr, stride_cc),
        order=(0, 1),
        offsets=(pid_m * ROW_TILE_SIZE, pid_k * K_TILE_SIZE),
        block_shape = (ROW_TILE_SIZE, K_TILE_SIZE)
    )
    tl.store(c_block_ptr, acc, boundary_check=(0, 1))

@triton.jit
def softmax_kernel(input_ptr, output_ptr, M, N, ROW_TILE_SIZE: tl.constexpr, COL_TILE_SIZE: tl.constexpr):
    pid_r = tl.program_id(0)

    cur_max = tl.full((ROW_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    cur_logits_sum= tl.zeros((ROW_TILE_SIZE,), dtype=tl.float32)

    input_block_ptr = tl.make_block_ptr(
        input_ptr,
        shape=(M, N),
        strides=(N, 1),
        order=(0, 1),
        offsets=(pid_r * ROW_TILE_SIZE, 0),
        block_shape = (ROW_TILE_SIZE, COL_TILE_SIZE)
    )

    row_offsets = pid_r * ROW_TILE_SIZE + tl.arange(0, ROW_TILE_SIZE)
    row_mask = row_offsets < M
    for i in range(0, N, COL_TILE_SIZE):
        col_offsets = tl.arange(0, COL_TILE_SIZE)
        col_mask = i + col_offsets < N
        mask = row_mask[:, None] & col_mask[None, :]
        tile = tl.load(input_block_ptr, boundary_check=(0, 1), padding_option="zero")
        tile = tl.where(mask, tile, float("-inf"))

        # new_max = tl.max(tile, axis=1)
        # tile_exp = tl.where(mask, tl.exp(tile - new_max[:, None]), 0.0)
        # cur_logits_sum = cur_logits_sum * tl.exp(cur_max - new_max) + tl.sum(tile_exp, axis=1)
        # cur_max = new_max
        # 1. Find the max of the current tile
        tile_max = tl.max(tile, axis=1)
        
        # 2. Find the "New Global Max" (The max of everything seen so far)
        next_max = tl.maximum(cur_max, tile_max)
        
        # 3. Rescale the previous sum and the current tile sum to the New Global Max
        # Note: (cur_max - next_max) and (tile - next_max) are now always <= 0
        cur_logits_sum = cur_logits_sum * tl.exp(cur_max - next_max) + \
                         tl.sum(tl.exp(tile - next_max[:, None]), axis=1)
        
        # 4. Update the global max for the next iteration
        cur_max = next_max

        input_block_ptr = input_block_ptr.advance((0, COL_TILE_SIZE))

    input_block_ptr = tl.make_block_ptr(
        input_ptr,
        shape=(M, N),
        strides=(N, 1),
        order=(0, 1),
        offsets=(pid_r * ROW_TILE_SIZE, 0),
        block_shape = (ROW_TILE_SIZE, COL_TILE_SIZE)
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(M, N),
        strides=(N, 1),
        order=(0, 1),
        offsets=(pid_r * ROW_TILE_SIZE, 0),
        block_shape = (ROW_TILE_SIZE, COL_TILE_SIZE)
    )

    for i in range(0, N, COL_TILE_SIZE):
        col_offsets = tl.arange(0, COL_TILE_SIZE)
        col_mask = i + col_offsets < N
        mask = row_mask[:, None] & col_mask[None, :]
        tile = tl.load(input_block_ptr, boundary_check=(0, 1), padding_option="zero")
        tile = tl.where(mask, tile, float("-inf"))
        
        output_tile = tl.exp(tile - cur_max[:, None]) / cur_logits_sum[:, None]
        tl.store(output_block_ptr, output_tile, boundary_check=(0, 1))

        input_block_ptr = input_block_ptr.advance((0, COL_TILE_SIZE))
        output_block_ptr = output_block_ptr.advance((0, COL_TILE_SIZE))

# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int
):
    device = Q.device
    QK_T = torch.empty(M, N, device=device)
    ROW_TILE_SIZE = 16
    COL_TILE_SIZE = 16
    N_TILE_SIZE = 16
    D_TILE_SIZE = 16

    # Use .item() if M and K are tensors, or ensure they are passed as ints
    grid = (triton.cdiv(int(M), ROW_TILE_SIZE), triton.cdiv(int(N), N_TILE_SIZE))
    matrix_mutmul[grid](
        Q, K, QK_T,
        M, d, N,        # (M, d) * (d, N)
        d, 1,
        1, d,
        N, 1,
        ROW_TILE_SIZE, COL_TILE_SIZE, N_TILE_SIZE
    )

    before_softmax_result = QK_T / math.sqrt(d)        # (M, N)
    softmax_result = torch.empty(M, N, device=device)
    softmax_kernel[(triton.cdiv(M, ROW_TILE_SIZE), )](
        before_softmax_result, softmax_result, M, N, ROW_TILE_SIZE, COL_TILE_SIZE
    )

    grid = (triton.cdiv(int(M), ROW_TILE_SIZE), triton.cdiv(int(d), D_TILE_SIZE))
    matrix_mutmul[grid](
        softmax_result, V, output,
        M, N, d,        # M * N, N * d
        N, 1,
        d, 1,
        d, 1,
        ROW_TILE_SIZE, COL_TILE_SIZE, D_TILE_SIZE
    )
    return output
