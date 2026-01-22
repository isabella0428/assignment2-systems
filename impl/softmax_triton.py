# import torch
# import triton
# import triton.language as tl

# @triton.jit
# def find_local_m_and_l_kernel(input_ptr, m, l, N, BLOCK_SIZE: tl.constexpr):
#     pid = tl.program_id(0)
#     offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     mask = offs < N

#     ininput_ptrput = input_ptr.to(tl.pointer_type(tl.float32))
#     tile = tl.load(input_ptr + offs, mask=mask, other=float("-inf"))

#     local_m = tl.max(tile)
#     tl.store(m + pid, local_m)

#     local_l = tl.sum(tl.exp(tile - local_m))
#     tl.store(l + pid, local_l)

# @triton.jit
# def find_global_m_and_l_kernel(m_ptr, l_ptr, M_ptr, L_ptr, num_blocks, BLOCK_SIZE: tl.constexpr):
#     offs = tl.arange(0, BLOCK_SIZE)
#     mask = offs < num_blocks

#     m = tl.load(m_ptr + offs, mask=mask, other=-float("inf"))
#     l = tl.load(l_ptr + offs, mask=mask, other=0.0)

#     global_M = tl.max(m, axis=0)
#     global_L = tl.sum(l * tl.exp(m - global_M), axis=0)

#     tl.store(M_ptr, global_M)
#     tl.store(L_ptr, global_L)

# @triton.jit
# def softmax_kernel(input_ptr, output_ptr, M_ptr, L_ptr, N, BLOCK_SIZE: tl.constexpr):
#     pid = tl.program_id(0)
#     offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     mask = offs < N

#     input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
#     output_ptr = output_ptr.to(tl.pointer_type(tl.float32))

#     M = tl.load(M_ptr)
#     L = tl.load(L_ptr)

#     input_tile = tl.load(input_ptr + offs, mask=mask, other=0.0)
#     result = tl.exp(input_tile - M) / L

#     tl.store(output_ptr + offs, result, mask=mask)

# # input, output are tensors on the GPU
# def solve(input: torch.Tensor, output: torch.Tensor, N: int):
#     BLOCK_SIZE = triton.next_power_of_2(N)
#     NUM_BLOCKS = triton.cdiv(N, BLOCK_SIZE)
#     device = torch.device("cuda:0")

#     # local maximum and logits
#     m = torch.empty(NUM_BLOCKS, dtype=torch.float32,device=device)
#     l = torch.empty(NUM_BLOCKS, dtype=torch.float32,device=device)

#     # Global maximum and logits
#     M = torch.empty(1, dtype=torch.float32, device=device)
#     L = torch.empty(1, dtype=torch.float32, device=device)

#     find_local_m_and_l_kernel[(NUM_BLOCKS, )](input, m, l, N, BLOCK_SIZE)
#     find_global_m_and_l_kernel[(NUM_BLOCKS, )]( m, l, M, L, NUM_BLOCKS, BLOCK_SIZE)
    
#     softmax_kernel[(NUM_BLOCKS, )](
#         input, output, M, L, N, 
#         BLOCK_SIZE=BLOCK_SIZE
#     )
#     return output # The result is now in this tensor


import torch
import triton
import triton.language as tl

@triton.jit
def fused_softmax_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Since we want "one" worker to do everything for this vector, 
    # we use pid 0 and loop through the data.
    
    # -----------------------------------------------------------
    # PASS 1: COMPUTE GLOBAL STATS (MAX & SUM)
    # -----------------------------------------------------------
    cur_max = float("-inf")
    cur_sum = 0.0
    
    # Loop over the full length of N in chunks of BLOCK_SIZE
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        tile = tl.load(input_ptr + offsets, mask=mask, other=float("-inf"))
        
        # Get max of current tile
        tile_max = tl.max(tile, axis=0)
        
        # Update global stats using Online Softmax formula
        new_max = tl.maximum(cur_max, tile_max)
        
        # Rescale the previous running sum to the new max
        # and add the contribution from the current tile
        cur_sum = cur_sum * tl.exp(cur_max - new_max) + tl.sum(tl.exp(tile - new_max) * mask, axis=0)
        cur_max = new_max

    # -----------------------------------------------------------
    # PASS 2: COMPUTE FINAL PROBABILITIES AND STORE
    # -----------------------------------------------------------
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        tile = tl.load(input_ptr + offsets, mask=mask, other=float("-inf"))
        
        # Use the final global max and global sum calculated in Pass 1
        probs = tl.exp(tile - cur_max) / cur_sum
        
        # Store result, ensuring we don't write out of bounds
        tl.store(output_ptr + offsets, probs, mask=mask)

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    N = input.numel()
    
    # 1024 is a good balance for T4 registers. 
    # If N is smaller than 1024, next_power_of_2(N) is fine.
    BLOCK_SIZE = min(1024, triton.next_power_of_2(N))
    
    # Use grid=(1,) because this fused version handles the full vector internally
    fused_softmax_kernel[(1,)](
        input, output, N, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output
