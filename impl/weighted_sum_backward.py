import triton
_

@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr,                  # Input
    grad_output_ptr,                   # Grad output
    grad_x_ptr, partial_grad_weight_ptr, # Grad input
    stride_xr, stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr, stride_gxd,
    stride_gwb, stride_gwd,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
	row_tile_idx = tl.program_id(0)
	n_row_tiles = tl.num_programs(0)

	x_block_ptr = tl.make_block_ptr(
		x_ptr,
		shape=(NUM_ROWS, D),
		strides=(stride_xr, stride_xd),
		offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
		block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
		order=(1, 0)
	)

	weight_block_ptr = tl.make_block_ptr(
		weight_ptr,
		shape=(D),
		strides=(stride_wd,),
		offsets=(0,),
		block_shape=(D_TILE_SIZE),
		order=(0,)
	)

	grad_x_block_ptr = tl.make_block_ptr(
		grad_x_ptr,
		shape=(NUM_ROWS, D),
		strides=(stride_gxr, stride_gxd),
		offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
		block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
		order=(1, 0)
	)

	grad_output_block_ptr = tl.make_block_ptr(
		grad_output_ptr,
		shape=(NUM_ROWS,),
		strides=(stride_gr),
		offsets=(0,),
		block_shape=(0,),
		order=(1)
	)

	# We need to add all these on axis=0 to get final grad w
	partial_grad_weight_block_ptr = tl.make_block_ptr(
		partial_grad_weight_ptr,
		shape=(n_row_tiles, D), strides=(stride_gwb, stride_gwd),
		offsets=(n_row_tiles, 0),
		block_shape=(1, D_TILE_SIZE),
		order=(1, 0),
	)

	for i in range(triton.ceil_div(D, D_TILE_SIZE)):
		weight = tl.load(
			weight_block_ptr,				# (D_TILE_SIZE)
			boundary_check=(0),
			padding_option="zero"
		)

		grad_output = tl.load(				# (ROW_TILE_SIZE)
			grad_output_block_ptr,
			boundary_check=(0),
			padding_option="zero"
		)

		grad_x_row = grad_output[:, None] * weight[None, :]	# (ROW_TILE_SIZE, D_TILE_SIZE)
		tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))


		x_row = tl.load(
			x_block_ptr,
			boundary_check=(0, 1),
			padding_option="zero"
		)

		partial_grad_weight = tl.sum(x_row, axis=0, keep_dims=True) * grad_output[:, None]
		tl.store(partial_grad_weight_block_ptr, partial_grad_weight, boundary_check=(1,))

		# Move the pointers to the next tile along D
		x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
		weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
		partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
		grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))





