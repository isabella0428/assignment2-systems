import argparse
import timeit
import numpy.typing as npt
import numpy as np
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

def benchmark(args: argparse.Namespace):
	dataset = np.random.randint(0, args.vocab_size, size=(2 * args.context_length,))
	
	(input_batch, label) = get_batch(
		dataset,
		args.batch_size,
		args.context_length,
		"mps"
	)
	input_batch = input_batch.to("mps")
	label = label.to("mps")
	
	model = BasicsTransformerLM(
		args.vocab_size, args.context_length,
		args.d_model, args.num_layers,
		args.num_heads, args.d_ff,
		args.rope_theta
	).to("mps")

	optimizer = AdamW(model.parameters())

	for _ in range(args.warmup_iters+1):
		forward_start_time = timeit.default_timer()
		result = model.forward(input_batch)
		torch.mps.synchronize()
		forward_finish_time = timeit.default_timer()
		
		loss = cross_entropy(result, label)
		torch.mps.synchronize()

		optimizer.zero_grad()
		torch.mps.synchronize()
		
		backward_start_time = timeit.default_timer()
		loss.backward()
		torch.mps.synchronize()
		backward_finish_time = timeit.default_timer()
		
		optimizer.step()
		torch.mps.synchronize()

	print(f"Average forward time: {np.mean(forward_finish_time - forward_start_time)}")
	print(f"Average backward time: {np.mean(backward_finish_time - backward_start_time)}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Benchmarking script")
	parser.add_argument("--vocab_size", type=int, default=10000)
	parser.add_argument("--context_length", type=int, default=100)
	parser.add_argument("--d_model", type=int)
	parser.add_argument("--num_layers", type=int)
	parser.add_argument("--num_heads", type=int)
	parser.add_argument("--d_ff", type=int)
	parser.add_argument("--rope_theta", type=float, default=0.01)
	parser.add_argument("--batch_size", type=int, default=4)
	parser.add_argument("--warmup_iters", type=int, default=10)

	args = parser.parse_args()
	benchmark(args)
