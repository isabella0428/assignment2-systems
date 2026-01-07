import argparse
import timeit
import numpy.typing as npt
import numpy as np
import torch

from cs336_basics.model import BasicsTransformerLM
from impl.toy_model import ToyModel
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

def benchmark(args: argparse.Namespace):
	batch = torch.tensor(np.random.randint(0, 100, size=(args.in_features, )))
	label = torch.tensor(1)
	
	batch = batch.to("cuda", dtype=torch.float32)
	label = label.to("cuda")
	
	with torch.autocast(device_type="cuda",dtype=torch.float16):
		model = ToyModel(
			args.in_features, args.out_features
		).to("cuda")

	optimizer = AdamW(model.parameters())

	for _ in range(args.warmup_iters+1):
		forward_start_time = timeit.default_timer()
		result = model.forward(batch)
		torch.cuda.synchronize()
		forward_finish_time = timeit.default_timer()
		
		loss = cross_entropy(result, label)
		torch.cuda.synchronize()

		optimizer.zero_grad()
		torch.cuda.synchronize()
		
		backward_start_time = timeit.default_timer()
		loss.backward()
		torch.cuda.synchronize()
		backward_finish_time = timeit.default_timer()
		
		optimizer.step()
		torch.cuda.synchronize()

	print(f"Average forward time: {np.mean(forward_finish_time - forward_start_time)}")
	print(f"Average backward time: {np.mean(backward_finish_time - backward_start_time)}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Benchmarking script")
	parser.add_argument("--in_features", type=int, default=10)
	parser.add_argument("--out_features", type=int, default=1)
	parser.add_argument("--warmup_iters", type=int, default=0)

	args = parser.parse_args()
	benchmark(args)
