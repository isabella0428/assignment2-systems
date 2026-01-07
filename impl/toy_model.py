import torch

class ToyModel(torch.nn.Module):
	def __init__(self, in_features: int, out_features: int):
		super().__init__()
		self.fc1 = torch.nn.Linear(in_features, 10, bias=False)
		self.ln = torch.nn.LayerNorm(10)
		self.fc2 = torch.nn.Linear(10, out_features, bias=False)
		self.relu = torch.nn.ReLU()

	def forward(self, x):
		temp = self.fc1(x)
		print(f"self.fc1(x): {temp.dtype}")

		x = self.relu(temp)
		print(f"self.relu(temp): {x.dtype}")
		x = self.ln(x)
		print(f"self.ln(x): {x.dtype}")
		x = self.fc2(x)
		print(f"self.f2(x): {x.dtype}")
		return x