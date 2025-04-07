import torch

def ToTensor(self, x):
			return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
