import torch

_, losses = torch.load(f"./models/0/model_001677111668.pt")

torch.save(losses, './losses.pt')


