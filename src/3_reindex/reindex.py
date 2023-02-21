import torch

name = "articles"
path = "../../data"

def transform_unique(t):
    unique = {e: i for i, e in enumerate(t.unique().sort()[0].numpy())}
    t = t.clone()
    t.apply_(lambda x: unique[x])
    return t, unique

observations = torch.load(f'{path}/{name}.pt')
t0 = observations[:, 0]
t1 = observations[:, 1]

X, transform = transform_unique(observations)
torch.save((X, transform), f'{path}/{name}_prepared.pt')