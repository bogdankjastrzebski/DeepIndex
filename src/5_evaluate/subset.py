
import torch
from tqdm import tqdm

print("Loading data...")
X, transform = torch.load('../../data/articles_prepared.pt')

print("Loading model...")
sd, _ = torch.load('../4_train/models/0/model_001677253747.pt', map_location=torch.device('cpu'))

print("Extracting parameters...")
W = sd['embeddings.W.weight']
b = sd['embeddings.b.weight']

print("Selecting subset...")
torch.manual_seed(25022023)
ind = torch.randint(3549980, size=(1000,)).sort()[0]

print("Creating subset...")
subset = torch.cat([X[(X==i).any(1)] for i in tqdm(ind)])

print("Saving...")
torch.save(subset, 'subset.pt')

