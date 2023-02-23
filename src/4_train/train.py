import numpy as np
import torch
import time
from tqdm import trange, tqdm
import os

class SparseMatrix(torch.utils.data.Dataset):
    
    def __init__(self, ls, rs, vs):
        self.ls = ls 
        self.rs = rs
        self.vs  = vs
        self.len = vs.shape[0]
        
    def __getitem__(self, idx):
        return self.ls[idx], self.rs[idx], self.vs[idx]
    
    def __len__(self):
        return self.len 
    
class NegativeSampler(torch.utils.data.Dataset):
    
    def __init__(self, ls, rs, vs):
        self.unique_ls, self.counts_ls = ls.unique(return_counts=True)
        self.unique_rs, self.counts_rs = rs.unique(return_counts=True)
        self.unique_vs, self.counts_vs = vs.unique(return_counts=True)
        self.len = vs.shape[0]
        self.p_ls = self.counts_ls/self.len
        self.p_rs = self.counts_rs/self.len
        self.p_vs = self.counts_vs/self.len
        
    def __getitem__(self, idx):
        l = torch.multinomial(self.p_ls, 1)[0]
        r = torch.multinomial(self.p_rs, 1)[0]
        v = torch.multinomial(self.p_vs, 1)[0]
        return l, r, v
    
    def __len__(self):
        return self.len
        
class NegativeSampler(torch.utils.data.Dataset):
    def __init__(self, ls, rs, vs):
        self.ls = ls
        self.rs = rs
        self.vs = vs
        self.len = vs.shape[0]
        
    def __getitem__(self, idx):
        return self.ls[idx], self.rs[idx], self.vs[idx]
    
    def __len__(self):
        return self.len 
    
    def resample(self):
        self.ls = self.ls[torch.randperm(self.ls.shape[0])]
        self.rs = self.rs[torch.randperm(self.rs.shape[0])]
        self.vs = self.vs[torch.randperm(self.vs.shape[0])]
        return 
    
class EmbeddingWithBias(torch.nn.Module):
    
    def __init__(self, n, embedding_dim):
        super(EmbeddingWithBias, self).__init__()
        
        self.embedding_dim = embedding_dim 
        self.shape = (n, embedding_dim)
        self.W = torch.nn.Embedding(n, embedding_dim, sparse=True)
        self.b = torch.nn.Embedding(n, 1, sparse=True)
        
    def forward(self, idx):
        return self.W(idx), self.b(idx)

class MatrixFactorization(torch.nn.Module):
    
    def __init__(self, n, m, embedding_dim): 
        super(MatrixFactorization, self).__init__()
        self.embedding_dim = embedding_dim
        self.shape = (n, m, embedding_dim)
        self.L = EmbeddingWithBias(n, embedding_dim)
        self.R = EmbeddingWithBias(m, embedding_dim)
        
    def forward(self, ls, rs):
        
        LW, Lb = self.L(ls)
        RW, Rb = self.R(rs)
        
        return (LW * RW).sum(1) + Lb + Rb
    
class SymmetricMatrixFactorization(torch.nn.Module):
    
    def __init__(self, n, embedding_dim):
        super(SymmetricMatrixFactorization, self).__init__()
        self.embedding_dim = embedding_dim
        self.shape = (n, embedding_dim)
        self.embeddings = EmbeddingWithBias(n, embedding_dim)
        
    def forward(self, ls, rs):
        LW, Lb = self.embeddings(ls)
        RW, Rb = self.embeddings(rs)
        
        return (LW * RW).sum(1) + Lb + Rb
    
def create_model(n, dim, seed=26122022):
    torch.manual_seed(seed)
    m = SymmetricMatrixFactorization(n, dim).cuda()
    opt = torch.optim.SparseAdam(m.parameters(), lr=0.001)
    losses = {
        'train': {
            'pos' : [],
            'neg' : []
        },
        'valid': {
            'pos' : [], 
            'neg' : []
        }
    }
    return m, opt, losses


def prepare_dataloaders(X, batch_size, lengths, seed=27122022):
    torch.manual_seed(seed)

    ds_pos = SparseMatrix(X[:,0], X[:,1], torch.ones(X.shape[0]))
    ds_train_pos, ds_valid_pos, ds__test_pos = torch.utils.data.random_split(ds_pos, lengths)
    
    ds_neg = NegativeSampler(X[:,0], X[:,1], torch.zeros(X.shape[0]))
    ds_train_neg, ds_valid_neg, ds__test_neg = torch.utils.data.random_split(ds_neg, lengths)
    
    ds_train = torch.utils.data.ConcatDataset([ds_train_pos, ds_train_neg])
    ds_valid = torch.utils.data.ConcatDataset([ds_valid_pos, ds_valid_neg])
    ds__test = torch.utils.data.ConcatDataset([ds__test_pos, ds__test_neg])
    
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True) 
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=batch_size, shuffle=True, drop_last=True) 
    dl__test = torch.utils.data.DataLoader(ds__test, batch_size=batch_size, shuffle=False, drop_last=False) 
    
    return dl_train, dl_valid, dl__test, ds_neg

def train_model(m, opt, dl_train, dl_valid, losses, EPOCHS=1, seed=26012023):
    
    L_train_pos_prev = 0
    L_train_neg_prev = 0
    L_valid_pos_prev = 0
    L_valid_neg_prev = 0
    
    print(f" {'itr': >6} {'ech': >6} {'cur': >6} {'tp': >6} {'vp': >6} {'tn': >6} {'vn': >6} ")
    
    for epoch in (pbar := trange(EPOCHS)):
        
        L_train_pos = 0
        L_train_neg = 0
        L_valid_pos = 0
        L_valid_neg = 0
        
        for iteration, (ls, rs, vs) in enumerate(dl_train):
            
            vs = vs == 1
            
            opt.zero_grad()

            p = m(ls, rs).sigmoid()
            
            L_pos = -(0.001+p[vs]).log().mean()
            L_neg = -(1.001-p[~vs]).log().mean()
            L = L_pos + L_neg

            L.backward()
            opt.step()
            
            L_train_pos += L_pos.item() 
            L_train_neg += L_neg.item() 
            
            pbar.set_description(f" {iteration: 6d} {epoch: 6d} {L.item():.03f} {L_train_pos_prev:.03f} {L_valid_pos_prev:.03f} {L_train_neg_prev:.03f} {L_valid_neg_prev:.03f} ")
            
        
        with torch.no_grad():
            for ls, rs, vs in dl_valid:
                
                vs = vs == 1
                p = m(ls, rs).sigmoid()
                L_pos = -(0.001+p[vs]).log().mean()
                L_neg = -(1.001-p[~vs]).log().mean()
                L_valid_pos += L_pos.item()
                L_valid_neg += L_neg.item()
        
        L_train_pos_prev = L_train_pos/len(dl_train)
        L_train_neg_prev = L_train_neg/len(dl_train)
        L_valid_pos_prev = L_valid_pos/len(dl_valid)
        L_valid_neg_prev = L_valid_neg/len(dl_valid)
        
        losses['train']['pos'].append(L_train_pos_prev)
        losses['train']['neg'].append(L_train_neg_prev)
        losses['valid']['pos'].append(L_valid_pos_prev)
        losses['valid']['neg'].append(L_valid_neg_prev)
        
        pbar.set_description(f" {L_train_pos_prev:.03f} {L_valid_pos_prev:.03f} {L_train_neg_prev:.03f} {L_valid_neg_prev:.03f} ")


        torch.save(
            (
                m.state_dict(),
                losses
            ),
            f"{model_path}/model_{round(time.time()):012d}.pt"
        )
        
def evaluate_model(m, dl_test):
    L_test_pos = 0
    
    with torch.no_grad():
        for ls, rs, vs in dl_valid:
            v = v == 1  
            p = m(ls, rs).sigmoid()
            L_pos = -p[vs].log().mean()
            L_neg = -(1-p[~vs]).log().mean()
            L_test_pos += L_pos.item()
            L_test_neg += L_neg.item()
       
    
    
# --------------------------------------------------------------------------------- # 
#                                  Training                                         # 
# --------------------------------------------------------------------------------- #

path = '../../data'
name = 'articles_prepared'
model_path = "./models/0"

assert len(os.listdir(model_path)) == 0

X, transform = torch.load(f'{path}/{name}.pt')

X = X.cuda()

n = X.shape[0]
m = n // 100

dl_train, dl_valid, dl_test, ds_neg = prepare_dataloaders(X, 2*1024, [n - 2*m, m, m]) # [0.9, 0.05, 0.05])

print(len(dl_train))
print(len(dl_valid))
print(len(dl_test))

NUMBER_UNIQUE = X.unique().shape[0]

m, opt, losses = create_model(NUMBER_UNIQUE, 256)


torch.save((m.state_dict(), losses), f"{model_path}/model_{round(time.time()):012d}.pt")

train_model(m, opt, dl_train, dl_valid, losses, EPOCHS=10)

torch.save((m.state_dict(), losses), f"{model_path}/model_{round(time.time()):012d}.pt")


