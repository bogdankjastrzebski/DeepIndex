import torch

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
        
        LW, Lb = self.L(ls).squeeze()
        RW, Rb = self.R(rs).squeeze()
        
        return (LW * RW).sum(1) + Lb + Rb
    
class SymmetricMatrixFactorization(torch.nn.Module):
    
    def __init__(self, n, embedding_dim):
        super(SymmetricMatrixFactorization, self).__init__()
        self.embedding_dim = embedding_dim
        self.shape = (n, embedding_dim)
        self.embeddings = EmbeddingWithBias(n, embedding_dim)
        
    def forward(self, ls, rs):
        LW, Lb = self.embeddings(ls).squeeze()
        RW, Rb = self.embeddings(rs).squeeze()
        
        return (LW * RW).sum(1) + Lb + Rb

