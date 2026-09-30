# An embedding layer is a learnable lookup 
# table that maps discrete token indices 
# into continuous, dense vectors.
import torch
import torch.nn as nn


class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings : int , embedding_dim : int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        
    def forward(self,indices):
        return self.weight[indices]







emb = MyEmbedding(10,4)
idx = torch.tensor([0,3,7])

print('output shape: ', emb(idx).shape)
print('matches manual: ', torch.equal(emb(idx)[0], emb.weight[0]))