

'''
Various computation utilities for features.
'''
import torch
import os

'''
Computes expectation of pairwise squared dist, E[n(x_i)^2],
E[n(y_i)^2], 2E[xiyi]
Input: 
-features: features/embeddings for a given layer
-
'''
def pairwise_mx(num_layers):
    
    #compute E for each class
    embs_path = 'data/train_embs_norm{}.pth'.format(num_layers)
    targets_path = 'data/train_classes_norm{}.pth'.format(num_layers)
    features = torch.load(embs_path)
    targets = torch.load(targets_path)
    features = features.to('cpu')
    targets = targets.to('cpu')
    #e.g. torch.Size([50000, 512]) targets sz torch.Size([50000])
    #print('features sz {} targets sz {}'.format(features.size(), targets.size()))
    
    ones_vec = torch.ones(targets.size(0))
    
    #compute the individual norms and mixed E
    
    #get the counts of each class
    counts_vec = torch.zeros(10)
    counts_vec = counts_vec.scatter_add(0, targets, ones_vec)
    
    #separate into classes with scatteradd
    features2 = features.norm(p=2, dim=1)
    norms_vec = torch.zeros(10)
    #
    
    #for computing norms
    norms_vec = norms_vec.scatter_add(0, targets, features2)
    #print('counts_vec {} {}'.format(norms_vec, counts_vec))
    norms_vec /= counts_vec
    
    features1 = torch.zeros(10, features.size(1))
    
    targets_exp = targets.view(-1,1).expand_as(features)
    #for computing mixed E
    features1 = features1.scatter_add(0, targets_exp, features)
    features1 /= counts_vec.view(10,1)
    
    #compute E
    features1 = torch.matmul(features1, features1.t())
    #rows
    norms_i = norms_vec.unsqueeze(0).t().repeat(1, 10)
    norms_j = norms_vec.unsqueeze(0).repeat(10, 1)
    
    
    mx = norms_i + norms_j - 2*features1
    
    return mx
    
if __name__ == '__main__':
    num_layers = 18
    mx = pairwise_mx(num_layers)
    print(num_layers)
    
