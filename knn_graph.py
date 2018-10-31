

'''
Create knn graphs from features.
'''
import model
import torch

use_gpu = model.use_gpu
device = model.device #'cuda' if use_gpu else 'cpu'
'''
Input:
-features: embeddings. FloatTensor. Normalized.
-k, number of neighbors to take.
'''
def create_knn_graph(features, k):
    neighbors = torch.LongTensor((features.size(0), k), device=device)
    dist = torch.matmul(features, features.t())
    var, ranks = torch.topk(dist, k=k+1, dim=1)
    return ranks[:, 1:]
    
if __name__ == '__main__':
    num_layers = 5
    k = 5
    features, targets = model.load_features(num_layers)
    create_knn_graph(features, k)
