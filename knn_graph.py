

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
Returns:
-tensor of indices of nearest points
'''
def create_knn_graph(features, k):
    #neighbors = torch.LongTensor((features.size(0), k), device=device)
    dist = torch.matmul(features, features.t())
    var, ranks = torch.topk(dist, k=k+1, dim=1)
    
    return ranks[:, 1:]

'''
Compute density. Density as measured by edges/vertices within clusters, vs in-between clusters.
Input:
-features: embeddings. FloatTensor. Normalized.
-k, number of neighbors to take.
'''
def compute_density(features, targets, k):
    #tally of edges within clusters
    density_tally = torch.zeros(10, device=device, dtype=torch.long)
    
    rank_labels = targets[create_knn_graph(features, k)]
    targets_exp = targets.view(-1,1).expand_as(rank_labels)
    correct = (rank_labels==targets_exp).sum(dim=1)
    print('intra-class {}, total {} '.format(correct.sum(), k*targets.size(0) ))
    
    density_tally = density_tally.scatter_add(0, targets, correct.view(-1))
    print('density_tally {}'.format(density_tally))
    return density_tally
    
if __name__ == '__main__':
    num_layers_l = [5]
    num_layers_l = [15, 18, 21, 25, 28]
    num_layers_l = [28]
    
    for num_layers in num_layers_l:
        k = 5
        features, targets = model.load_features(num_layers)
        #print('targets.size {}'.format(targets.size()))
        #create_knn_graph(features, k)
        print('{} layers peeled off:'.format(num_layers))
        with torch.no_grad():
            compute_density(features, targets, k)
