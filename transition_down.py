# Transition Down block
#
# Sources: https://github.com/qq456cvb/Point-Transformers
#         https://github.com/yzheng97/Point-Transformer-Cls
#         https://github.com/lucidrains/point-transformer-pytorch

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def square_distance(src, dst):
    """
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, K]
    Output:
        new_points:, indexed points data, [B, S, K, C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

def farthest_point_sample_batch(coords, n_sample):
    """
    Input:
        coords: pointcloud data, [B, N, 3]
        n_sample: number of samples
    Return:
        centroids: sampled pointcloud index, [B, n_sample]
    """
    device = coords.device
    B, N, C = coords.shape
    centroids = torch.zeros(B, n_sample, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(n_sample):
        centroids[:, i] = farthest
        centroid = coords[batch_indices, farthest, :].view(B, 1, 3)  
        dist = torch.sum((coords - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids.to(device)

class TransitionDown(nn.Module):
    def __init__(self, output_n, k, input_dim, output_dim, mlp_type='linear') -> None:
        """
        output_n: target number of points after transition down
        k: number of neighbors to max pool the new features from
        input_dim: dimension of input features for each point
        outut_dim: dimension of output features for each point
        """
        super().__init__()
        self.output_n = output_n
        self.k = k
        self.mlp_type = mlp_type
        if mlp_type == 'linear':
            self.mlp = nn.Sequential(
                nn.Conv1d(input_dim, output_dim, 1),
                nn.BatchNorm1d(output_dim),
                nn.ReLU()
            )
        elif mlp_type == 'conv':
            self.convs = nn.ModuleList([
                nn.Conv2d(input_dim, output_dim, 1),
                nn.Conv2d(output_dim, output_dim, 1)
            ])
            self.bns = nn.ModuleList([
                nn.BatchNorm2d(output_dim),
                nn.BatchNorm2d(output_dim)
            ])
        
    def forward(self, coords, features):
        """
        Input:
            coords: input points position data, [B, N, 3]
            features: input points data, [B, N, D]
        Return:
            new_coords: sampled points position data, [B, S, 3]
            new_features: new points feature data, [B, S, D']
        """
        fps_idx = farthest_point_sample_batch(coords, self.output_n) # B x output_n
        torch.cuda.empty_cache()
        new_coords = index_points(coords, fps_idx) # B x output_n x 3
        torch.cuda.empty_cache()
        dists = square_distance(new_coords, coords)  # B x output_n x N
        idx = dists.argsort()[:, :, :self.k]  # B x output_n x k
        torch.cuda.empty_cache()
        grouped_coords = index_points(coords, idx) # B x output_n x k x 3
        torch.cuda.empty_cache()

        B, N, D = features.shape

        if self.mlp_type == 'linear':
            new_features = features.transpose(1,2) # B x D x N
            new_features = self.mlp(new_features) # B x D' x N
            new_features = new_features.transpose(1,2) # B x N x D'
            grouped_features = index_points(new_features, idx) # B x output_n x k x D'
            new_features, _ = torch.max(grouped_features, 2) # B x output_n x D'

        elif self.mlp_type == 'conv':
            new_features = index_points(features, idx) # B x output_n x k x D
            new_features = new_features.permute(0, 3, 2, 1) # B x D x k x output_n
            for i, conv in enumerate(self.convs):
                bn = self.bns[i]
                new_features =  F.relu(bn(conv(new_features))) # B x D' x k x output_n
            new_features, _ = torch.max(new_features, 2) # B x D' x output_n
            new_features = new_features.transpose(1,2) # B x output_n x D'

        return new_coords, new_features


if __name__ == '__main__':
    B, N, input_dim = 16, 1024, 64
    layer = TransitionDown(output_n=N//2, k=16, input_dim=input_dim, output_dim=input_dim*2)
    coords = torch.rand((B, N, 3))
    features = torch.rand((B, N, input_dim))
    print('Input number of points: ', N)
    print('New coordinates shape (batch size, number of points, coordinates dimension) : ', 
            layer.forward(coords, features)[0].shape)
    print('New features shape (batch size, number of points, features dimension) : ', layer.forward(coords, features)[1].shape)
