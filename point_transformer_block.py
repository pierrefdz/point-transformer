# Point Transformer block
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

class PointTransformerBlock(nn.Module):
    def __init__(self, d_feature, d_trans, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_feature, d_trans)
        self.phi = nn.Linear(d_trans, d_trans, bias=False) # queries
        self.psi = nn.Linear(d_trans, d_trans, bias=False) # keys
        self.alpha = nn.Linear(d_trans, d_trans, bias=False) # values
        self.k = k
        self.delta = nn.Sequential(nn.Linear(3, d_trans), nn.ReLU(), nn.Linear(d_trans, d_trans))
        self.gamma = nn.Sequential(nn.Linear(d_trans, d_trans), nn.ReLU(), nn.Linear(d_trans, d_trans))
        self.fc2 = nn.Linear(d_trans, d_feature)
        
    # coords: b x n x 3, features: b x n x f (f=d_feature)
    def forward(self, coords, features):
        dists = square_distance(coords, coords)  # b x n x n
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_coords = index_points(coords, knn_idx) # b x n x k x 3
        
        residual = features # b x n x f

        x = self.fc1(features) # b x n x d_trans

        q = self.phi(x) # b x n x d_trans
        k = index_points(self.psi(x), knn_idx) # b x n x k x d_trans
        v = index_points(self.alpha(x), knn_idx) # b x n x k x d_trans

        pos_enc = self.delta(coords[:, :, None] - knn_coords)  # b x n x k x d_trans
        
        attn = self.gamma(q[:, :, None] - k + pos_enc) # b x n x k x d_trans
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x d_trans
        
        res = torch.einsum('bnkd,bnkd->bnd', attn, v + pos_enc) # b x n x d_trans
        res = self.fc2(res) + residual # b x n x f
        return res, attn


if __name__ == '__main__':
    layer = PointTransformerBlock(d_feature=64, d_trans=4, k=16)
    B, N, d_feature = 16, 1024, 64
    coords = torch.rand((B, N, 3))
    features = torch.rand((B, N, d_feature))
    print('New feature shape : ', layer.forward(coords, features)[0].shape)
    print('Attention shape : ', layer.forward(coords, features)[1].shape)
