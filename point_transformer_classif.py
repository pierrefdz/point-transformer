# Point Transformer Network for classification
#
# Sources: https://github.com/qq456cvb/Point-Transformers 
#         https://github.com/yzheng97/Point-Transformer-Cls
#         https://github.com/lucidrains/point-transformer-pytorch

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from point_transformer_block import PointTransformerBlock
from transition_down import TransitionDown

class PointTransformerClassif(nn.Module):
    def __init__(self, n_point=1024, k=16, n_class=40, d_feature=6, n_block=4, d_trans=16) -> None:
        super().__init__()
        self.mlp_first = nn.Sequential(
            nn.Linear(d_feature, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = PointTransformerBlock(32, d_trans, k)
        self.transition_downs = nn.ModuleList(
            [TransitionDown(n_point // 4**(i + 1), k, (32*2**(i + 1)) // 2, 32*2**(i + 1)) for i in range(n_block)]
        )
        self.point_transformers = nn.ModuleList(
            [PointTransformerBlock(32*2**(i + 1), d_trans, k) for i in range(n_block)]
        )
            
        self.mlp_last = nn.Sequential(
            nn.Linear(32 * 2 ** n_block, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_class)
        )
        self.n_block = n_block
    
    def forward(self, x):
        coords = x[..., :3]
        features = self.transformer1(coords, self.mlp_first(x))[0]
        for i in range(self.n_block):
            coords, features = self.transition_downs[i](coords, features)
            features = self.point_transformers[i](coords, features)[0]
        res = self.mlp_last(features.mean(1))
        return res


if __name__ == '__main__':
    B, N, input_dim = 16, 1024, 6
    model = PointTransformerClassif()
    features = torch.rand((B, N, input_dim))
    print(model.forward(features).shape)
