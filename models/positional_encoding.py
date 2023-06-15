# # Inspired by https://pytorch.org/tutorials/beginner/transformer_tutorial.html

# import torch
# import torch.nn as nn
# import math

# class PositionalEncoding(nn.Module):
    
#     def __init__(self, d_model: int, max_len: int = 5000):
#         super().__init__()

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).unsqueeze(1)
#         # print(position.shape)
#         # print(div_term.shape)
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         Arguments:
#             x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#         """
#         x[:3] = x[:3] + self.pe[:x.size(0)]
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, max_range=1.0, d_model=3):
        """
        Initialize the PositionalEncoding module.

        Args:
            max_range (float): Maximum range of coordinates.
            d_model (int): Dimensionality of the positional encoding.
        """
        super(PositionalEncoding, self).__init__()
        self.max_range = max_range
        self.d_model = d_model

    def forward(self, point_cloud):
        """
        Apply positional encoding to a point cloud.

        Args:
            point_cloud (torch.Tensor): Point cloud with shape (N, 6), where [:, :3] are the coordinates.

        Returns:
            torch.Tensor: Point cloud with positional encoding, shape (N, 6 + d_model).
        """
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        N = point_cloud.size(0)
        coordinates = point_cloud[:, :3]
        colours = point_cloud[:, 3:]
        encoding = torch.zeros(N, self.d_model, device=device)

        for i in range(N):
            for j in range(self.d_model // 2):
                pos_val = i / (10000 ** ((2 * j) / self.d_model))
                encoding[i, 2 * j] = torch.sin(torch.tensor(pos_val, dtype=torch.float32, device=device))
                encoding[i, 2 * j + 1] = torch.cos(torch.tensor(pos_val, dtype=torch.float32, device=device))

        encoding = self.max_range * encoding
        out = torch.cat((coordinates, encoding, colours), dim=1)
        return out.to(device)