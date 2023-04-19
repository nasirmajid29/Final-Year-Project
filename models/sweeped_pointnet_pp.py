import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius

from GPUtil import showUtilization as gpu_usage
import matplotlib.pyplot as plt
from point_dataset import PointDataset
import numpy as np

import wandb
import pyvista
import matplotlib.pyplot as plt


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        # visualise_pc_radius(x.detach().numpy(), idx.detach().numpy(), self.r)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]

        # print("X dtype", x.dtype)
        # print("Pos dtype", pos.dtype)
        # print("Edge dtype", edge_index.dtype)


        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.1, 0.05, MLP([9, 32, 64]))
        self.sa2_module = SAModule(0.05, 0.1, MLP([64 + 3, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512]))

        self.mlp = MLP([512, 256, 32, 3], dropout=0.5, norm=None)

    def forward(self, data):
        # print(data.dtype)
        sa0_out = (data.x, data.pos, data.batch)
        # print(np.array(data.batch).shape)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        # print(x.dtype)

        return self.mlp(x)

gpu_usage()
torch.cuda.empty_cache()
gpu_usage()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

def test(loader):
    model.eval()

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
    return F.mse_loss(pred, data.y)

def train():
    wandb.init(project="test-project", entity="final-year-project")
    print("Config, ", wandb.config)

    data_loc = "/vol/bitbucket/nm219/data/reach_target_10eps"
    pre_transform = T.NormalizeScale()
    
    point_cloud_data_train = PointDataset(data_loc, "data.pt", True, pre_transform)
    point_cloud_data_test = PointDataset(data_loc, "data.pt", False, pre_transform)
    

    train_loader = DataLoader(point_cloud_data_train, batch_size=wandb.config.batch_size, shuffle=False, num_workers=6) 
    test_loader = DataLoader(point_cloud_data_test, batch_size=wandb.config.batch_size, shuffle=False,num_workers=6)


    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    # Train the model
    model.train()
    total_loss = 0
    for epoch in range(wandb.config.epochs):
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = F.mse_loss(pred, data.y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        loss = test(test_loader)
        loss = loss.detach().cpu().numpy()
    
        wandb.log({"validation loss": loss})
        
        
        avg_loss = total_loss / len(train_loader)
        wandb.log({'training_loss': avg_loss, 'epoch': epoch})
        total_loss = 0

    return avg_loss

sweep_config = {
        'method': 'bayes',
        'metric': {
            'goal': 'minimize',
            'name': 'validation loss'
            },
        
        'parameters': {
            'batch_size': {
                "distribution": "int_uniform",
                "max": 64,
                "min": 16
            },
            "epochs":{
                "distribution": "int_uniform",
                "max": 60,
                "min": 5
            },
            "learning_rate": {
                "distribution": "uniform",
                "max": 0.01,
                "min": 0.0001
            },
            'ratio': {
                'distribution': 'uniform',
                'min': 0.05,
                'max': 0.5
            },
            'radius': {
                'distribution': 'uniform',
                'min': 0.01,
                'max': 0.5
            }
        }
    }

    # wandb.agent("khptq5vx", function = simple_pointnet_pp.main(), count=1)

# Run the sweep
# sweep_id = wandb.sweep(sweep_config, project="test-project", entity="final-year-project")

wandb.agent("a735kg94", function=train, project="test-project", entity="final-year-project", count=100)
