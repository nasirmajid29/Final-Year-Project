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
import os

wandb.init(project="test-project", entity="final-year-project")

wandb.config.update({
  "learning_rate": 0.001,
  "optimiser": "Adam",
  "epochs": 30,
  "batch_size": 32,
  "ratio": [0.1, 0.05],
  "radius": [0.05, 0.1]
})


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

        self.sa1_module = GlobalSAModule(MLP([9, 64, 128,]))

        self.mlp = MLP([128, 64, 32, 3], dropout=0.5, norm=None)

    def forward(self, data):
        # print(data.dtype)
        sa0_out = (data.x, data.pos, data.batch)
        # print(np.array(data.batch).shape)
        sa1_out = self.sa1_module(*sa0_out)
        x, pos, batch = sa1_out

        # print(x.dtype)

        return self.mlp(x)


def train(epoch):
    model.train()
    total_loss = 0
    
    for data in train_loader:


        data = data.to(device)
        optimizer.zero_grad()

        pred =  model(data)
        loss = F.mse_loss(pred, data.y) 
        
        total_loss += loss

        
        loss.backward()
        optimizer.step()
        # break


    avg_loss = total_loss / len(train_loader)
    print(f'Epoch: {epoch:03d}, Average Training Loss: {avg_loss:.4f}')

    wandb.log({"training loss": avg_loss})
    wandb.watch(model)
        

def test(loader):
    model.eval()

    # correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            # print("Prediction: ", pred)
        #     predmax = model(data).max(1)[1]
        # correct += predmax.eq(data.y).sum().item()
    return F.mse_loss(pred, data.y) #, correct / len(loader.dataset),  
        


if __name__ == '__main__':

    gpu_usage()
    torch.cuda.empty_cache()
    gpu_usage()
    
    data_loc = "data/reach_target_100eps"
    
    pre_transform = T.NormalizeScale()
    
    point_cloud_data_train = PointDataset(data_loc, "data.pt", True, pre_transform)    
    point_cloud_data_test = PointDataset(data_loc, "data.pt", False, pre_transform)
        

    train_loader = DataLoader(point_cloud_data_train, batch_size=32, shuffle=False, num_workers=6) #shuffle true #batch 32
    test_loader = DataLoader(point_cloud_data_test, batch_size=32, shuffle=False,num_workers=6)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #0.001

    torch.cuda.empty_cache()

    epoch_losses = []
    accuracies = []
    for epoch in range(30): #201
        train(epoch)
        loss = test(test_loader)
        loss = loss.detach().cpu().numpy()
        epoch_losses.append(loss)
        # accuracies.append(test_acc)
        print(f'Epoch: {epoch:03d}, Validation Loss: {loss:.4f}')

        wandb.log({"validation loss": loss})
        wandb.watch(model)

    torch.save(model, "pointnet.pt")